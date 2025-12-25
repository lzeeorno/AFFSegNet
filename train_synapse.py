import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset import RandomGenerator
from engine_synapse import *

from models.DWSegNet import DWSegNet
from models.HBFormer import HBFormer
from models.vmunet.vmunet import VMUNet

import os
import sys
import csv
import pandas as pd
import numpy as np
import torch.nn.functional as F
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting_synapse import setting_config

import warnings
warnings.filterwarnings("ignore")


def seed_worker(worker_id):
    """设置DataLoader worker的随机种子"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    # Create CSV files for recording training and validation metrics
    train_csv_file = os.path.join(config.work_dir, 'train_record.csv')
    val_csv_file = os.path.join(config.work_dir, 'val_record.csv')
    
    # Define organ names for Synapse dataset (8 organs + background)
    organ_names = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 
                   'esophagus', 'liver', 'stomach', 'aorta']
    
    # Training CSV columns
    train_columns = ['epoch', 'lr', 'loss', 'avg_dice', 'miou'] + [f'dice_{organ}' for organ in organ_names]
    
    # Validation CSV columns  
    val_columns = ['epoch', 'avg_dice', 'avg_hd95', 'miou'] + [f'dice_{organ}' for organ in organ_names] + [f'hd95_{organ}' for organ in organ_names]
    
    # 如果不是恢复训练，清空并重新创建CSV文件
    if not config.resume_training:
        # 创建新的CSV文件并写入表头
        with open(train_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_columns)
            writer.writeheader()
            
        with open(val_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=val_columns)
            writer.writeheader()
    else:
        # 恢复训练时，如果CSV文件不存在则创建
        if not os.path.exists(train_csv_file):
            with open(train_csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=train_columns)
                writer.writeheader()
                
    if not os.path.exists(val_csv_file):
        with open(val_csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=val_columns)
            writer.writeheader()

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    gpus_type, gpus_num = torch.cuda.get_device_name(), torch.cuda.device_count()
    print(f"GPU Device: {gpus_type}, GPU Count: {gpus_num}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    
    if config.distributed:
        print('#----------Start DDP----------#')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.manual_seed_all(config.seed)
        config.local_rank = torch.distributed.get_rank()





    print('#----------Preparing dataset----------#')
    train_dataset = config.datasets(
        base_dir=config.data_path, list_dir=config.list_dir, split="train",
        transform=transforms.Compose([RandomGenerator([config.input_size_h, config.input_size_w])]))
    
    # 创建slice-by-slice验证数据集，与训练保持一致
    class ValTransform(object):
        def __init__(self, output_size=[256, 256]):
            self.output_size = output_size
            
        def __call__(self, sample):
            image, label = sample['image'], sample['label']
            
            # 确保数据类型一致
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            
            # 调整大小 - 与RandomGenerator保持一致
            if image.shape != tuple(self.output_size):
                from scipy.ndimage.interpolation import zoom
                x, y = image.shape
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
            # 伪HDR三通道预处理 - 与RandomGenerator完全一致
            image_tensor = torch.from_numpy(image)
            
            # 创建三个通道：原始、增强对比度、平滑
            channel1 = image_tensor  # 原始图像
            channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
            channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                                   kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
            
            # 合并为三通道
            image = torch.stack([channel1, channel2, channel3], dim=0)
            label = torch.from_numpy(label)
            
            sample = {'image': image, 'label': label.long()}
            return sample
    
    # 使用slice-by-slice验证数据集
    val_dataset_slice = config.datasets(
        base_dir=config.data_path,  # 使用相同的数据路径
        list_dir=config.list_dir,
        split="test_slice",  # 使用测试slice列表
        transform=ValTransform([config.input_size_h, config.input_size_w])
    )
    
    # 原始volume验证数据集（保留作为对比）
    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation slice dataset size: {len(val_dataset_slice)}")
    print(f"Validation volume dataset size: {len(val_dataset)}")
    
    train_sampler = DistributedSampler(train_dataset) if config.distributed else None
    train_loader = DataLoader(train_dataset, 
                              batch_size=config.batch_size, 
                              num_workers=config.num_workers, 
                              pin_memory=True, 
                              sampler=train_sampler, 
                              shuffle=(train_sampler is None), 
                              worker_init_fn=seed_worker)
    
    # Slice验证数据加载器
    val_loader_slice = DataLoader(val_dataset_slice,
                                 batch_size=config.batch_size,
                                shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True)
    
    # Volume验证数据加载器（保留）
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    
    # 根据配置选择模型
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        if model_cfg['load_ckpt_path'].split('/')[-1] == 'vmamba_small_e238_ema.pth':
            model.load_from()
        else:
            print("当前模型的状态字典:")
            print(model.state_dict().keys())
            print('---------------------------')

            continue_training = torch.load(model_cfg['load_ckpt_path'])
            print("继续训练的状态字典:")
            print(continue_training.keys())

            # 加载模型参数
            model_dict = model.state_dict()

            # 只保留在模型字典中存在的参数
            new_dict = {k: v for k, v in continue_training.items() if k in model_dict.keys()}

            # 更新模型的状态字典
            model_dict.update(new_dict)

            # 打印未加载的键
            not_loaded = [k for k in model_dict.keys() if k not in new_dict.keys()]
            print(f"未加载的键: {not_loaded}")

            # 加载更新后的状态字典
            model.load_state_dict(model_dict)

    elif config.network == 'DWSegNet':
        # 创建args对象，DWSegNet需要这个参数
        class Args:
            def __init__(self):
                self.dataset = config.datasets_name  # 使用配置中的数据集名称
        
        args = Args()
        
        model = DWSegNet(
            args=args,
            img_size=(config.input_size_h, config.input_size_w),
            feature_size=model_cfg.get('feature_size', 48),
            use_boundary_refinement=model_cfg.get('use_boundary_refinement', True)
        )
        
        # 如果有预训练权重，尝试加载
        if model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
            try:
                print("尝试加载预训练权重...")
                checkpoint = torch.load(model_cfg['load_ckpt_path'], map_location='cpu')
                
                # 处理不同的checkpoint格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # 尝试加载兼容的权重
                model_dict = model.state_dict()
                compatible_dict = {}
                
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f"成功加载 {len(compatible_dict)}/{len(model_dict)} 个权重参数")
                
            except Exception as e:
                print(f"预训练权重加载失败: {e}")
                print("将使用随机初始化权重")
    
    elif config.network == 'HBFormer':
        # 创建args对象，AFFSegNet需要这个参数
        class Args:
            def __init__(self):
                self.dataset = config.datasets_name  # 使用配置中的数据集名称
        
        args = Args()
        
        model = HBFormer(
            args=args,
            img_size=(config.input_size_h, config.input_size_w),
            feature_size=model_cfg.get('feature_size', 48)
        )
        
        # 如果有预训练权重，尝试加载
        if model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
            try:
                print("尝试加载预训练权重...")
                checkpoint = torch.load(model_cfg['load_ckpt_path'], map_location='cpu')
                
                # 处理不同的checkpoint格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # 尝试加载兼容的权重
                model_dict = model.state_dict()
                compatible_dict = {}
                
                for k, v in state_dict.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        compatible_dict[k] = v
                
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f"成功加载 {len(compatible_dict)}/{len(model_dict)} 个权重参数")
                
            except Exception as e:
                print(f"预训练权重加载失败: {e}")
                print("将使用随机初始化权重")

    else:
        raise ValueError(f'不支持的网络类型: {config.network}。请选择: vmunet, DWSegNet, AFFSegNet')

    model = model.cuda()
    cal_params_flops(model, config.input_size_w, logger)

    if config.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])




    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    # 根据配置决定是否恢复训练
    if config.resume_training and os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch = saved_epoch + 1  # 修复：从下一个epoch开始，而不是累加
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        best_dice = checkpoint.get('best_dice', 0.0)  # 恢复best_dice，如果不存在则设为0

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}, best_dice: {best_dice:.4f}'
        logger.info(log_info)
    else:
        if not config.resume_training:
            print('#----------Starting fresh training (resume_training=False)----------#')
        else:
            print('#----------No checkpoint found, starting fresh training----------#')





    print('#----------Training----------#')
    best_dice = 0.0
    patience = 10  # 早停耐心值
    patience_counter = 0  # 计数器
    
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) if config.distributed else None

        # 记录当前epoch开始时的学习率（在scheduler.step()之前）
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        train_metrics = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        if train_metrics['loss'] < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = train_metrics['loss']
            min_epoch = epoch

        # Initialize validation metrics
        val_metrics = None

        if epoch % config.val_interval == 0:
            # 使用slice-by-slice验证（推荐）
            from engine_synapse import val_one_epoch_slice
            val_metrics = val_one_epoch_slice(
                val_dataset_slice,
                val_loader_slice,
                model,
                epoch,
                logger,
                config,
                test_save_path=outputs,
                val_or_test=False
            )
            
            # 可选：同时进行volume验证作为对比
            # if epoch % (config.val_interval * 3) == 0:  # 每3个验证周期进行一次volume验证
            #     logger.info("进行volume验证作为对比...")
            #     val_metrics_volume = val_one_epoch(
            #         val_dataset,
            #         val_loader,
            #         model,
            #         epoch,
            #         logger,
            #         config,
            #         test_save_path=outputs,
            #         val_or_test=False
            #     )
            #     logger.info(f"Volume验证结果: avg_dice={val_metrics_volume['avg_dice']:.4f}, slice验证结果: avg_dice={val_metrics['avg_dice']:.4f}")
            
            # Save best model based on dice score
            if val_metrics['avg_dice'] > best_dice:
                best_dice = val_metrics['avg_dice']
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best_dice.pth'))
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1  # 增加计数器

            # 打印早停信息
            logger.info(f'Best validation dice: {best_dice:.4f}, current: {val_metrics["avg_dice"]:.4f}, patience: {patience_counter}/{patience}')
        
        # Record training metrics to CSV
        train_record = {
            'epoch': epoch,
            'lr': current_lr,
            'loss': train_metrics['loss'],
            'avg_dice': train_metrics['avg_dice'],
            'miou': train_metrics['miou']
        }
        
        # Add per-organ dice scores for training
        for i, organ in enumerate(organ_names):
            train_record[f'dice_{organ}'] = train_metrics['dice_per_class'][i]
            
        with open(train_csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_columns)
            writer.writerow(train_record)
        
        # Record validation metrics to CSV if validation was performed
        if val_metrics is not None:
            val_record = {
                'epoch': epoch,
                'avg_dice': val_metrics['avg_dice'],
                'avg_hd95': val_metrics['avg_hd95'],
                'miou': val_metrics['miou']
            }
            
            # Add per-organ dice and hd95 scores for validation
            for i, organ in enumerate(organ_names):
                val_record[f'dice_{organ}'] = val_metrics['dice_per_class'][i]
                val_record[f'hd95_{organ}'] = val_metrics['hd95_per_class'][i]
                
            with open(val_csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=val_columns)
                writer.writerow(val_record)

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': train_metrics['loss'],
                'best_dice': best_dice,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth')) 

        # 检查是否需要早停
        if val_metrics is not None and patience_counter >= patience:
            print(f'Early stopping at epoch {epoch} (no improvement for {patience} validation epochs)')
            logger.info(f'Early stopping at epoch {epoch} (no improvement for {patience} validation epochs)')
            break

        # 定期保存模型权重
        if epoch % config.save_interval == 0:
            save_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(model.module.state_dict(), save_path)
            logger.info(f'定期保存模型权重: {save_path}')

    print('#----------Training Completed----------#')
    print(f'Best loss: {min_loss:.4f} at epoch {min_epoch}')
    print(f'Best dice: {best_dice:.4f}')
    print('Use test_synapse.py for final evaluation')


if __name__ == '__main__':
    config = setting_config()
    main(config)       