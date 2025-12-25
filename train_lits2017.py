import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets.dataset_lits2017 import LitsRandomGenerator, LitsValTransform
from engine_lits2017 import *

from models.HBFormer import HBFormer

import os
import sys
import csv
import pandas as pd
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
from configs.config_setting_lits2017 import setting_config

import warnings
warnings.filterwarnings("ignore")



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LiTS2017五折交叉验证训练')
    parser.add_argument('--fold', type=int, default=None, help='当前训练的fold编号 (0-4)，不指定则使用配置文件中的current_fold')
    parser.add_argument('--all_folds', action='store_true', help='是否训练所有5个folds')
    parser.add_argument('--save_interval', type=int, default=None, help='权重保存间隔(epoch)，不指定则使用配置文件中的save_interval')
    return parser.parse_args()

def main(config, fold=None, save_interval=None):
    # 如果指定了save_interval参数，则更新配置
    if save_interval is not None:
        setting_config.set_save_interval(save_interval)
        # 重新创建配置实例以获取更新后的save_interval
        config = setting_config()
    
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
    
    # Define class names for LITS dataset (liver, tumor)
    class_names = ['liver', 'tumor']
    
    # Training CSV columns
    train_columns = ['epoch', 'lr', 'loss', 'avg_dice', 'miou'] + [f'dice_{cls}' for cls in class_names]
    
    # Validation CSV columns  
    val_columns = ['epoch', 'avg_dice', 'avg_hd95', 'miou'] + [f'dice_{cls}' for cls in class_names] + [f'hd95_{cls}' for cls in class_names]
    
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
    
    # 记录当前fold信息
    print(f'当前使用的fold: {config.current_fold}')
    if fold is not None:
        logger.info(f'开始训练 Fold {config.current_fold}/{5-1}')
        logger.info(f'工作目录: {config.work_dir}')
    
    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]
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
    # 传递fold参数到数据集
    train_dataset = config.datasets(
        data_dir=config.data_path, 
        split="train",
        transform=LitsRandomGenerator(output_size=[config.input_size_h, config.input_size_w]),
        fold=config.current_fold if config.cross_validation else None,
        num_folds=5,  # 直接使用5作为fold数
        cross_validation=config.cross_validation
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if config.distributed else None
    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size//gpus_num if config.distributed else config.batch_size, 
                                shuffle=(train_sampler is None),
                                pin_memory=False,  # 改为False避免GPU传输问题
                                num_workers=0,  # 暂时设为0避免多进程问题
                                sampler=train_sampler)

    val_dataset = config.datasets(
        data_dir=config.data_path, 
        split="val",
        transform=LitsValTransform(output_size=[config.input_size_h, config.input_size_w]),
        fold=config.current_fold if config.cross_validation else None,
        num_folds=5,  # 直接使用5作为fold数
        cross_validation=config.cross_validation
    )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if config.distributed else None
    val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=False,  # 改为False避免GPU传输问题
                                num_workers=0,  # 暂时设为0避免多进程问题
                                sampler=val_sampler,
                                drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config

    # LiTS2017仅使用HBFormer
    class Args:
        def __init__(self):
            self.dataset = config.datasets_name

    args = Args()
    model = HBFormer(
        args=args,
        img_size=(config.input_size_h, config.input_size_w),
        feature_size=model_cfg.get('feature_size', 48),
    )

    # 如果有预训练权重，尝试加载
    if model_cfg.get('load_ckpt_path') and os.path.exists(model_cfg['load_ckpt_path']):
        try:
            print("尝试加载HBFormer预训练权重...")
            checkpoint = torch.load(model_cfg['load_ckpt_path'], map_location='cpu')

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            model_dict = model.state_dict()
            compatible_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict)
            print(f"成功加载 {len(compatible_dict)}/{len(model_dict)} 个HBFormer权重参数")
        except Exception as e:
            print(f"HBFormer预训练权重加载失败: {e}")
            print("将使用随机初始化权重")

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
        
        # 过滤掉FLOPs计算相关的keys（与test脚本保持一致）
        model_state_dict = checkpoint['model_state_dict']
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if not key.endswith('.total_ops') and not key.endswith('.total_params'):
                filtered_state_dict[key] = value
        
        model.module.load_state_dict(filtered_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        print(log_info)
        logger.info(log_info)
    elif config.resume_training and not os.path.exists(resume_model):
        print(f'#----------Resume training enabled but checkpoint not found: {resume_model}----------#')
        print('#----------Starting fresh training----------#')
    else:
        print('#----------Starting fresh training----------#')

    step = 0
    print('#----------Training Configuration----------#')
    print(f'训练总epoch数: {config.epochs}')
    print(f'验证间隔: 每{config.val_interval}个epoch验证一次')
    print(f'权重保存间隔: 每{config.save_interval}个epoch保存一次权重')
    print(f'批量大小: {config.batch_size}')
    print(f'学习率: {config.lr}')
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()

        step += 1
        if config.distributed:
            train_sampler.set_epoch(epoch)

        # Training
        train_metrics = train_one_epoch_lits(
            train_loader,
            model,
            criterion, 
            optimizer, 
            scheduler,
            epoch, 
            logger, 
            config, 
            scaler=scaler if config.amp else None
        )

        # Record training metrics
        train_record = {
            'epoch': epoch,
            'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            'loss': train_metrics['loss'],
            'avg_dice': train_metrics['avg_dice'],
            'miou': train_metrics['miou']
        }
        
        for i, cls_name in enumerate(class_names):
            if i < len(train_metrics['dice_per_class']):
                train_record[f'dice_{cls_name}'] = train_metrics['dice_per_class'][i]

        with open(train_csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=train_columns)
            writer.writerow(train_record)

        # Validation
        if epoch % config.val_interval == 0:
            val_metrics = val_one_epoch_lits(
                val_dataset,
                val_loader,
                model,
                epoch, 
                logger,
                config,
                outputs,
                val_or_test=False
            )

            # Record validation metrics
            val_record = {
                'epoch': epoch,
                'avg_dice': val_metrics['avg_dice'],
                'avg_hd95': val_metrics['avg_hd95'],
                'miou': val_metrics['miou']
            }
            
            for i, cls_name in enumerate(class_names):
                if i < len(val_metrics['dice_per_class']):
                    val_record[f'dice_{cls_name}'] = val_metrics['dice_per_class'][i]
                if i < len(val_metrics['hd95_per_class']):
                    val_record[f'hd95_{cls_name}'] = val_metrics['hd95_per_class'][i]

            with open(val_csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=val_columns)
                writer.writerow(val_record)

            # Save best model
            if val_metrics['avg_dice'] > (1 - min_loss):
                min_loss = 1 - val_metrics['avg_dice']  # Convert dice to loss-like metric
                min_epoch = epoch
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'fold': config.current_fold if config.cross_validation else None,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': min_loss,
                    'min_loss': min_loss,
                    'min_epoch': min_epoch,
                    'dice_score': val_metrics['avg_dice']
                }, os.path.join(checkpoint_dir, 'best.pth'))

        # Save latest model every epoch
        torch.save({
            'epoch': epoch,
            'fold': config.current_fold if config.cross_validation else None,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_metrics['loss'],
            'min_loss': min_loss,
            'min_epoch': min_epoch,
        }, os.path.join(checkpoint_dir, 'latest.pth'))

        # 定期保存完整模型权重 - 根据save_interval参数控制
        if epoch % config.save_interval == 0:
            # 保存完整的checkpoint信息，便于恢复训练
            checkpoint_data = {
                'epoch': epoch,
                'fold': config.current_fold if config.cross_validation else None,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_metrics['loss'],
                'train_dice': train_metrics['avg_dice'],
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'config': {
                    'save_interval': config.save_interval,
                    'network': config.network,
                    'fold': config.current_fold
                }
            }
            
            # 如果该epoch也进行了验证，添加验证指标
            if epoch % config.val_interval == 0 and 'val_metrics' in locals():
                checkpoint_data.update({
                    'val_dice': val_metrics['avg_dice'],
                    'val_hd95': val_metrics['avg_hd95'],
                    'val_miou': val_metrics['miou']
                })
            
            save_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}_dice{train_metrics["avg_dice"]:.2f}.pth')
            torch.save(checkpoint_data, save_path)
            
            log_msg = f'定期保存模型权重 (每{config.save_interval}个epoch): {save_path}'
            print(log_msg)
            logger.info(log_msg)

    fold_info = f"Fold {config.current_fold}" if config.cross_validation else "Standard split"
    logger.info(f'{fold_info} Training completed! Best epoch: {min_epoch}, Best dice: {1-min_loss:.4f}')
    
    return 1-min_loss  # 返回最佳dice score

def train_all_folds(save_interval=None):
    """训练所有5个folds"""
    all_results = []
    
    for fold in range(5):  # 直接使用5作为fold数
        print(f"\n{'='*50}")
        print(f"开始训练 Fold {fold}/{5-1}")
        print(f"{'='*50}")
        
        # 设置当前fold
        setting_config.set_fold(fold)
        config = setting_config()
        
        try:
            best_dice = main(config, fold, save_interval)
            all_results.append({
                'fold': fold,
                'best_dice': best_dice,
                'work_dir': config.work_dir
            })
            print(f"Fold {fold} 完成，最佳Dice: {best_dice:.4f}")
        except Exception as e:
            print(f"Fold {fold} 训练失败: {e}")
            all_results.append({
                'fold': fold,
                'best_dice': 0.0,
                'work_dir': config.work_dir,
                'error': str(e)
            })
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("五折交叉验证结果汇总:")
    print(f"{'='*50}")
    
    valid_results = [r for r in all_results if 'error' not in r]
    if valid_results:
        dice_scores = [r['best_dice'] for r in valid_results]
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        for result in all_results:
            if 'error' in result:
                print(f"Fold {result['fold']}: 失败 - {result['error']}")
            else:
                print(f"Fold {result['fold']}: Dice = {result['best_dice']:.4f}")
        
        print(f"\n平均Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        
        # 保存汇总结果
        summary_file = f"results/LiTS2017_5fold_summary.csv"
        os.makedirs("results", exist_ok=True)
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fold', 'best_dice', 'work_dir'])
            writer.writeheader()
            for result in valid_results:
                writer.writerow(result)
        
        print(f"结果已保存到: {summary_file}")
    else:
        print("所有fold都训练失败!")

if __name__ == '__main__':
    args = parse_args()
    
    if args.all_folds:
        # 训练所有5个folds
        if args.save_interval is not None:
            print(f"设置所有fold的权重保存间隔为: {args.save_interval} epoch")
        train_all_folds(args.save_interval)
    else:
        # 训练指定的fold
        if args.fold is not None:
            # 如果命令行指定了fold，使用命令行参数
            setting_config.set_fold(args.fold)
            print(f"使用命令行指定的fold: {args.fold}")
        else:
            # 如果命令行没有指定fold，使用配置文件中的设置
            print(f"使用配置文件中的fold: {setting_config.current_fold}")
        
        if args.save_interval is not None:
            print(f"使用命令行指定的权重保存间隔: {args.save_interval} epoch")
        
        config = setting_config()
        main(config, args.fold if args.fold is not None else config.current_fold, args.save_interval) 