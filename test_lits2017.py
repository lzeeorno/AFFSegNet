#!/usr/bin/env python3
"""
LITS2017数据集测试脚本（PNG格式）
支持AFFSegNet和DWSegNet模型切换
支持五折交叉验证模式
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt
from datetime import datetime
import time
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config_setting_lits2017 import setting_config as config
from engine_lits2017 import test_single_slice_lits, calculate_dice_per_class, calculate_miou, calculate_hd95
from datasets.dataset_lits2017 import LiTS2017_dataset

from models.HBFormer import HBFormer

import warnings
warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体避免乱码
plt.rcParams['axes.unicode_minus'] = False

class TestTransform:
    """测试时的数据预处理，必须与训练时保持一致"""
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size
    
    def __call__(self, sample):
        image = sample['image']
        label = sample['label'] 
        case_name = sample.get('case_name', '')
        
        # 确保图像尺寸正确
        if image.shape[:2] != self.output_size:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 关键：伪HDR三通道预处理（与训练时完全一致）
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        # 创建三个通道：原始、增强对比度、平滑
        channel1 = image_tensor  # 原始图像
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
        
        # 合并为三通道
        image = torch.stack([channel1, channel2, channel3], dim=0)
        
        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long(), 'case_name': case_name}

def parse_args():
    """解析命令行参数，使用配置文件的默认值"""
    parser = argparse.ArgumentParser(description='LITS2017数据集测试')
    parser.add_argument('--fold', type=int, default=config.test_config['fold'], 
                       help=f'测试的fold编号 (0-4)，默认: {config.test_config["fold"]}')
    parser.add_argument('--model', type=str, default=config.test_config['model'], 
                       choices=['HBFormer'], 
                       help=f'使用的模型，默认: {config.test_config["model"]}')
    parser.add_argument('--weights', type=str, default=config.test_config['weights'], 
                       help=f'模型权重路径，默认: {config.test_config["weights"]}')
    parser.add_argument('--test_mode', type=str, default=config.test_config['mode'], 
                       choices=['best', 'latest'], 
                       help=f'测试权重类型，默认: {config.test_config["mode"]}')
    parser.add_argument('--split', type=str, default=config.test_config['split'], 
                       choices=['test', 'val'], 
                       help=f'测试数据集分割，默认: {config.test_config["split"]}')
    parser.add_argument('--save_vis', action='store_true', default=config.test_config['save_vis'],
                       help=f'是否保存可视化结果，默认: {config.test_config["save_vis"]}')
    parser.add_argument('--num_vis', type=int, default=config.test_config['num_vis'], 
                       help=f'保存可视化样本的数量，默认: {config.test_config["num_vis"]}')
    return parser.parse_args()

def save_test_config(config_instance, test_save_path, args):
    """保存测试配置到txt文件"""
    config_file = os.path.join(test_save_path, 'test_config.txt')
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("LITS2017 Dataset Test Configuration\n")  # 英文避免乱码
        f.write("=" * 50 + "\n")
        f.write(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Fold: {args.fold}\n")
        f.write(f"Model Weights: {args.weights}\n")
        f.write(f"Weights Type: {args.test_mode}\n")
        f.write(f"Dataset Split: {args.split}\n")
        f.write(f"Dataset: {config.datasets_name}\n")
        f.write(f"Input Size: {config.input_size_h}x{config.input_size_w}\n")
        f.write(f"Num Classes: {config.num_classes}\n")
        f.write(f"Loss Weights: {config.loss_weight}\n")
        f.write(f"Random Seed: {config.seed}\n")
        f.write(f"Feature Size: {config.model_config.get('feature_size', 48)}\n")
        f.write("=" * 50 + "\n")

def save_test_record(test_results, test_save_path, args, num_samples):
    """保存测试记录到CSV文件"""
    csv_file = os.path.join(test_save_path, 'test_record.csv')
    
    # 准备数据
    record_data = {
        'Test_Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Model': [args.model],
        'Fold': [args.fold],
        'Test_Mode': [args.test_mode],
        'Dataset_Split': [args.split],
        'Test_Samples': [num_samples],
        'Mean_Dice': [test_results['avg_dice']],
        'Mean_IoU': [test_results['avg_miou']],
        'Mean_HD95': [test_results['avg_hd95']],
        'Liver_Dice': [test_results['dice_liver']],
        'Tumor_Dice': [test_results['dice_tumor']],
        'Liver_HD95': [test_results['hd95_liver']],
        'Tumor_HD95': [test_results['hd95_tumor']],
        'Inference_Time': [test_results.get('inference_time', 0)]
    }
    
    # 创建DataFrame并保存
    df = pd.DataFrame(record_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Test record saved to: {csv_file}")

def save_attention_heatmap(model, image, save_path, case_name):
    """保存注意力热力图 - 针对肝脏和肿瘤的attention权重"""
    try:
        model.eval()
        
        # 简化策略：直接从模型输出中获取类别概率，生成类别特定的attention
        attention_weights = []
        
        # 获取attention权重的hook
        def attention_hook(module, input, output):
            # 这是一个简化的hook，我们直接保存输入特征用于后续分析
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                attention_weights.append(output.detach())
        
        # 注册hooks到encoder的attention模块
        hooks = []
        for name, module in model.named_modules():
            if 'attn' in name and hasattr(module, 'qkv'):
                hooks.append(module.register_forward_hook(attention_hook))
        
        # 前向传播
        with torch.no_grad():
            output = model(image)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 获取模型输出的预测概率
        pred_softmax = torch.softmax(output, dim=1)
        liver_prob = pred_softmax[0, 1].cpu().numpy()  # 肝脏概率图
        tumor_prob = pred_softmax[0, 2].cpu().numpy()  # 肿瘤概率图
        
        # 生成类别特定的attention热力图
        plt.figure(figsize=(12, 4))
        
        # 肝脏attention热力图
        plt.subplot(1, 3, 1)
        plt.imshow(liver_prob, cmap='Reds', alpha=0.8)
        plt.colorbar()
        plt.title(f'Liver Attention - {case_name}')
        plt.axis('off')
        
        # 肿瘤attention热力图
        plt.subplot(1, 3, 2)
        plt.imshow(tumor_prob, cmap='Greens', alpha=0.8)
        plt.colorbar()
        plt.title(f'Tumor Attention - {case_name}')
        plt.axis('off')
        
        # 组合attention热力图
        combined_attention = liver_prob + tumor_prob
        plt.subplot(1, 3, 3)
        plt.imshow(combined_attention, cmap='viridis', alpha=0.8)
        plt.colorbar()
        plt.title(f'Combined Attention - {case_name}')
        plt.axis('off')
        
        plt.tight_layout()
        save_file = os.path.join(save_path, f'attention_liver_tumor_{case_name}.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved attention heatmap for {case_name}")
            
    except Exception as e:
        print(f"Failed to save attention heatmap: {e}")

def save_activation_heatmap(model, image, save_path, case_name):
    """保存HBFormer的中间特征激活图（编码器4层 + 解码器4层）"""
    try:
        model.eval()

        encoder_activations = []
        decoder_activations = []

        def create_hook(activation_list):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    # 统一成[B,C,H,W]方便后处理
                    if output.dim() == 4:
                        activation_list.append(output.detach())
                    elif output.dim() == 3:
                        B, L, C = output.shape
                        H = W = int(L ** 0.5)
                        if H * W == L:
                            reshaped = output.permute(0, 2, 1).view(B, C, H, W)
                            activation_list.append(reshaped.detach())
            return hook_fn

        encoder_hooks = []
        decoder_hooks = []

        # HBFormer 编码器：MWAencoder 返回的4个尺度特征
        if hasattr(model, 'MWAencoder'):
            encoder_hooks.append(
                model.MWAencoder.register_forward_hook(create_hook(encoder_activations))
            )

        # HBFormer 解码器：4 个 MFF_Decoder
        for name in ['MFF_decoder4', 'MFF_decoder3', 'MFF_decoder2', 'MFF_decoder1']:
            if hasattr(model, name):
                module = getattr(model, name)
                decoder_hooks.append(
                    module.register_forward_hook(create_hook(decoder_activations))
                )

        if not encoder_hooks and not decoder_hooks:
            print(f"No suitable encoder/decoder modules found for activation heatmap in {model.__class__.__name__}")
            return

        with torch.no_grad():
            _ = model(image)

        for hook in encoder_hooks + decoder_hooks:
            hook.remove()

        # 保存encoder激活图（最多4层）
        for i, activation in enumerate(encoder_activations[:4]):
            try:
                if activation.shape[1] > 1:
                    act_map = activation[0, :min(16, activation.shape[1])].mean(0).cpu().numpy()
                else:
                    act_map = activation[0, 0].cpu().numpy()

                if act_map.shape != (256, 256):
                    act_map = cv2.resize(act_map, (256, 256))

                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)

                plt.figure(figsize=(8, 8))
                plt.imshow(act_map, cmap='viridis')
                plt.colorbar()
                plt.title(f'HBFormer Encoder {i+1} Activation - {case_name}')
                plt.axis('off')

                save_file = os.path.join(save_path, f'hbformer_activation_encoder{i+1}_{case_name}.png')
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error processing HBFormer encoder {i+1} activation: {e}")
                continue

        # 保存decoder激活图（最多4层）
        for i, activation in enumerate(decoder_activations[:4]):
            try:
                if activation.shape[1] > 1:
                    act_map = activation[0, :min(16, activation.shape[1])].mean(0).cpu().numpy()
                else:
                    act_map = activation[0, 0].cpu().numpy()

                if act_map.shape != (256, 256):
                    act_map = cv2.resize(act_map, (256, 256))

                act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)

                plt.figure(figsize=(8, 8))
                plt.imshow(act_map, cmap='plasma')
                plt.colorbar()
                plt.title(f'HBFormer Decoder {4-i} Activation - {case_name}')
                plt.axis('off')

                save_file = os.path.join(save_path, f'hbformer_activation_decoder{4-i}_{case_name}.png')
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Error processing HBFormer decoder {4-i} activation: {e}")
                continue

        print(f"Saved {len(encoder_activations)} HBFormer encoder and {len(decoder_activations)} decoder activations for {case_name}")

    except Exception as e:
        print(f"Failed to save activation heatmap: {e}")

def save_comparison_images(images, labels, predictions, case_names, save_path, num_save=20):
    """保存预测结果与ground truth的对比图 - 英文标题避免乱码"""
    os.makedirs(save_path, exist_ok=True)
    
    class_names = ['Background', 'Liver', 'Tumor']  # 英文避免乱码
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # 背景(黑色)、肝脏(红色)、肿瘤(绿色)
    
    def create_colored_mask(mask, colors):
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(colors):
            colored[mask == class_id] = color
        return colored
    
    for i in range(min(num_save, len(images))):
        # 获取数据
        if torch.is_tensor(images[i]):
            image = images[i][0].cpu().numpy()  # 取第一个通道作为灰度图
        else:
            image = images[i]
            
        if torch.is_tensor(labels[i]):
            label = labels[i].cpu().numpy()
        else:
            label = labels[i]
            
        pred = predictions[i]
        case_name = case_names[i] if i < len(case_names) else f"sample_{i}"
        
        # 生成对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Sample {i+1} ({case_name}): Prediction vs Ground Truth', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original CT Image')
        axes[0, 0].axis('off')
        
        # Ground Truth
        gt_colored = create_colored_mask(label, colors)
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # 预测结果
        pred_colored = create_colored_mask(pred, colors)
        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title('Prediction')
        axes[1, 0].axis('off')
        
        # 重叠显示
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(pred_colored, alpha=0.3)
        axes[1, 1].set_title('Overlay')
        axes[1, 1].axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                         for color, name in zip(colors, class_names)]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'comparison_{i+1:03d}_{case_name}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison images saved to: {save_path}")

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def test_lits2017():
    """主测试函数"""
    args = parse_args()
    
    print(f"Test Configuration:")
    print(f"Model: {args.model}")
    print(f"Test Fold: {args.fold}")
    print(f"Test Mode: {args.test_mode}")
    print(f"Weights Path: {args.weights}")
    print(f"Dataset Split: {args.split}")
    
    # 设置配置
    config.set_model(args.model)
    config.set_test_config(
        test_fold=args.fold,
        test_mode=args.test_mode,
        test_weights_path=args.weights,
        test_dataset_split=args.split
    )
    
    # 创建保存目录
    test_save_path = f'test_result/{args.model}_LITS2017_fold{args.fold}_{args.split}'
    os.makedirs(test_save_path, exist_ok=True)
    
    # 创建子目录
    prediction_dir = os.path.join(test_save_path, 'predictions')
    attention_dir = os.path.join(test_save_path, 'attention_heatmap')
    activation_dir = os.path.join(test_save_path, 'activation_heatmap')
    
    os.makedirs(prediction_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(attention_dir, exist_ok=True)
        os.makedirs(activation_dir, exist_ok=True)
    
    # 保存测试配置
    save_test_config(config, test_save_path, args)
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    print(f"Creating model: {args.model}")
    
    class Args:
        def __init__(self):
            self.dataset = config.datasets_name
    
    model_args = Args()
    
    if args.model == 'HBFormer':
        model = HBFormer(
            args=model_args,
            img_size=(config.input_size_h, config.input_size_w),
            feature_size=config.model_config.get('feature_size', 48),
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # 加载权重
    weights_path = args.weights if args.weights else config.get_test_weights_path()
    print(f"Loading weights: {weights_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 获取模型权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 过滤掉FLOPs计算相关的keys
    filtered_state_dict = {}
    for key, value in state_dict.items():
        # 跳过FLOPs相关的keys和有问题的sobel权重
        if (not key.endswith('.total_ops') and 
            not key.endswith('.total_params') and
            not ('edge_enhancement.sobel' in key and 'weight' in key)):
            filtered_state_dict[key] = value
    
    # 加载过滤后的权重，使用strict=False忽略不匹配的keys
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            print(f"Missing keys: {missing_keys}")
    
    if unexpected_keys:
        print(f"Warning: Unexpected {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            print(f"Unexpected keys: {unexpected_keys}")
    
    print(f"Successfully loaded weights, filtered {len(state_dict) - len(filtered_state_dict)} FLOPs-related keys")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据集 - 关键：使用正确的预处理
    print("Creating test dataset...")
    # 根据当前实现，LiTS2017_dataset 已经按照四目录结构划分好
    # trainImage/trainMask 用于 train/val，testImage/testMask 用于 test
    # 这里保持split使用命令行参数，并显式关闭交叉验证标志
    test_dataset = LiTS2017_dataset(
        data_dir=config.data_path,
        split=args.split,
        transform=TestTransform(output_size=(config.input_size_h, config.input_size_w)),
        fold=None,
        num_folds=5,
        cross_validation=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 避免多进程问题
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # 开始测试
    print("Starting test...")
    start_time = time.time()
    
    dice_scores_liver = []
    dice_scores_tumor = []
    hd95_scores_liver = []
    hd95_scores_tumor = []
    
    all_images = []
    all_labels = []
    all_predictions = []
    all_case_names = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data['image'], data['label']
            case_names = data['case_name']
            
            # 确保数据在正确设备上
            images = images.to(device)
            labels = labels.to(device)
            
            # 获取案例名称
            case_name = case_names[0] if isinstance(case_names, (list, tuple)) else case_names
            
            # 进行预测
            metric_list, pred_np, pred_softmax = test_single_slice_lits(
                images, labels[0], model, config.num_classes,
                patch_size=[config.input_size_h, config.input_size_w],
                case_name=case_name
            )
            
            # 提取指标
            if len(metric_list) >= 2:
                dice_liver, hd95_liver = metric_list[0]
                dice_tumor, hd95_tumor = metric_list[1]
                
                dice_scores_liver.append(dice_liver)
                dice_scores_tumor.append(dice_tumor)
                hd95_scores_liver.append(hd95_liver)
                hd95_scores_tumor.append(hd95_tumor)
            
            # 保存样本用于可视化
            if len(all_images) < args.num_vis:
                all_images.append(images[0])
                all_labels.append(labels[0])
                all_predictions.append(pred_np)
                all_case_names.append(case_name)
            
                # 保存注意力和激活热力图
                if args.save_vis:
                    save_attention_heatmap(model, images, attention_dir, case_name)
                    save_activation_heatmap(model, images, activation_dir, case_name)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{len(test_loader)} samples")
    
    # 计算总体指标
    avg_dice_liver = np.mean(dice_scores_liver) if dice_scores_liver else 0
    avg_dice_tumor = np.mean(dice_scores_tumor) if dice_scores_tumor else 0
    avg_dice = (avg_dice_liver + avg_dice_tumor) / 2
    
    avg_hd95_liver = np.mean(hd95_scores_liver) if hd95_scores_liver else 0
    avg_hd95_tumor = np.mean(hd95_scores_tumor) if hd95_scores_tumor else 0
    avg_hd95 = (avg_hd95_liver + avg_hd95_tumor) / 2
    
    # 计算mIoU
    miou_scores = []
    for i in range(len(all_predictions)):
        pred_tensor = torch.from_numpy(all_predictions[i]).unsqueeze(0).to(device)
        
        # 确保label_tensor也在同一设备上
        if torch.is_tensor(all_labels[i]):
            label_tensor = all_labels[i].unsqueeze(0).to(device)
        else:
            label_tensor = torch.from_numpy(all_labels[i]).unsqueeze(0).to(device)
        
        pred_onehot = torch.zeros(1, config.num_classes, pred_tensor.shape[1], pred_tensor.shape[2]).to(device)
        pred_onehot.scatter_(1, pred_tensor.unsqueeze(1), 1)
        
        miou = calculate_miou(pred_onehot, label_tensor, config.num_classes)
        miou_scores.append(miou)
    
    avg_miou = np.mean(miou_scores) if miou_scores else 0
    
    end_time = time.time()
    inference_time = (end_time - start_time) / len(test_dataset)
    
    # 整理测试结果
    test_results = {
        'avg_dice': avg_dice,
        'dice_liver': avg_dice_liver,
        'dice_tumor': avg_dice_tumor,
        'avg_hd95': avg_hd95,
        'hd95_liver': avg_hd95_liver,
        'hd95_tumor': avg_hd95_tumor,
        'avg_miou': avg_miou,
        'inference_time': inference_time
    }
    
    # 打印结果
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Average Dice: {avg_dice:.4f}")
    print(f"Liver Dice: {avg_dice_liver:.4f}")
    print(f"Tumor Dice: {avg_dice_tumor:.4f}")
    print(f"Average HD95: {avg_hd95:.4f}")
    print(f"Liver HD95: {avg_hd95_liver:.4f}")
    print(f"Tumor HD95: {avg_hd95_tumor:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average inference time: {inference_time:.4f}s/sample")
    print(f"Total test time: {end_time - start_time:.2f}s")
    print("="*50)
    
    # 保存结果
    save_test_record(test_results, test_save_path, args, len(test_dataset))
    
    # 保存对比图像
    if all_images:
        save_comparison_images(all_images, all_labels, all_predictions, all_case_names, 
                             prediction_dir, args.num_vis)
    
    print(f"\nAll results saved to: {test_save_path}")

if __name__ == "__main__":
    test_lits2017() 