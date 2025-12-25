import os
import sys
# 在导入任何模块之前设置环境变量来抑制torchvision警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
import json
import csv
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from datasets.dataset import RandomGenerator
from engine_synapse import val_one_epoch_with_visualization
from models.vmunet.vmunet import VMUNet
from models.DWSegNet import DWSegNet
from models.HBFormer import HBFormer

from utils import *
from configs.config_setting_synapse import setting_config

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", "Failed to load image Python extension")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略thop相关的INFO信息
logging.getLogger('thop').setLevel(logging.WARNING)


def save_test_results(config, results, save_dir):
    """保存测试结果到CSV和JSON文件"""
    
    # 保存详细结果到JSON
    json_file = os.path.join(save_dir, 'test_results_detailed.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 保存汇总结果到CSV
    csv_file = os.path.join(save_dir, 'test_results_summary.csv')
    
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    
    # 准备CSV数据
    csv_data = {
        'model': config.network,
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'avg_dice': results['summary']['avg_dice'],
        'mean_hd95': results['summary']['mean_hd95'],
    }
    
    # 添加每个器官的指标
    for i, organ in enumerate(organ_names):
        # 兼容旧结构：如果summary里没有器官级指标，则从volume_eval里取
        dice_list = results.get('summary', {}).get('dice_per_organ',
                    results.get('volume_eval', {}).get('dice_per_organ', []))
        hd95_list = results.get('summary', {}).get('hd95_per_organ',
                    results.get('volume_eval', {}).get('hd95_per_organ', []))

        if i < len(dice_list):
            csv_data[f'dice_{organ}'] = float(dice_list[i])
        if i < len(hd95_list):
            csv_data[f'hd95_{organ}'] = float(hd95_list[i])
    
    # 写入CSV
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['model', 'test_time', 'avg_dice', 'mean_hd95'] + \
                    [f'dice_{organ}' for organ in organ_names] + \
                    [f'hd95_{organ}' for organ in organ_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_data)
    
    print(f"✅ 测试结果已保存:")
    print(f"    详细结果: {json_file}")
    print(f"    汇总结果: {csv_file}")


def create_test_visualization_dirs(test_save_path):
    """创建测试可视化目录"""
    prediction_vis_dir = os.path.join(test_save_path, 'prediction_visualization')
    attention_vis_dir = os.path.join(test_save_path, 'attention_heatmaps')
    comparison_dir = os.path.join(test_save_path, 'comparison_images')
    
    os.makedirs(prediction_vis_dir, exist_ok=True)
    os.makedirs(attention_vis_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    return prediction_vis_dir, attention_vis_dir, comparison_dir


def extract_attention_maps(model, input_tensor):
    """提取注意力图（内存友好版）：限制钩子数量并对特征进行下采样"""
    import torch.nn.functional as F
    MAX_HOOKS_ATTENTION = 24
    DOWNSAMPLED_SIZE = 128

    attention_weights = []

    def attention_hook_fn(module, inputs, output):
        try:
            if len(attention_weights) >= MAX_HOOKS_ATTENTION:
                return
            # 优先从 last_attn_weights 取
            if hasattr(module, 'last_attn_weights') and module.last_attn_weights is not None:
                attn = module.last_attn_weights
                if isinstance(attn, torch.Tensor):
                    attn = attn.detach()
                else:
                    return
            elif hasattr(module, 'attn') or 'Attention' in str(type(module)):
                if isinstance(output, tuple) and len(output) >= 2 and isinstance(output[1], torch.Tensor):
                    attn = output[1].detach()
                elif isinstance(output, torch.Tensor):
                    attn = output.detach().mean(dim=1, keepdim=True)
                else:
                    return
            elif any(x in str(type(module)) for x in ['VSSBlock', 'SS2D', 'SelectiveScan']):
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    attn = torch.var(output.detach(), dim=1, keepdim=True)
                else:
                    return
            else:
                return

            # 将注意力图规约为单通道 2D 并下采样
            if isinstance(attn, torch.Tensor):
                if attn.dim() == 4:
                    # (B, C, H, W) -> (B, 1, H, W)
                    if attn.size(1) > 1:
                        attn = attn.mean(dim=1, keepdim=True)
                    h, w = attn.shape[-2:]
                    target_h = min(DOWNSAMPLED_SIZE, h)
                    target_w = min(DOWNSAMPLED_SIZE, w)
                    if (h, w) != (target_h, target_w):
                        attn = F.interpolate(attn, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    attention_weights.append(attn.cpu())
        except RuntimeError:
            # 遇到显存问题直接跳过该层
            return

    # 先收集候选模块，再等间隔抽样注册钩子
    candidates = []
    for name, module in model.named_modules():
        if any(x in str(type(module)) for x in ['Attention', 'MultiHeadAttention', 'VSSBlock', 'SS2D', 'SelectiveScan']):
            candidates.append((name, module))

    # 如果候选为空，退而选择名称包含关键字的模块
    if not candidates:
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['encoder', 'decoder', 'block', 'layer', 'stage']):
                candidates.append((name, module))

    if not candidates:
        return []

    num_to_hook = min(MAX_HOOKS_ATTENTION, len(candidates))
    if num_to_hook < len(candidates):
        import numpy as np
        sel_idx = np.linspace(0, len(candidates) - 1, num_to_hook, dtype=int).tolist()
    else:
        sel_idx = list(range(len(candidates)))

    attention_hooks = []
    for i in sel_idx:
        hook = candidates[i][1].register_forward_hook(attention_hook_fn)
        attention_hooks.append(hook)

    # 前向传播（混合精度 + no_grad）
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                _ = model(input_tensor)
    except RuntimeError as e:
        # OOM 容错
        for h in attention_hooks:
            h.remove()
        torch.cuda.empty_cache()
        print(f"⚠️ 注意力图提取发生异常: {e}")
        return []

    for h in attention_hooks:
        h.remove()

    print(f"✅ 提取到 {len(attention_weights)} 个attention图(已限流)")
    return attention_weights


def save_attention_heatmaps(image, attention_maps, case_name, save_dir, slice_idx=None):
    """保存注意力热图（限流 + 深色主题）"""
    if len(attention_maps) == 0:
        return []

    # 限制最多保存的层数，等间隔采样
    MAX_SAVE = 24
    num_layers_total = len(attention_maps)
    if num_layers_total > MAX_SAVE:
        import numpy as np
        sel_idx = np.linspace(0, num_layers_total - 1, MAX_SAVE, dtype=int).tolist()
        attention_maps = [attention_maps[i] for i in sel_idx]
    num_layers = len(attention_maps)

    saved_paths = []

    def process_attn_map(attn_map):
        # DWA_Block: (nW*B, num_heads, N, N)
        if len(attn_map.shape) == 4 and attn_map.shape[2] == attn_map.shape[3]:
            attn_map = attn_map[:, 0].mean(dim=0)  # (N, N)
            attn_map = attn_map.diag() if attn_map.shape[0] == attn_map.shape[1] else attn_map.mean(dim=0)
            side = int(np.sqrt(attn_map.shape[0]))
            attn_map = attn_map[:side * side].reshape(side, side)
        return attn_map

    # 3D：随机选择中间 slice 附近的一张
    if len(image.shape) == 3:
        import random
        num_slices = image.shape[0]
        start_slice = max(0, int(num_slices * 0.1))
        end_slice = min(num_slices, int(num_slices * 0.9))
        selected_slice = random.randint(start_slice, end_slice - 1) if end_slice > start_slice else num_slices // 2
        image_slice = image[selected_slice]
        if image_slice.max() > 1.1:
            image_slice = image_slice / 255.0

        cols = min(6, num_layers)
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        fig.patch.set_facecolor('#0a1929')

        for i in range(num_layers):
            attn_map = attention_maps[i]
            attn_map = process_attn_map(attn_map)
            if len(attn_map.shape) == 4:
                attn_map = attn_map[0, 0]
            elif len(attn_map.shape) == 3:
                attn_map = attn_map[0]

            target_size = image_slice.shape
            if attn_map.shape != target_size:
                attn_map = cv2.resize(attn_map.numpy(), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            else:
                attn_map = attn_map.numpy()

            attn_min, attn_max = attn_map.min(), attn_map.max()
            if attn_max > attn_min:
                attn_map = (attn_map - attn_min) / (attn_max - attn_min)
            else:
                attn_map = np.zeros_like(attn_map)

            background = np.full_like(image_slice, 0.05)
            im = axes[i].imshow(background, cmap='Blues', vmin=0, vmax=1)
            attention_overlay = axes[i].imshow(attn_map, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
            axes[i].set_title(f'Attention {i+1}', fontsize=10, color='white', fontweight='bold')
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
            cbar = plt.colorbar(attention_overlay, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Attention', color='white', fontsize=8)
            cbar.ax.tick_params(colors='white', labelsize=6)

        total_subplots = rows * cols
        for i in range(num_layers, total_subplots):
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')

        fig.suptitle(f'Attention Heatmaps - {case_name} - Slice {selected_slice:03d}', fontsize=14, color='white', fontweight='bold', y=0.95)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{case_name}_random_slice{selected_slice:03d}_attention_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a1929', edgecolor='none')
        plt.close()
        saved_paths.append(save_path)
        print(f" 已保存注意力热图: {case_name} - 随机slice {selected_slice}")
    else:
        # 2D 情况（保持简化处理）
        pass

    return saved_paths


def save_prediction_comparison(image, ground_truth, prediction, case_name, save_dir, slice_idx=None):
    """保存预测对比图：输入图像、ground truth、预测结果的三联图，并添加器官颜色图示"""
    
    # 定义Synapse数据集的器官标签和对应颜色
    organ_labels = {
        0: ('Background', '#000000'),  # 黑色
        1: ('Aorta', '#FF0000'),       # 红色
        2: ('Gallbladder', '#00FF00'), # 绿色
        3: ('Left Kidney', '#0000FF'), # 蓝色
        4: ('Right Kidney', '#FFFF00'),# 黄色
        5: ('Liver', '#FF00FF'),       # 洋红色
        6: ('Pancreas', '#00FFFF'),    # 青色
        7: ('Spleen', '#FFA500'),      # 橙色
        8: ('Stomach', '#800080')      # 紫色
    }
    
    saved_paths = []
    
    # 如果是3D数据，保存所有slice
    if len(image.shape) == 3:
        num_slices = image.shape[0]
        for slice_idx in range(num_slices):
            image_slice = image[slice_idx]
            gt_slice = ground_truth[slice_idx] 
            pred_slice = prediction[slice_idx]
            
            # 归一化图像到0-1范围
            if image_slice.max() > 1.1:
                image_slice = image_slice / 255.0
            
            # 创建主图和图例的布局
            fig = plt.figure(figsize=(20, 6))
            
            # 创建网格布局：3个主图 + 1个图例
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.3], wspace=0.15)
            
            # 输入图像
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(image_slice, cmap='gray')
            ax1.set_title('Input Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Ground Truth
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(image_slice, cmap='gray', alpha=0.3)
            im_gt = ax2.imshow(gt_slice, cmap='jet', alpha=0.7, vmin=0, vmax=8)
            ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            # 预测结果
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(image_slice, cmap='gray', alpha=0.3)
            im_pred = ax3.imshow(pred_slice, cmap='jet', alpha=0.7, vmin=0, vmax=8)
            ax3.set_title('Prediction', fontsize=14, fontweight='bold')
            ax3.axis('off')
            
            # 创建器官颜色图例
            ax_legend = fig.add_subplot(gs[0, 3])
            ax_legend.axis('off')
            
            # 获取当前slice中存在的器官标签
            unique_labels_gt = np.unique(gt_slice)
            unique_labels_pred = np.unique(pred_slice)
            unique_labels = np.unique(np.concatenate([unique_labels_gt, unique_labels_pred]))
            
            # 绘制图例
            legend_y_start = 0.9
            legend_spacing = 0.08
            
            ax_legend.text(0.05, 0.95, 'Organ Legend:', fontsize=12, fontweight='bold', 
                          transform=ax_legend.transAxes)
            
            # 获取jet colormap用于颜色映射
            import matplotlib.cm as cm
            jet_cmap = cm.get_cmap('jet')
            
            for i, label in enumerate(sorted(unique_labels)):
                if int(label) in organ_labels:
                    organ_name = organ_labels[int(label)][0]
                    # 使用jet colormap的颜色
                    color = jet_cmap(label / 8.0) if label > 0 else (0, 0, 0, 1)
                    
                    y_pos = legend_y_start - i * legend_spacing
                    
                    # 绘制颜色方块
                    rect = plt.Rectangle((0.05, y_pos-0.02), 0.15, 0.04, 
                                       facecolor=color, transform=ax_legend.transAxes)
                    ax_legend.add_patch(rect)
                    
                    # 添加器官名称
                    ax_legend.text(0.25, y_pos, f'{int(label)}: {organ_name}', 
                                 fontsize=10, transform=ax_legend.transAxes, 
                                 verticalalignment='center')
            
            # 添加slice信息
            fig.suptitle(f'{case_name} - Slice {slice_idx:03d}', fontsize=16, fontweight='bold')
            
            # 保存图像
            save_path = os.path.join(save_dir, f'{case_name}_slice{slice_idx:03d}_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_paths.append(save_path)
    else:
        # 2D数据处理
        image_slice = image
        gt_slice = ground_truth
        pred_slice = prediction
        
        # 归一化图像到0-1范围
        if image_slice.max() > 1.1:
            image_slice = image_slice / 255.0
        
        # 创建主图和图例的布局
        fig = plt.figure(figsize=(20, 6))
        
        # 创建网格布局：3个主图 + 1个图例
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.3], wspace=0.15)
        
        # 输入图像
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image_slice, cmap='gray')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Ground Truth
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image_slice, cmap='gray', alpha=0.3)
        im_gt = ax2.imshow(gt_slice, cmap='jet', alpha=0.7, vmin=0, vmax=8)
        ax2.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 预测结果
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image_slice, cmap='gray', alpha=0.3)
        im_pred = ax3.imshow(pred_slice, cmap='jet', alpha=0.7, vmin=0, vmax=8)
        ax3.set_title('Prediction', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 创建器官颜色图例
        ax_legend = fig.add_subplot(gs[0, 3])
        ax_legend.axis('off')
        
        # 获取当前slice中存在的器官标签
        unique_labels_gt = np.unique(gt_slice)
        unique_labels_pred = np.unique(pred_slice)
        unique_labels = np.unique(np.concatenate([unique_labels_gt, unique_labels_pred]))
        
        # 绘制图例
        ax_legend.text(0.05, 0.95, 'Organ Legend:', fontsize=12, fontweight='bold', 
                      transform=ax_legend.transAxes)
        
        # 获取jet colormap用于颜色映射
        import matplotlib.cm as cm
        jet_cmap = cm.get_cmap('jet')
        
        for i, label in enumerate(sorted(unique_labels)):
            if int(label) in organ_labels:
                organ_name = organ_labels[int(label)][0]
                # 使用jet colormap的颜色
                color = jet_cmap(label / 8.0) if label > 0 else (0, 0, 0, 1)
                
                y_pos = 0.9 - i * 0.08
                
                # 绘制颜色方块
                rect = plt.Rectangle((0.05, y_pos-0.02), 0.15, 0.04, 
                                   facecolor=color, transform=ax_legend.transAxes)
                ax_legend.add_patch(rect)
                
                # 添加器官名称
                ax_legend.text(0.25, y_pos, f'{int(label)}: {organ_name}', 
                             fontsize=10, transform=ax_legend.transAxes, 
                             verticalalignment='center')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(save_dir, f'{case_name}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_paths.append(save_path)
    
    return saved_paths


def extract_activation_maps(model, input_tensor):
    """提取激活图（内存友好版）：限量钩子 + 下采样 + 混合精度"""
    import torch.nn.functional as F
    MAX_HOOKS_ACTIVATION = 24
    DOWNSAMPLED_SIZE = 128

    activation_maps = []
    print(f" 开始提取激活图，输入tensor形状: {input_tensor.shape}")

    def hook_fn(module, inputs, output):
        try:
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                act = output.detach()
                # (B, C, H, W) -> (B, 1, H, W)
                act = torch.max(act, dim=1, keepdim=True)[0]
                h, w = act.shape[-2:]
                target_h = min(DOWNSAMPLED_SIZE, h)
                target_w = min(DOWNSAMPLED_SIZE, w)
                if (h, w) != (target_h, target_w):
                    act = F.interpolate(act, size=(target_h, target_w), mode='bilinear', align_corners=False)
                activation_maps.append(act.cpu())
        except RuntimeError:
            return

    # 先筛选候选模块
    target_modules = ['Conv2d', 'ConvTranspose2d', 'VSSBlock', 'SS2D', 'Linear', 'LayerNorm', 'BatchNorm2d', 'DWA_Block', 'DualScopeFusionBlock', 'DynamicExpertBlock']
    candidates = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if any(target in module_type for target in target_modules) and any(keyword in name.lower() for keyword in ['encoder', 'decoder', 'block', 'layer', 'conv', 'stage']):
            candidates.append((name, module))

    if not candidates:
        return []

    # 等间隔抽样注册钩子
    num_to_hook = min(MAX_HOOKS_ACTIVATION, len(candidates))
    if num_to_hook < len(candidates):
        import numpy as np
        sel_idx = np.linspace(0, len(candidates) - 1, num_to_hook, dtype=int).tolist()
    else:
        sel_idx = list(range(len(candidates)))

    hooks = []
    for i in sel_idx:
        hooks.append(candidates[i][1].register_forward_hook(hook_fn))

    # 前向传播（混合精度 + no_grad）
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                _ = model(input_tensor)
    except RuntimeError as e:
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()
        print(f"⚠️ 激活图提取发生异常: {e}")
        return []

    for h in hooks:
        h.remove()

    if not activation_maps:
        print("⚠️ 没有提取到激活图，使用模型输出作为fallback")
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(input_tensor)
                if isinstance(outputs, torch.Tensor) and outputs.dim() == 4:
                    act = torch.mean(outputs, dim=1, keepdim=True)
                    h, w = act.shape[-2:]
                    target_h = min(DOWNSAMPLED_SIZE, h)
                    target_w = min(DOWNSAMPLED_SIZE, w)
                    if (h, w) != (target_h, target_w):
                        act = F.interpolate(act, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    activation_maps.append(act.cpu())
        except RuntimeError:
            torch.cuda.empty_cache()
            return []

    print(f" 最终提取到 {len(activation_maps)} 个激活图(已限流)")
    return activation_maps


def save_activation_heatmaps(image, activation_maps, case_name, save_dir, slice_idx=None):
    """保存激活热图（限流 + 深色主题），均匀采样层索引"""
    if len(activation_maps) == 0:
        print("⚠️ 没有可用的激活图")
        return []

    # 动态选择最多保存的层数
    MAX_SAVE = 24
    num_total = len(activation_maps)
    if num_total > MAX_SAVE:
        import numpy as np
        sel_idx = np.linspace(0, num_total - 1, MAX_SAVE, dtype=int).tolist()
        filtered_activation_maps = [activation_maps[i] for i in sel_idx]
        selected_layer_indices = sel_idx
    else:
        filtered_activation_maps = activation_maps
        selected_layer_indices = list(range(num_total))

    print(f" 开始生成激活热图: {case_name}, 选取层数: {len(filtered_activation_maps)}, 保存目录: {save_dir}")

    saved_paths = []
    if len(image.shape) == 3:
        import random
        num_slices = image.shape[0]
        start_slice = max(0, int(num_slices * 0.1))
        end_slice = min(num_slices, int(num_slices * 0.9))
        selected_slice = random.randint(start_slice, end_slice - 1) if end_slice > start_slice else num_slices // 2

        image_slice = image[selected_slice]
        if image_slice.max() > 1.1:
            image_slice = image_slice / 255.0

        num_layers = len(filtered_activation_maps)
        if num_layers == 0:
            return []

        cols = min(6, num_layers)
        rows = (num_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        fig.patch.set_facecolor('#0a1929')

        def process_act_map(act_map):
            if len(act_map.shape) == 4:
                act_map = act_map[0].mean(dim=0)
            elif len(act_map.shape) == 3:
                act_map = act_map.mean(dim=0)
            return act_map

        for i in range(num_layers):
            activation_map = filtered_activation_maps[i]
            actual_layer_idx = selected_layer_indices[i] + 1
            activation_map = process_act_map(activation_map)

            target_size = image_slice.shape
            if activation_map.shape != target_size:
                activation_map = cv2.resize(activation_map.numpy(), (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            else:
                activation_map = activation_map.numpy()

            act_min, act_max = activation_map.min(), activation_map.max()
            if act_max > act_min:
                activation_map = (activation_map - act_min) / (act_max - act_min)
            else:
                activation_map = np.zeros_like(activation_map)

            background = np.full_like(image_slice, 0.05)
            im = axes[i].imshow(background, cmap='Blues', vmin=0, vmax=1)
            activation_overlay = axes[i].imshow(activation_map, cmap='RdYlGn', alpha=0.8, vmin=0, vmax=1)
            axes[i].set_title(f'Layer {actual_layer_idx}', fontsize=10, color='white', fontweight='bold')
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')
            cbar = plt.colorbar(activation_overlay, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.set_label('Activation', color='white', fontsize=8)
            cbar.ax.tick_params(colors='white', labelsize=6)

        total_subplots = rows * cols
        for i in range(num_layers, total_subplots):
            axes[i].axis('off')
            axes[i].set_facecolor('#0a1929')

        fig.suptitle(f'Selected Activation Heatmaps - {case_name} - Slice {selected_slice:03d}', fontsize=14, color='white', fontweight='bold', y=0.95)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{case_name}_random_slice{selected_slice:03d}_selected_activation_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#0a1929', edgecolor='none')
        plt.close()
        saved_paths.append(save_path)
        print(f" 已保存选定层激活热图: {case_name} - 随机slice {selected_slice}, 共 {num_layers} 层")
    else:
        # 2D 情况（简化）
        pass

    return saved_paths


def main():
    config = setting_config
    if config.network == 'DWSegNet':
        config.network = 'DWSegNet'
        config.model_config = {
            'num_classes': 9, 
            'input_channels': 3, 
            'feature_size': 48,  # DWSegNet的特征维度
            'use_boundary_refinement': False,
            'load_ckpt_path': '',  # 暂时不使用预训练权重
        }
    elif config.network == 'vmunet':
        config.network = 'vmunet'
        config.model_config = {
            'num_classes': 9, 
            'input_channels': 3, 
            'feature_size': 48,  # DWSegNet的特征维度
            'use_boundary_refinement': False,
            'load_ckpt_path': '',  # 暂时不使用预训练权重
        }
    elif config.network == 'HBFormer':
        config.network = 'HBFormer'
        config.model_config = {
            'num_classes': 9,
            'input_channels': 3,
            'feature_size': 48,
            'use_boundary_refinement': False,
        }
    #设置权重路径
    config.test_weights_path = config.test_weights_path
    
    print('#----------创建测试环境----------#')
    print(f" 当前网络架构: {config.network}")
    print(f" 模型配置: {config.model_config}")
    
    # 创建测试输出目录
    test_work_dir = f'test_result/{config.network}_synapse'
    os.makedirs(test_work_dir, exist_ok=True)
    
    # 创建可视化目录
    prediction_vis_dir, attention_vis_dir, comparison_dir = create_test_visualization_dirs(test_work_dir)
    
    # 添加activation heatmap目录
    activation_vis_dir = os.path.join(test_work_dir, 'activation_heatmaps')
    os.makedirs(activation_vis_dir, exist_ok=True)
    
    # 创建日志
    log_dir = os.path.join(test_work_dir, 'log')
    global logger
    logger = get_logger('test', log_dir)
    
    print('#----------GPU初始化----------#')
    set_seed(config.seed)
    torch.cuda.empty_cache()
    
    print('#----------准备数据集----------#')
    val_dataset = config.datasets(base_dir=config.volume_path, split="test_vol", list_dir=config.list_dir)
    val_loader = DataLoader(val_dataset,
                           batch_size=1,
                           shuffle=False,
                           pin_memory=True, 
                           num_workers=config.num_workers,
                           drop_last=False)
    
    print('#----------准备模型----------#')
    model_cfg = config.model_config
    
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'], # 恢复原版VMUNet层深度配置
            input_channels=model_cfg['input_channels'],
            depths= [2, 2, 2, 2],
            depths_decoder=[2, 2, 2, 1],
            drop_path_rate=0.2,
            load_ckpt_path=None,  # 测试时不使用预训练权重
        )

        
    elif config.network == 'DWSegNet':
        # 创建args对象 - DWSegNet需要这个
        class Args:
            def __init__(self):
                self.dataset = 'Synapse'  # 这会让模型自动设置为9个类别
        
        args = Args()
        model = DWSegNet(
            args=args,
            img_size=(256, 256),
            feature_size=config.model_config['feature_size'],
            use_boundary_refinement=config.model_config.get('use_boundary_refinement', False)
        )

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
        
    else:
        raise ValueError(f"Unsupported network: {config.network}")
    
    model = model.cuda()
    
    # 加载最佳权重（优先使用best_dice.pth，与训练验证一致）
    def load_checkpoint_safely(model, ckpt_path):
        print(f" 尝试加载测试权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        # 兼容多种保存格式
        if isinstance(ckpt, dict) and any(k in ckpt for k in ['state_dict', 'model_state_dict']):
            state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict'))
        else:
            state_dict = ckpt
        # 去除module.前缀
        cleaned = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith('module.') else k
            cleaned[nk] = v
        # 仅保留在模型中存在且shape匹配的权重
        model_dict = model.state_dict()
        filtered = {k: v for k, v in cleaned.items() if k in model_dict and model_dict[k].shape == v.shape}
        missing = [k for k in model_dict.keys() if k not in filtered]
        unexpected = [k for k in cleaned.keys() if k not in model_dict]
        model.load_state_dict(filtered, strict=False)
        print(f"✅ 加载权重(过滤后): 匹配 {len(filtered)} 个; 缺失 {len(missing)}; 忽略 {len(unexpected)}")

    ckpt_dir = f'results/{config.network}_{config.datasets_name}/checkpoints'
    preferred = os.path.join(ckpt_dir, 'best_dice.pth')
    fallback = config.test_weights_path or os.path.join(ckpt_dir, 'best.pth')
    if os.path.exists(preferred):
        load_checkpoint_safely(model, preferred)
    elif os.path.exists(fallback):
        load_checkpoint_safely(model, fallback)
    else:
        raise FileNotFoundError(f"❌ 找不到测试权重文件: {preferred} 或 {fallback}")
    
    print('#----------开始测试----------#')
    
    # 保持与训练一致：先进行 slice-by-slice 验证（与train完全一致的预处理）
    class ValTransform(object):
        def __init__(self, output_size=[256, 256]):
            self.output_size = output_size
        def __call__(self, sample):
            image, label = sample['image'], sample['label']
            image = image.astype(np.float32)
            label = label.astype(np.float32)
            if image.shape != tuple(self.output_size):
                from scipy.ndimage import zoom
                x, y = image.shape
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image_tensor = torch.from_numpy(image)
            channel1 = image_tensor
            channel2 = torch.clamp(image_tensor * 1.2, 0, 1)
            channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()
            image = torch.stack([channel1, channel2, channel3], dim=0)
            label = torch.from_numpy(label)
            sample = {'image': image, 'label': label.long()}
            return sample

    from datasets.dataset import Synapse_dataset
    val_dataset_slice = Synapse_dataset(
        base_dir=config.data_path,
        list_dir=config.list_dir,
        split="test_slice",
        transform=ValTransform([config.input_size_h, config.input_size_w])
    )
    val_loader_slice = DataLoader(val_dataset_slice,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True)

    from engine_synapse import val_one_epoch_slice, val_one_epoch_with_visualization
    slice_metrics = val_one_epoch_slice(
        val_dataset_slice,
        val_loader_slice,
        model,
        0,
        logger,
        config,
        test_save_path=test_work_dir,
        val_or_test=True
    )

    # 同时运行原volume评估与可视化，便于对照（可选）
    mean_dice, mean_hd95, dice_per_organ, hd95_per_organ = val_one_epoch_with_visualization(
        val_dataset,
        val_loader,
        model,
        0,
        logger,
        config,
        test_save_path=test_work_dir,
        prediction_vis_dir=prediction_vis_dir,
        attention_vis_dir=attention_vis_dir,
        val_or_test=True,
        save_vis_every_n=1,
        save_prediction_comparison_func=save_prediction_comparison,
        extract_attention_maps_func=extract_attention_maps,
        save_attention_heatmaps_func=save_attention_heatmaps,
        extract_activation_maps_func=extract_activation_maps,
        save_activation_heatmaps_func=save_activation_heatmaps,
        activation_vis_dir=activation_vis_dir
    )
    
    # 确保获得了详细的器官评估指标（volume评估）
    organ_names = ['Aorta', 'Gallbladder', 'Left_Kidney', 'Right_Kidney', 'Liver', 'Pancreas', 'Spleen', 'Stomach']
    if len(dice_per_organ) == 0:
        print("⚠️ 警告：没有获得详细的器官Dice信息，使用默认值")
        dice_per_organ = [0.0] * len(organ_names)
        hd95_per_organ = [0.0] * len(organ_names)
    
    # 整理测试结果
    # 以 slice 评估为主（与训练验证一致），同时记录 volume 评估供对照
    results = {
        'model_info': {
            'network': config.network,
            'model_config': dict(model_cfg),
            'weights_dir': f"results/{config.network}_{config.datasets_name}/checkpoints"
        },
        'summary': {
            'avg_dice': float(slice_metrics['avg_dice']),
            'mean_hd95': float(slice_metrics['avg_hd95']),
        },
        'slice_eval': {
            'avg_dice': float(slice_metrics['avg_dice']),
            'avg_hd95': float(slice_metrics['avg_hd95']),
            'miou': float(slice_metrics['miou'])
        },
        'volume_eval': {
            'avg_dice': float(mean_dice),
            'avg_hd95': float(mean_hd95),
            'dice_per_organ': dice_per_organ,
            'hd95_per_organ': hd95_per_organ
        },
        'test_info': {
            'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_test_cases': len(val_dataset),
            'output_dir': test_work_dir
        }
    }
    
    # 保存测试结果
    save_test_results(config, results, test_work_dir)
    
    print('\n' + '='*80)
    print(' 测试完成！')
    print('='*80)
    print(f" Slice评估 Dice: {slice_metrics['avg_dice']:.4f}, HD95: {slice_metrics['avg_hd95']:.4f}, mIoU: {slice_metrics['miou']:.4f}")
    print(f" Volume评估 Dice: {mean_dice:.4f}, HD95: {mean_hd95:.4f}")
    print(f" 结果保存路径: {test_work_dir}")
    print(f"️  预测可视化: {prediction_vis_dir}")
    print(f" 注意力热图: {attention_vis_dir}")
    print(f"⚡ 激活热图: {activation_vis_dir}")
    print('='*80)


if __name__ == '__main__':
    main() 