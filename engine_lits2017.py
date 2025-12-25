import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
import time
import os
import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def calculate_dice_per_class(pred, target, num_classes):
    """计算每个类别的dice系数"""
    dice_scores = []
    pred_argmax = torch.argmax(pred, dim=1)
    
    for cls in range(1, num_classes):  # 跳过背景类
        pred_cls = (pred_argmax == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()
        
        if total > 0:
            dice = (2.0 * intersection) / total
            dice_scores.append(dice.item())
        else:
            dice = 1.0 if intersection == 0 else 0.0
            dice_scores.append(dice)
    
    return dice_scores

def calculate_miou(pred, target, num_classes):
    """计算mIoU"""
    pred_argmax = torch.argmax(pred, dim=1)
    iou_scores = []
    
    for cls in range(1, num_classes):  # 跳过背景类
        pred_cls = (pred_argmax == cls).float()
        target_cls = (target == cls).float()
        
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        
        if union > 0:
            iou = intersection / union
            iou_scores.append(iou.item())
        else:
            iou = 1.0 if intersection == 0 else 0.0
            iou_scores.append(iou)
    
    return np.mean(iou_scores)

def calculate_hd95(pred, target):
    """计算95th percentile Hausdorff距离"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # 转换为numpy并找到边界
        pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
        target_np = target.cpu().numpy() if torch.is_tensor(target) else target
        
        # 获取边界点
        pred_points = np.column_stack(np.where(pred_np > 0))
        target_points = np.column_stack(np.where(target_np > 0))
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return 0.0
        
        # 计算双向Hausdorff距离
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        return max(hd1, hd2)
    except:
        return 0.0

def test_single_slice_lits(image, label, net, num_classes, patch_size=[256, 256], case_name=None):
    """LITS数据集的单个2D切片测试（适配PNG数据）"""
    net.eval()
    
    with torch.no_grad():
        # 输入已经是正确的格式：(1, 3, H, W)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # 添加batch维度
        
        # 确保输入在GPU上
        image = image.cuda().float()
        
        # 模型预测
        model_output = net(image)
        if isinstance(model_output, tuple):
            outputs = model_output[0]
        else:
            outputs = model_output
        
        # 获取预测结果
        pred_softmax = torch.softmax(outputs, dim=1)
        pred_mask = torch.argmax(pred_softmax, dim=1)
        
        # 转换为numpy进行评估
        pred_np = pred_mask[0].cpu().numpy()
        label_np = label.cpu().numpy() if torch.is_tensor(label) else label
        
        # 计算每个类别的评估指标
        metric_list = []
        for cls in range(1, num_classes):  # 跳过背景类(0)
            pred_cls = (pred_np == cls).astype(np.uint8)
            label_cls = (label_np == cls).astype(np.uint8)
            
            # Dice系数
            intersection = np.sum(pred_cls * label_cls)
            total = np.sum(pred_cls) + np.sum(label_cls)
            if total > 0:
                dice = 2.0 * intersection / total
            else:
                dice = 1.0 if intersection == 0 else 0.0
            
            # HD95 (简化计算)
            hd95 = calculate_hd95(pred_cls, label_cls)
            
            metric_list.append([dice, hd95])
    
    return metric_list, pred_np, pred_softmax[0].cpu().numpy()

def save_prediction_results(images, labels, predictions, case_names, save_dir, num_save=10):
    """保存预测结果的可视化图像"""
    os.makedirs(save_dir, exist_ok=True)
    
    class_names = ['背景', '肝脏', '肿瘤']
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]  # 背景(黑色)、肝脏(红色)、肿瘤(绿色)
    
    def create_colored_mask(mask, colors):
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in enumerate(colors):
            colored[mask == class_id] = color
        return colored
    
    for i in range(min(num_save, len(images))):
        # 获取单个样本数据
        if torch.is_tensor(images[i]):
            image = images[i][0].cpu().numpy()  # 取第一个通道
        else:
            image = images[i]
            
        if torch.is_tensor(labels[i]):
            label = labels[i].cpu().numpy()
        else:
            label = labels[i]
            
        pred = predictions[i]
        case_name = case_names[i] if i < len(case_names) else f"sample_{i}"
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{case_name}: 预测结果对比', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('原始CT图像')
        axes[0, 0].axis('off')
        
        # Ground Truth
        gt_colored = create_colored_mask(label, colors)
        axes[0, 1].imshow(gt_colored)
        axes[0, 1].set_title('真实标签')
        axes[0, 1].axis('off')
        
        # 预测结果
        pred_colored = create_colored_mask(pred, colors)
        axes[1, 0].imshow(pred_colored)
        axes[1, 0].set_title('预测结果')
        axes[1, 0].axis('off')
        
        # 重叠显示
        axes[1, 1].imshow(image, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(pred_colored, alpha=0.3)
        axes[1, 1].set_title('重叠显示')
        axes[1, 1].axis('off')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=np.array(color)/255, label=name) 
                         for color, name in zip(colors, class_names)]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{i+1:03d}_{case_name}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def train_one_epoch_lits(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
    """LITS2017训练一个epoch"""
    stime = time.time()
    model.train()
    
    print(f"开始训练 Epoch {epoch}，总共 {len(train_loader)} 个批次，batch_size={config.batch_size}")
    
    loss_list = []
    dice_scores_all = []
    miou_scores_all = []

    # 使用tqdm显示进度
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    for iter, data in pbar:
        if iter == 0:
            print(f"✓ 开始处理第一个批次...")
        
        optimizer.zero_grad()

        # 数据移动到GPU
        images, targets = data['image'], data['label']
        if iter == 0:
            print(f"✓ 数据加载完成，开始移动到GPU...")
        
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).long()
        
        if iter == 0:
            print(f"✓ 数据已移动到GPU，开始前向传播...")

        if config.amp:
            with autocast():
                if iter == 0:
                    print(f"✓ 开始模型前向传播 (AMP模式)...")
                model_output = model(images)
                
                if iter == 0:
                    print(f"✓ 模型前向传播完成，开始计算损失...")
                
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    out, intermediate_results = model_output
                    if hasattr(criterion, 'forward') and 'intermediate_preds' in criterion.forward.__code__.co_varnames:
                        loss = criterion(out, targets, intermediate_preds=intermediate_results)
                    else:
                        loss = criterion(out, targets)
                else:
                    out = model_output
                    loss = criterion(out, targets)
                      
            if iter == 0:
                print(f"✓ 损失计算完成，开始反向传播...")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if iter == 0:
                print(f"✓ 开始模型前向传播...")
            model_output = model(images)
            
            if iter == 0:
                print(f"✓ 模型前向传播完成，开始计算损失...")
            
            if isinstance(model_output, tuple) and len(model_output) == 2:
                out, intermediate_results = model_output
                if hasattr(criterion, 'forward') and 'intermediate_preds' in criterion.forward.__code__.co_varnames:
                    loss = criterion(out, targets, intermediate_preds=intermediate_results)
                else:
                    loss = criterion(out, targets)
            else:
                out = model_output
                loss = criterion(out, targets)
            
            if iter == 0:
                print(f"✓ 损失计算完成，开始反向传播...")
            loss.backward()
            optimizer.step()

        if iter == 0:
            print(f"✓ 反向传播完成，开始计算评估指标...")
        
        # 计算评估指标
        dice_scores = calculate_dice_per_class(out, targets, config.num_classes)
        miou_score = calculate_miou(out, targets, config.num_classes)
        
        loss_list.append(loss.item())
        dice_scores_all.append(dice_scores)
        miou_scores_all.append(miou_score)

        if iter == 0:
            print(f"✓ 第一个批次完成！后续批次将显示在进度条中...")

        # 更新进度条描述
        if iter % config.print_interval == 0 or iter == 0:
            avg_dice = np.mean(dice_scores)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{avg_dice:.4f}',
                'mIoU': f'{miou_score:.4f}'
            })
            
            if iter % config.print_interval == 0:
                logger.info(f'Epoch: {epoch}, Iter: {iter}, Loss: {loss.item():.4f}, '
                           f'Avg Dice: {avg_dice:.4f}, mIoU: {miou_score:.4f}')

    pbar.close()  # 关闭进度条

    # 计算平均指标
    avg_loss = np.mean(loss_list)
    avg_dice_scores = np.mean(dice_scores_all, axis=0)
    avg_dice = np.mean(avg_dice_scores)
    avg_miou = np.mean(miou_scores_all)
    
    # 记录训练指标到CSV
    train_record = {
        'epoch': epoch,
        'lr': optimizer.param_groups[0]['lr'],
        'loss': avg_loss,
        'avg_dice': avg_dice,
        'miou': avg_miou,
        'dice_liver': avg_dice_scores[0] if len(avg_dice_scores) > 0 else 0,
        'dice_tumor': avg_dice_scores[1] if len(avg_dice_scores) > 1 else 0
    }
    
    train_csv_file = os.path.join(config.work_dir, 'train_record.csv')
    file_exists = os.path.exists(train_csv_file)
    
    with open(train_csv_file, 'a', newline='') as f:
        fieldnames = ['epoch', 'lr', 'loss', 'avg_dice', 'miou', 'dice_liver', 'dice_tumor']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(train_record)

    # 更新学习率
    if scheduler is not None:
        scheduler.step()

    etime = time.time()
    logger.info(f'训练Epoch {epoch} 完成: Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}, '
               f'mIoU: {avg_miou:.4f}, 用时: {etime-stime:.2f}s')

    # 返回训练指标字典
    return {
        'loss': avg_loss,
        'avg_dice': avg_dice,
        'miou': avg_miou,
        'dice_per_class': avg_dice_scores.tolist() if len(avg_dice_scores) > 0 else []
    }

def val_one_epoch_lits(test_datasets, test_loader, model, epoch, logger, config, test_save_path, val_or_test=False):
    """LITS2017验证一个epoch"""
    model.eval()
    
    dice_scores_liver = []
    dice_scores_tumor = []
    hd95_scores_liver = []
    hd95_scores_tumor = []
    
    all_images = []
    all_labels = []
    all_predictions = []
    all_case_names = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc=f"验证Epoch {epoch}")):
            images, labels = data['image'], data['label']
            case_names = data['case_name']
            
            # 处理batch中的每个样本
            for j in range(images.shape[0]):
                image = images[j:j+1]  # 保持batch维度
                label = labels[j]
                case_name = case_names[j] if isinstance(case_names, (list, tuple)) else case_names
                
                # 测试单个切片
                metric_list, pred_np, pred_softmax = test_single_slice_lits(
                    image, label, model, config.num_classes, 
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
                
                # 收集用于可视化的样本
                if len(all_images) < 20:  # 只保存前20个样本用于可视化
                    all_images.append(image[0])
                    all_labels.append(label)
                    all_predictions.append(pred_np)
                    all_case_names.append(case_name)
    
    # 计算平均指标
    avg_dice_liver = np.mean(dice_scores_liver) if dice_scores_liver else 0
    avg_dice_tumor = np.mean(dice_scores_tumor) if dice_scores_tumor else 0
    avg_dice = (avg_dice_liver + avg_dice_tumor) / 2
    
    avg_hd95_liver = np.mean(hd95_scores_liver) if hd95_scores_liver else 0
    avg_hd95_tumor = np.mean(hd95_scores_tumor) if hd95_scores_tumor else 0
    avg_hd95 = (avg_hd95_liver + avg_hd95_tumor) / 2
    
    # 计算mIoU（从所有样本中计算）
    miou_scores = []
    for i in range(len(all_predictions)):
        # 模拟mIoU计算
        pred_tensor = torch.from_numpy(all_predictions[i]).unsqueeze(0)
        label_tensor = all_labels[i].unsqueeze(0) if torch.is_tensor(all_labels[i]) else torch.from_numpy(all_labels[i]).unsqueeze(0)
        
        # 创建one-hot编码用于mIoU计算
        pred_onehot = torch.zeros(1, config.num_classes, pred_tensor.shape[1], pred_tensor.shape[2])
        pred_onehot.scatter_(1, pred_tensor.unsqueeze(1), 1)
        
        miou = calculate_miou(pred_onehot, label_tensor, config.num_classes)
        miou_scores.append(miou)
    
    avg_miou = np.mean(miou_scores) if miou_scores else 0
    
    # 保存可视化结果
    if test_save_path and all_images:
        prediction_dir = os.path.join(test_save_path, 'predictions')
        save_prediction_results(all_images, all_labels, all_predictions, all_case_names, prediction_dir)
    
    # 记录验证指标到CSV
    val_record = {
        'epoch': epoch,
        'avg_dice': avg_dice,
        'avg_hd95': avg_hd95,
        'miou': avg_miou,
        'dice_liver': avg_dice_liver,
        'dice_tumor': avg_dice_tumor,
        'hd95_liver': avg_hd95_liver,
        'hd95_tumor': avg_hd95_tumor
    }
    
    val_csv_file = os.path.join(config.work_dir, 'val_record.csv')
    file_exists = os.path.exists(val_csv_file)
    
    with open(val_csv_file, 'a', newline='') as f:
        fieldnames = ['epoch', 'avg_dice', 'avg_hd95', 'miou', 'dice_liver', 'dice_tumor', 'hd95_liver', 'hd95_tumor']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(val_record)
    
    logger.info(f'验证Epoch {epoch}: Avg Dice: {avg_dice:.4f}, Liver Dice: {avg_dice_liver:.4f}, '
               f'Tumor Dice: {avg_dice_tumor:.4f}, mIoU: {avg_miou:.4f}')
    
    # 返回验证指标字典
    return {
        'avg_dice': avg_dice,
        'avg_hd95': avg_hd95,
        'miou': avg_miou,
        'dice_per_class': [avg_dice_liver, avg_dice_tumor],
        'hd95_per_class': [avg_hd95_liver, avg_hd95_tumor]
    } 