import numpy as np
from tqdm import tqdm

from torch.cuda.amp import autocast as autocast
import torch
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

from scipy.ndimage.morphology import binary_fill_holes, binary_opening

from utils import test_single_volume

import time


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
            dice_scores.append(dice.item())  # 这里需要.item()因为dice是tensor
        else:
            dice = 1.0 if intersection == 0 else 0.0
            dice_scores.append(dice)  # 这里不需要.item()因为dice已经是float
    
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
            iou_scores.append(iou.item())  # 这里需要.item()因为iou是tensor
        else:
            iou = 1.0 if intersection == 0 else 0.0
            iou_scores.append(iou)  # 这里不需要.item()因为iou已经是float
    
    return np.mean(iou_scores)


def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    stime = time.time()
    model.train() 
 
    loss_list = []
    dice_scores_all = []
    miou_scores_all = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()

        images, targets = data['image'], data['label']
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).long()

        if config.amp:
            with autocast():
                # AFFSegNet返回两个输出：主预测和中间结果
                if hasattr(model, 'module'):
                    # 处理DataParallel或DistributedDataParallel
                    model_output = model(images)
                else:
                    model_output = model(images)
                
                # 解包输出
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    out, intermediate_results = model_output
                    # 如果criterion支持中间监督，传入中间结果
                    if hasattr(criterion, 'forward') and 'intermediate_preds' in criterion.forward.__code__.co_varnames:
                        loss = criterion(out, targets, intermediate_preds=intermediate_results)
                    else:
                        loss = criterion(out, targets)
                else:
                    out = model_output
                    loss = criterion(out, targets)
                      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # AFFSegNet返回两个输出：主预测和中间结果
            model_output = model(images)
            
            # 解包输出
            if isinstance(model_output, tuple) and len(model_output) == 2:
                out, intermediate_results = model_output
                # 如果criterion支持中间监督，传入中间结果
                if hasattr(criterion, 'forward') and 'intermediate_preds' in criterion.forward.__code__.co_varnames:
                    loss = criterion(out, targets, intermediate_preds=intermediate_results)
                else:
                    loss = criterion(out, targets)
            else:
                out = model_output
                loss = criterion(out, targets)
            
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        
        # 计算训练时的dice和mIoU
        with torch.no_grad():
            dice_scores = calculate_dice_per_class(out, targets, config.num_classes)
            miou_score = calculate_miou(out, targets, config.num_classes)
            dice_scores_all.append(dice_scores)
            miou_scores_all.append(miou_score)
        
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        mean_loss = np.mean(loss_list)
        if iter % config.print_interval == 0 and iter != 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {loss.item():.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    
    scheduler.step()
    
    # 计算平均指标
    mean_loss = np.mean(loss_list)
    mean_dice_per_class = np.mean(dice_scores_all, axis=0)
    mean_dice_avg = np.mean(mean_dice_per_class)
    mean_miou = np.mean(miou_scores_all)
    
    etime = time.time()
    log_info = f'Finish one epoch train: epoch {epoch}, loss: {mean_loss:.4f}, avg_dice: {mean_dice_avg:.4f}, mIoU: {mean_miou:.4f}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    
    return {
        'loss': mean_loss,
        'avg_dice': mean_dice_avg,
        'dice_per_class': mean_dice_per_class,
        'miou': mean_miou
    }


def val_one_epoch(test_datasets,
                    test_loader,
                    model,
                    epoch, 
                    logger,
                    config,
                    test_save_path,
                    val_or_test=False):
    # switch to evaluate mode
    stime = time.time()
    model.eval()
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        all_dice_scores = []
        all_hd95_scores = []
        
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, patch_size=[config.input_size_h, config.input_size_w],
                                    test_save_path=test_save_path, case=case_name, z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)
            
            # 收集每个case的dice和hd95分数
            case_dice_scores = [metric[0] for metric in metric_i]  # 每个类别的dice
            case_hd95_scores = [metric[1] for metric in metric_i]  # 每个类别的hd95
            
            all_dice_scores.append(case_dice_scores)
            all_hd95_scores.append(case_hd95_scores)

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
            
        metric_list = metric_list / len(test_datasets)
        
        # 计算平均指标
        avg_dice_per_class = np.mean(all_dice_scores, axis=0)  # 每个类别的平均dice
        avg_hd95_per_class = np.mean(all_hd95_scores, axis=0)  # 每个类别的平均hd95
        
        performance = np.mean(avg_dice_per_class)  # 所有类别的平均dice
        mean_hd95 = np.mean(avg_hd95_per_class)   # 所有类别的平均hd95
        
        # 计算mIoU (基于dice分数的近似)
        mean_miou = np.mean([dice / (2 - dice) if dice < 1.0 else 1.0 for dice in avg_dice_per_class])
        
        for i in range(len(avg_dice_per_class)):
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i+1, avg_dice_per_class[i], avg_hd95_per_class[i]))
            
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance:.4f}, mean_hd95: {mean_hd95:.4f}, mIoU: {mean_miou:.4f}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return {
        'avg_dice': performance,
        'dice_per_class': avg_dice_per_class,
        'avg_hd95': mean_hd95,
        'hd95_per_class': avg_hd95_per_class,
        'miou': mean_miou
    }


def val_one_epoch_slice(val_dataset,
                       val_loader,
                       model,
                       epoch, 
                       logger,
                       config,
                       test_save_path,
                       val_or_test=False):
    """
    使用slice-by-slice方式进行验证，与训练时的数据处理保持一致
    """
    stime = time.time()
    model.eval()
    
    all_dice_scores = []
    all_miou_scores = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            images, targets = data['image'], data['label']
            images = images.cuda(non_blocking=True).float()
            targets = targets.cuda(non_blocking=True).long()
            
            # 前向传播
            model_output = model(images)
            
            # 解包输出 - AFFSegNet返回两个值
            if isinstance(model_output, tuple) and len(model_output) == 2:
                out, intermediate_results = model_output
            else:
                out = model_output
            
            # 计算dice和mIoU
            dice_scores = calculate_dice_per_class(out, targets, config.num_classes)
            miou_score = calculate_miou(out, targets, config.num_classes)
            
            all_dice_scores.append(dice_scores)
            all_miou_scores.append(miou_score)
    
    # 计算平均指标
    avg_dice_per_class = np.mean(all_dice_scores, axis=0)  # 每个类别的平均dice
    avg_dice = np.mean(avg_dice_per_class)                # 总体平均dice
    avg_miou = np.mean(all_miou_scores)                   # 平均mIoU
    
    # HD95暂时设为0，因为slice-by-slice计算HD95不太合理
    avg_hd95_per_class = np.zeros_like(avg_dice_per_class)
    avg_hd95 = 0.0
    
    # 打印每个类别的性能
    organ_names = ['spleen', 'right_kidney', 'left_kidney', 'gallbladder', 
                   'esophagus', 'liver', 'stomach', 'aorta']
    
    for i, organ in enumerate(organ_names):
        if i < len(avg_dice_per_class):
            logger.info(f'Validation class {i+1} ({organ}): dice={avg_dice_per_class[i]:.4f}')
    
    etime = time.time()
    log_info = f'val_slice epoch: {epoch}, mean_dice: {avg_dice:.4f}, mIoU: {avg_miou:.4f}, slices: {len(all_dice_scores)}, time(s): {etime-stime:.2f}'
    print(log_info)
    logger.info(log_info)
    
    return {
        'avg_dice': avg_dice,
        'dice_per_class': avg_dice_per_class,
        'avg_hd95': avg_hd95,
        'hd95_per_class': avg_hd95_per_class,
        'miou': avg_miou
    }


def val_one_epoch_with_visualization(test_datasets,
                                   test_loader,
                                   model,
                                   epoch, 
                                   logger,
                                   config,
                                   test_save_path,
                                   prediction_vis_dir=None,
                                   attention_vis_dir=None,
                                   activation_vis_dir=None,
                                   val_or_test=False,
                                   save_vis_every_n=5,
                                   save_prediction_comparison_func=None,
                                   extract_attention_maps_func=None,
                                   save_attention_heatmaps_func=None,
                                   extract_activation_maps_func=None,
                                   save_activation_heatmaps_func=None):
    """
    验证函数，支持生成预测对比图、注意力热图和激活热图
    Args:
        save_vis_every_n: 每n个案例保存一次可视化图像
        save_prediction_comparison_func: 保存预测对比图的函数
        extract_attention_maps_func: 提取注意力图的函数
        save_attention_heatmaps_func: 保存注意力热图的函数
        extract_activation_maps_func: 提取激活图的函数
        save_activation_heatmaps_func: 保存激活热图的函数
        activation_vis_dir: 激活热图保存目录
    """
    
    stime = time.time()
    model.eval()
    
    with torch.no_grad():
        metric_list = 0.0
        i_batch = 0
        
        for data in tqdm(test_loader):
            img, msk, case_name = data['image'], data['label'], data['case_name'][0]
            
            # 执行预测并计算指标
            metric_i = test_single_volume(img, msk, model, classes=config.num_classes, 
                                        patch_size=[config.input_size_h, config.input_size_w],
                                        test_save_path=test_save_path, case=case_name, 
                                        z_spacing=config.z_spacing, val_or_test=val_or_test)
            metric_list += np.array(metric_i)
            
            # 每隔save_vis_every_n个案例生成可视化图像
            if i_batch % save_vis_every_n == 0 and (prediction_vis_dir or attention_vis_dir or activation_vis_dir):
                try:
                    # 生成预测结果用于可视化
                    img_np = img.squeeze(0).cpu().detach().numpy()
                    msk_np = msk.squeeze(0).cpu().detach().numpy()
                    
                    # 创建模型预测
                    if len(img_np.shape) == 3:  # 3D数据
                        prediction_np = np.zeros_like(msk_np)
                        
                        # 为3D数据生成注意力热图和激活热图 - 完整的3D数据处理
                        if attention_vis_dir and extract_attention_maps_func and save_attention_heatmaps_func:
                            try:
                                # 使用完整的3D数据生成注意力热图
                                if config.network.startswith('DWSegNet'):

                                    input_tensor = torch.from_numpy(img_np).unsqueeze(0).float().cuda()  # (D, H, W) -> (1, D, H, W)
                                    # 为3通道创建适当的输入格式，这里选择中间slice进行通道复制
                                    mid_slice = img_np[img_np.shape[0] // 2]
                                    x, y = mid_slice.shape[0], mid_slice.shape[1]
                                    if x != config.input_size_h or y != config.input_size_w:
                                        from scipy.ndimage import zoom
                                        mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                    input_tensor = torch.from_numpy(mid_slice).unsqueeze(0).float().cuda()  # (H, W) -> (1, H, W)
                                    input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, H, W) -> (3, H, W) -> (1, 3, H, W)
                                else:
                                    # VMUNet等其他模型使用1通道输入
                                    mid_slice = img_np[img_np.shape[0] // 2]
                                    x, y = mid_slice.shape[0], mid_slice.shape[1]
                                    if x != config.input_size_h or y != config.input_size_w:
                                        from scipy.ndimage import zoom
                                        mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                    input_tensor = torch.from_numpy(mid_slice).unsqueeze(0).unsqueeze(0).float().cuda()  # (H, W) -> (1, 1, H, W)
                                
                                attention_maps = extract_attention_maps_func(model, input_tensor)
                                if attention_maps:
                                    save_attention_heatmaps_func(
                                        img_np, attention_maps, case_name, attention_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成注意力热图失败 {case_name}: {e}")
                        
                        # 为3D数据生成激活热图
                        if activation_vis_dir and extract_activation_maps_func and save_activation_heatmaps_func:
                            try:
                                # 使用完整的3D数据生成激活热图
                                if config.network.startswith('DWSegNet'):
                                    mid_slice = img_np[img_np.shape[0] // 2]
                                    x, y = mid_slice.shape[0], mid_slice.shape[1]
                                    if x != config.input_size_h or y != config.input_size_w:
                                        from scipy.ndimage import zoom
                                        mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                    input_tensor = torch.from_numpy(mid_slice).unsqueeze(0).float().cuda()  # (H, W) -> (1, H, W)
                                    input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, H, W) -> (3, H, W) -> (1, 3, H, W)
                                else:
                                    # VMUNet等其他模型使用1通道输入
                                    mid_slice = img_np[img_np.shape[0] // 2]
                                    x, y = mid_slice.shape[0], mid_slice.shape[1]
                                    if x != config.input_size_h or y != config.input_size_w:
                                        from scipy.ndimage import zoom
                                        mid_slice = zoom(mid_slice, (config.input_size_h / x, config.input_size_w / y), order=3)
                                    input_tensor = torch.from_numpy(mid_slice).unsqueeze(0).unsqueeze(0).float().cuda()  # (H, W) -> (1, 1, H, W)
                                
                                activation_maps = extract_activation_maps_func(model, input_tensor)
                                if activation_maps:
                                    save_activation_heatmaps_func(
                                        img_np, activation_maps, case_name, activation_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成激活热图失败 {case_name}: {e}")
                        
                        # 生成每个slice的预测
                        for ind in range(img_np.shape[0]):
                            slice_img = img_np[ind, :, :]
                            x, y = slice_img.shape[0], slice_img.shape[1]
                            
                            # 调整尺寸
                            if x != config.input_size_h or y != config.input_size_w:
                                from scipy.ndimage import zoom
                                slice_img = zoom(slice_img, (config.input_size_h / x, config.input_size_w / y), order=3)
                            
                            # 根据网络类型处理输入通道数
                            if config.network.startswith('DWSegNet'):
                                # CUMamba系列模型需要3通道输入
                                input_tensor = torch.from_numpy(slice_img).unsqueeze(0).float().cuda()  # (H, W) -> (1, H, W)
                                input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, H, W) -> (3, H, W) -> (1, 3, H, W)
                            else:
                                # VMUNet等其他模型使用1通道输入
                                input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()  # (H, W) -> (1, 1, H, W)
                            
                            # 获取预测结果
                            outputs = model(input_tensor)
                            pred = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                            
                            # 调整回原尺寸
                            if x != config.input_size_h or y != config.input_size_w:
                                pred = zoom(pred, (x / config.input_size_h, y / config.input_size_w), order=0)
                            
                            prediction_np[ind] = pred
                    else:
                        # 2D数据处理
                        # 根据网络类型处理输入通道数
                        if config.network.startswith('DWSegNet'):
                            input_tensor = torch.from_numpy(img_np).unsqueeze(0).float().cuda()  # (H, W) -> (1, H, W)
                            input_tensor = input_tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, H, W) -> (3, H, W) -> (1, 3, H, W)
                        else:
                            # VMUNet等其他模型使用1通道输入
                            input_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().cuda()  # (H, W) -> (1, 1, H, W)
                        
                        outputs = model(input_tensor)
                        prediction_np = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()
                        
                        # 生成注意力热图
                        if attention_vis_dir and extract_attention_maps_func and save_attention_heatmaps_func:
                            try:
                                attention_maps = extract_attention_maps_func(model, input_tensor)
                                if attention_maps:
                                    save_attention_heatmaps_func(
                                        img_np, attention_maps, case_name, attention_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成注意力热图失败 {case_name}: {e}")
                        
                        # 生成激活热图
                        if extract_activation_maps_func and save_activation_heatmaps_func:
                            try:
                                activation_maps = extract_activation_maps_func(model, input_tensor)
                                if activation_maps:
                                    save_activation_heatmaps_func(
                                        img_np, activation_maps, case_name, 
                                        activation_vis_dir
                                    )
                            except Exception as e:
                                logger.warning(f"生成激活热图失败 {case_name}: {e}")
                    
                    # 保存预测对比图
                    if prediction_vis_dir and save_prediction_comparison_func:
                        try:
                            save_prediction_comparison_func(
                                img_np, msk_np, prediction_np, case_name, prediction_vis_dir
                            )
                            logger.info(f"已保存预测对比图: {case_name}")
                        except Exception as e:
                            logger.warning(f"保存预测对比图失败 {case_name}: {e}")
                            
                except Exception as e:
                    logger.warning(f"可视化处理失败 {case_name}: {e}")

            logger.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name,
                        np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
            i_batch += 1
        
        metric_list = metric_list / len(test_datasets)
        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]
        
        # 提取每个器官的dice和hd95
        dice_per_organ = []
        hd95_per_organ = []
        
        for i in range(1, config.num_classes):
            dice_score = metric_list[i-1][0]
            hd95_score = metric_list[i-1][1]
            dice_per_organ.append(dice_score)
            hd95_per_organ.append(hd95_score)
            logger.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, dice_score, hd95_score))
        
        etime = time.time()
        log_info = f'val epoch: {epoch}, mean_dice: {performance}, mean_hd95: {mean_hd95}, time(s): {etime-stime:.2f}'
        print(log_info)
        logger.info(log_info)
    
    return performance, mean_hd95, dice_per_organ, hd95_per_organ

