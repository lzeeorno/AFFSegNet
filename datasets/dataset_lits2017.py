import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch.nn.functional as F
from scipy import ndimage
import random
from sklearn.model_selection import KFold
import glob

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class LitsRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')  # 保留case_name

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        h, w = image.shape
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 伪HDR三通道预处理
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        # 创建三个通道：原始、增强对比度、平滑
        channel1 = image_tensor  # 原始图像
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
        
        # 合并为三通道
        image = torch.stack([channel1, channel2, channel3], dim=0)
        
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'case_name': case_name}  # 保留case_name
        return sample

class LiTS2017_dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, fold=None, num_folds=5, cross_validation=False):
        """
        LITS2017预处理PNG数据集加载器

        Args:
            data_dir: 数据根目录，包含trainImage_lits2017_png/trainMask_lits2017_png
                     和testImage_lits2017_png/testMask_lits2017_png子目录
            split: 'train', 'val', 'test'之一
            transform: 数据变换
            fold, num_folds, cross_validation: 为兼容旧接口保留，但当前版本默认
                     使用 train 目录内部的 80/20 划分做 train/val，test 目录只
                     用于 split='test' 时的测试，不再对所有样本做KFold划分。
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 兼容旧接口参数但不再使用KFold做全量划分
        self.fold = fold
        self.num_folds = num_folds
        self.cross_validation = cross_validation

        # 按用户磁盘结构区分 train/test 四个目录
        if split in ['train', 'val']:
            self.ct_dir = os.path.join(data_dir, 'trainImage_lits2017_png')
            self.mask_dir = os.path.join(data_dir, 'trainMask_lits2017_png')
        elif split == 'test':
            self.ct_dir = os.path.join(data_dir, 'testImage_lits2017_png')
            self.mask_dir = os.path.join(data_dir, 'testMask_lits2017_png')
        else:
            raise ValueError(f"不支持的split: {split}，必须是'train'/'val'/'test'")

        ct_files = sorted(glob.glob(os.path.join(self.ct_dir, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))

        assert len(ct_files) == len(mask_files), f"CT文件数({len(ct_files)})与mask文件数({len(mask_files)})不一致"

        # 提取基名并验证配对
        all_samples = []
        for ct_file in ct_files:
            ct_name = os.path.basename(ct_file)
            mask_file = os.path.join(self.mask_dir, ct_name)
            if os.path.exists(mask_file):
                all_samples.append(ct_name[:-4])

        print(f"LITS2017 {split} 目录下共找到 {len(all_samples)} 个有效PNG样本")

        # 仅对 train 目录内部做 train/val 划分；test 目录完全作为测试集
        if split in ['train', 'val']:
            np.random.seed(42)
            shuffled = np.array(all_samples)
            np.random.shuffle(shuffled)

            n_total = len(shuffled)
            # 用户说明：train 文件夹 19206 张 PNG 用于 train+val
            # 这里按照 80%/20% 拆分
            n_train = int(0.8 * n_total)

            if split == 'train':
                self.samples = shuffled[:n_train].tolist()
            else:  # 'val'
                self.samples = shuffled[n_train:].tolist()
        else:  # 'test'
            # 测试集直接使用 test 目录全部样本
            self.samples = all_samples

        print(f"LITS2017 {split} split: {len(self.samples)} 样本")
        if len(self.samples) <= 20:
            print(f"样本列表: {self.samples}")
        else:
            print(f"样本示例: {self.samples[:10]}...")

    def _get_fold_samples(self, all_samples, split, fold, num_folds):
        """
        获取指定fold的样本列表
        
        实现策略：
        1. 对19206个样本进行五折交叉验证
        2. 每个fold：训练数据约80%，验证数据约20%
        3. 测试数据与验证数据使用相同的样本
        
        Args:
            all_samples: 所有可用的样本列表
            split: 'train', 'val', 'test'
            fold: 当前fold编号
            num_folds: 总fold数
            
        Returns:
            当前fold对应的样本列表
        """
        # 为了保证可重复性，对样本进行排序
        sorted_samples = sorted(all_samples)
        
        # 使用KFold进行分割
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_splits = list(kf.split(sorted_samples))
        
        if fold >= len(fold_splits):
            raise ValueError(f"Fold {fold} 超出范围，总共只有 {len(fold_splits)} 个folds")
        
        train_indices, val_indices = fold_splits[fold]
        
        # 获取当前fold的训练和验证数据
        train_samples = [sorted_samples[i] for i in train_indices]
        val_samples = [sorted_samples[i] for i in val_indices]
        
        print(f"Fold {fold}: 训练{len(train_samples)}个样本, 验证/测试{len(val_samples)}个样本")
        
        if split == 'train':
            return train_samples
        elif split == 'val':
            return val_samples
        else:  # test
            # 测试数据与验证数据相同
            return val_samples

    def _normalize_intensity(self, image):
        """PNG图像标准化到[0, 1]"""
        image = image.astype(np.float32) / 255.0
        return image
    
    def _process_mask(self, mask):
        """处理掩码，将像素值映射到类别标签"""
        # 原始像素值：0(背景), 150(肝脏), 255(肿瘤)
        # 映射到：0(背景), 1(肝脏), 2(肿瘤)
        processed_mask = np.zeros_like(mask, dtype=np.uint8)
        processed_mask[mask == 150] = 1  # 肝脏
        processed_mask[mask == 255] = 2  # 肿瘤
        return processed_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # 加载CT图像
        ct_path = os.path.join(self.ct_dir, f'{sample_name}.png')
        ct_image = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        if ct_image is None:
            raise ValueError(f"无法加载CT图像: {ct_path}")
        
        # 加载掩码
        mask_path = os.path.join(self.mask_dir, f'{sample_name}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {mask_path}")
        
        # 标准化CT图像
        ct_image = self._normalize_intensity(ct_image)
        
        # 处理掩码
        mask = self._process_mask(mask)
        
        sample = {
            'image': ct_image,
            'label': mask,
            'case_name': sample_name
        }
        
        if self.transform:
            sample = self.transform(sample)
        else:
            # 如果没有transform，也需要创建三通道图像
            image_tensor = torch.from_numpy(ct_image.astype(np.float32))
            channel1 = image_tensor
            channel2 = torch.clamp(image_tensor * 1.2, 0, 1)
            channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                                   kernel_size=3, stride=1, padding=1).squeeze()
            
            sample['image'] = torch.stack([channel1, channel2, channel3], dim=0)
            sample['label'] = torch.from_numpy(mask.astype(np.int64))
        
        return sample

# 为了兼容性，保持原有的导入接口
def get_lits_dataloader(data_dir, split, batch_size=1, num_workers=4, 
                       fold=None, cross_validation=False, transform=None):
    """获取LITS2017数据加载器的便捷函数"""
    from torch.utils.data import DataLoader
    
    dataset = LiTS2017_dataset(
        data_dir=data_dir,
        split=split,
        transform=transform,
        fold=fold,
        cross_validation=cross_validation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataset, dataloader 

class LitsValTransform(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case_name = sample.get('case_name', '')

        # resize 到固定大小
        h, w = image.shape
        if h != self.output_size[0] or w != self.output_size[1]:
            image = cv2.resize(image, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        channel1 = image_tensor
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0),
                                kernel_size=3, stride=1, padding=1).squeeze()
        image = torch.stack([channel1, channel2, channel3], dim=0)

        label = torch.from_numpy(label.astype(np.float32))
        return {'image': image, 'label': label.long(), 'case_name': case_name}