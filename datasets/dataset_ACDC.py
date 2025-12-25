import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import torch.nn.functional as F
from scipy import ndimage
import random
from sklearn.model_selection import train_test_split

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

class ACDCRandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 使用最近邻插值保持标签值
        
        # 修复zoom后可能的标签值问题
        # 将无效的标签值(-1之外的负值)设为-1
        label = np.where(label < -1, -1, label)
        # 将超出范围的标签值限制在有效范围内
        label = np.where(label > 2, -1, label)
        
        # 伪HDR三通道预处理
        image_tensor = torch.from_numpy(image.astype(np.float32))
        
        # 创建三个通道：原始、增强对比度、平滑
        channel1 = image_tensor  # 原始图像
        channel2 = torch.clamp(image_tensor * 1.2, 0, 1)  # 增强对比度
        channel3 = F.avg_pool2d(image_tensor.unsqueeze(0).unsqueeze(0), 
                               kernel_size=3, stride=1, padding=1).squeeze()  # 平滑
        
        # 合并为三通道
        image = torch.stack([channel1, channel2, channel3], dim=0)
        
        label = torch.from_numpy(label.astype(np.int64))  # 改为int64，直接使用映射后的标签
        sample = {'image': image, 'label': label}
        return sample

class ACDC_dataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        ACDC数据集加载器
        
        Args:
            data_dir: 数据根目录，包含database子目录
            split: 'train', 'val', 'test'之一
            transform: 数据变换
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # ACDC数据在database/training目录下
        if split in ['train', 'val']:
            data_path = os.path.join(data_dir, 'database', 'training')
        else:  # test
            data_path = os.path.join(data_dir, 'database', 'testing')
        
        # 获取所有患者目录
        all_patients = []
        for patient_dir in sorted(os.listdir(data_path)):
            if patient_dir.startswith('patient'):
                patient_path = os.path.join(data_path, patient_dir)
                if os.path.isdir(patient_path):
                    all_patients.append(patient_path)
        
        # 按照90%训练，10%验证划分训练数据（测试数据单独处理）
        if split in ['train', 'val']:
            # 使用train_test_split进行随机划分
            train_patients, val_patients = train_test_split(
                all_patients, test_size=0.1, random_state=42
            )
            
            if split == 'train':
                self.patient_dirs = train_patients
            else:  # val
                self.patient_dirs = val_patients
        else:  # test - 使用所有测试患者
            self.patient_dirs = all_patients
        
        print(f"ACDC {split} split: {len(self.patient_dirs)} patients")
        
        # 生成所有的样本（每个患者的每个时间帧）
        self.samples = self._generate_samples()

    def _generate_samples(self):
        """为数据集生成样本列表"""
        samples = []
        
        for patient_dir in self.patient_dirs:
            patient_id = os.path.basename(patient_dir)
            
            # 获取该患者的所有文件
            files = os.listdir(patient_dir)
            
            # 找到image和gt文件对
            image_files = [f for f in files if f.endswith('.nii.gz') and '_gt' not in f and '_4d' not in f]
            
            for img_file in image_files:
                # 对应的gt文件
                gt_file = img_file.replace('.nii.gz', '_gt.nii.gz')
                
                img_path = os.path.join(patient_dir, img_file)
                gt_path = os.path.join(patient_dir, gt_file)
                
                if os.path.exists(img_path) and os.path.exists(gt_path):
                    if self.split == 'train':
                        # 训练模式下，生成每个切片的样本
                        try:
                            img_data = nib.load(img_path).get_fdata()
                            num_slices = img_data.shape[2] if len(img_data.shape) > 2 else 1
                            
                            for slice_idx in range(num_slices):
                                samples.append((patient_id, img_file, gt_file, slice_idx))
                        except Exception as e:
                            print(f"处理 {img_path} 时出错: {e}")
                    else:
                        # 验证和测试模式下，每个文件作为一个样本
                        samples.append((patient_id, img_file, gt_file))
        
        return samples

    def _normalize_intensity(self, image):
        """强度标准化"""
        # 对心脏MRI进行标准化
        if np.sum(image > 0) > 0:
            mean = np.mean(image[image > 0])  # 只考虑非零像素
            std = np.std(image[image > 0])
            if std > 0:
                image = (image - mean) / std
        # 归一化到[0, 1]
        image_min, image_max = image.min(), image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.split == 'train':
            patient_id, img_file, gt_file, slice_idx = self.samples[idx]
            
            # 构建正确的路径
            patient_dir = None
            for p_dir in self.patient_dirs:
                if os.path.basename(p_dir) == patient_id:
                    patient_dir = p_dir
                    break
            
            if patient_dir is None:
                raise ValueError(f"找不到患者目录: {patient_id}")
            
            img_path = os.path.join(patient_dir, img_file)
            gt_path = os.path.join(patient_dir, gt_file)
            
            # 加载数据
            img_data = nib.load(img_path).get_fdata()
            gt_data = nib.load(gt_path).get_fdata()
            
            # 提取2D切片
            if len(img_data.shape) > 2:
                image = img_data[:, :, slice_idx]
                label = gt_data[:, :, slice_idx]
            else:
                image = img_data
                label = gt_data
            
            # 强度标准化
            image = self._normalize_intensity(image)
            
            # 标签重新映射：去除背景类
            # 原始标签：0=背景, 1=右心室, 2=心肌, 3=左心室
            # 新标签：0=右心室, 1=心肌, 2=左心室
            # 背景像素设为-1，在计算损失时会被忽略
            label_remapped = np.full_like(label, -1, dtype=np.int64)  # 初始化为-1
            label_remapped[label == 1] = 0  # 右心室
            label_remapped[label == 2] = 1  # 心肌  
            label_remapped[label == 3] = 2  # 左心室
            
            sample = {'image': image, 'label': label_remapped}
            
            if self.transform:
                sample = self.transform(sample)
            
            # 为训练模式添加case_name
            case_name = f"{patient_id}_{img_file.replace('.nii.gz', '')}_{slice_idx:03d}"
            sample['case_name'] = case_name
            
            return sample
        
        else:  # validation/test mode
            patient_id, img_file, gt_file = self.samples[idx]
            
            # 构建正确的路径
            patient_dir = None
            for p_dir in self.patient_dirs:
                if os.path.basename(p_dir) == patient_id:
                    patient_dir = p_dir
                    break
                    
            if patient_dir is None:
                raise ValueError(f"找不到患者目录: {patient_id}")
            
            img_path = os.path.join(patient_dir, img_file)
            gt_path = os.path.join(patient_dir, gt_file)
            
            # 加载3D数据
            img_data = nib.load(img_path).get_fdata()
            gt_data = nib.load(gt_path).get_fdata()
            
            # 强度标准化
            img_data = self._normalize_intensity(img_data)
            
            # 标签重新映射：去除背景类
            gt_remapped = np.full_like(gt_data, -1, dtype=np.int64)
            gt_remapped[gt_data == 1] = 0  # 右心室
            gt_remapped[gt_data == 2] = 1  # 心肌  
            gt_remapped[gt_data == 3] = 2  # 左心室
            
            # 转换为tensor并添加batch维度
            image = torch.from_numpy(img_data.astype(np.float32)).unsqueeze(0)  # [1, H, W, D]
            label = torch.from_numpy(gt_remapped.astype(np.int64)).unsqueeze(0)  # [1, H, W, D]
            
            case_name = f"{patient_id}_{img_file.replace('.nii.gz', '')}"
            
            return {
                'image': image,
                'label': label,
                'case_name': case_name
            } 