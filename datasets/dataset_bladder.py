import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import cv2

class Bladder_dataset(Dataset):
    """
    Bladder数据集加载器
    数据格式：PNG图像，512x512分辨率
    类别：背景(0)，膀胱(1)，肿瘤(2)
    原始像素值：背景(0)，膀胱(128)，肿瘤(255)
    映射后：背景(0)->0，膀胱(128)->1，肿瘤(255)->2
    """
    
    def __init__(self, args, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.args = args
        
        # 设置图像和标签路径
        if mode == 'train':
            self.image_dir = os.path.join(data_path, 'train', 'Images')
            self.label_dir = os.path.join(data_path, 'train', 'labels')
        elif mode == 'test':
            self.image_dir = os.path.join(data_path, 'test', 'Images')
            self.label_dir = os.path.join(data_path, 'test', 'Labels')  # 注意大写L
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 获取所有图像文件名
        self.image_list = []
        if os.path.exists(self.image_dir):
            self.image_list = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
            self.image_list.sort()
        
        # 验证对应的标签文件是否存在
        valid_images = []
        for img_name in self.image_list:
            label_path = os.path.join(self.label_dir, img_name)
            if os.path.exists(label_path):
                valid_images.append(img_name)
            else:
                print(f"警告: 找不到对应的标签文件: {label_path}")
        
        self.image_list = valid_images
        print(f"Bladder数据集 {mode} 模式: 加载了 {len(self.image_list)} 个样本")
        
        # 数据增强
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
            
            self.label_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            
            self.label_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        
        # 加载标签
        label_path = os.path.join(self.label_dir, img_name)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if label is None:
            raise ValueError(f"无法加载标签: {label_path}")
        
        # 确保图像和标签都是512x512
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        if label.shape != (512, 512):
            label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 标签映射处理：背景(0)->0，膀胱(128)->1，肿瘤(255)->2
        final_label = np.zeros_like(label, dtype=np.uint8)
        final_label[label == 0] = 0    # 背景 -> 0
        final_label[label == 128] = 1  # 膀胱 -> 1  
        final_label[label == 255] = 2  # 肿瘤 -> 2
        
        # 转换为3通道（伪HDR处理）
        image_3ch = np.stack([image, image, image], axis=2)
        
        # 应用数据增强
        if self.mode == 'train' and random.random() > 0.5:
            # 同步随机变换
            seed = random.randint(0, 2**32)
            
            # 设置随机种子确保图像和标签同步变换
            random.seed(seed)
            torch.manual_seed(seed)
            image_tensor = self.transform(image_3ch)
            
            random.seed(seed)
            torch.manual_seed(seed)
            label_tensor = self.label_transform(final_label)
        else:
            image_tensor = self.transform(image_3ch)
            label_tensor = self.label_transform(final_label)
        
        # 标签处理
        label_tensor = (label_tensor * 255).long().squeeze(0)  # 转换为long类型，去掉通道维度
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'case_name': img_name
        }


def get_loader(args):
    """
    获取Bladder数据集的数据加载器
    """
    from configs.config_setting_bladder import setting_config as config
    
    # 训练数据集
    train_dataset = Bladder_dataset(
        args=args,
        data_path=config.data_path,
        mode='train'
    )
    
    # 测试数据集
    test_dataset = Bladder_dataset(
        args=args,
        data_path=config.data_path,
        mode='test'
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # 测试时使用batch_size=1
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


# 膀胱分割的类别数量
bladder_num_classes = 3 