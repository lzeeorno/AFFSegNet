from pickle import FALSE
from torchvision import transforms
from datasets.dataset_bladder import *
from utils import *

from datetime import datetime
import ml_collections

class setting_config:
    """
    Bladder数据集训练配置
    """
    # 模型选择配置 - 新增功能
    available_models = ['DWSegNet', 'AFFSegNet']  # 可用的模型列表
    network = 'AFFSegNet'  # 默认使用AFFSegNet，可通过set_model方法切换
    
    model_config = {
        'num_classes': 3,  # 背景、膀胱、肿瘤
        'input_channels': 3, 
        'feature_size': 48,
        'use_boundary_refinement': False,
        'load_ckpt_path': '',  # 不使用预训练权重
    }
    datasets_name = 'Bladder'
    input_size_h = 512
    input_size_w = 512
    
    # 数据路径配置
    data_path = './data/Bladder/'
    datasets = Bladder_dataset
    
    # 测试相关配置
    test_mode = 'best'  # 测试权重类型: 'best' 或 'latest'
    test_weights_path = 'results/AFFSegNet_Bladder/checkpoints/epoch_100.pth'  # 如果指定具体路径，则使用此路径；否则自动构建
    test_dataset_split = 'test'  # 测试数据集分割: 'test' 或 'val'
    
    # 新增保存间隔配置
    save_interval = 50  # 每10个epoch保存一次权重
    
    def __init__(self):
        # 动态设置工作目录
        self.work_dir = f'results/{self.network}_{self.datasets_name}'
        
        # 动态设置测试权重路径
        if not self.test_weights_path:
            self.test_weights_path = f'results/{self.network}_{self.datasets_name}/checkpoints/{self.test_mode}.pth'
    
    pretrained_path = ''
    num_classes = 3  # 修改为3分类（背景、膀胱、肿瘤）
    loss_weight = [0.4, 0.5, 0.6]  # CE, Dice, Focal weights
    criterion = CeDiceFocalLoss(num_classes, loss_weight, ignore_index=-1)  # 不使用ignore_index
    z_spacing = 1
    input_channels = 3

    # 分布式训练配置
    distributed = False
    local_rank = -1
    num_workers = 2
    seed = 2050
    world_size = None
    rank = None
    amp = False

    # 训练参数
    batch_size = 8  # 适中的batch size
    epochs = 300
    resume_training = False  # 是否恢复训练，False表示重新开始训练
    print_interval = 20  # 打印间隔
    val_interval = 20  # 每1个epoch验证一次

    threshold = 0.5

    # 优化器配置
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    
    if opt == 'AdamW':
        lr = 3e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    
    # 学习率调度器配置
    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        T_max = 300  
        eta_min = 1e-6
        last_epoch = -1

    @classmethod
    def set_model(cls, model_name):
        """设置使用的模型"""
        if model_name not in cls.available_models:
            raise ValueError(f"模型必须是以下之一: {cls.available_models}")
        cls.network = model_name
        print(f"设置模型为: {model_name}")
        
        # 更新工作目录和测试权重路径
        cls.work_dir = f'results/{cls.network}_{cls.datasets_name}'
        
        # 更新测试权重路径
        if not hasattr(cls, '_custom_test_weights_path') or not cls._custom_test_weights_path:
            cls.test_weights_path = f'results/{cls.network}_{cls.datasets_name}/checkpoints/{cls.test_mode}.pth'
    
    @classmethod
    def set_save_interval(cls, interval):
        """设置保存间隔"""
        if interval <= 0:
            raise ValueError("保存间隔必须大于0")
        cls.save_interval = interval
        print(f"设置保存间隔为每{interval}个epoch保存一次")
    
    @classmethod
    def set_test_config(cls, test_mode=None, test_weights_path=None, test_dataset_split=None):
        """设置测试相关配置"""
        if test_mode is not None:
            if test_mode not in ['best', 'latest']:
                raise ValueError("测试模式必须是 'best' 或 'latest'")
            cls.test_mode = test_mode
            print(f"设置测试模式为: {test_mode}")
        
        if test_weights_path is not None:
            cls.test_weights_path = test_weights_path
            cls._custom_test_weights_path = True
            print(f"设置测试权重路径为: {test_weights_path}")
        
        if test_dataset_split is not None:
            if test_dataset_split not in ['test', 'val']:
                raise ValueError("测试数据集分割必须是 'test' 或 'val'")
            cls.test_dataset_split = test_dataset_split
            print(f"设置测试数据集分割为: {test_dataset_split}")
    
    def get_test_weights_path(self):
        """获取测试权重路径"""
        if self.test_weights_path and self.test_weights_path != '':
            return self.test_weights_path
        else:
            return f'results/{self.network}_{self.datasets_name}/checkpoints/{self.test_mode}.pth' 