from pickle import FALSE
from torchvision import transforms
from datasets.dataset_lits2017 import *
from utils import *

from datetime import datetime
import ml_collections

class setting_config:
    """
    LITS2017数据集训练配置
    """
    # 模型选择配置 - 新增功能
    available_models = ['DWSegNet', 'AFFSegNet']  # 可用的模型列表
    network = 'AFFSegNet'  # 默认使用AFFSegNet，可通过set_model方法切换
    
    model_config = {
        'num_classes': 3,  # 背景、肝脏、肿瘤
        'input_channels': 3, 
        'feature_size': 48,
        'use_boundary_refinement': False,
        'load_ckpt_path': '',  # 不使用预训练权重
    }
    datasets_name = 'LiTS2017'
    input_size_h = 256
    input_size_w = 256
    
    # 数据路径配置
    data_path = './data/LITS2017_nii/'
    datasets = LiTS2017_dataset
    
    # 五折交叉验证配置
    num_folds = 5  # 交叉验证折数
    current_fold = 0  # 当前使用的fold (0-4)，可通过命令行或手动设置
    cross_validation = True  # 是否启用交叉验证模式
    
    # 测试相关配置
    test_fold = 0  # 测试时使用的fold (0-4)
    test_mode = 'best'  # 测试权重类型: 'best' 或 'latest'
    # test_weights_path = 'results/AFFSegNet_LiTS2017_fold0/checkpoints/best.pth'  # 如果指定具体路径，则使用此路径；否则自动根据test_fold和test_mode构建
    test_dataset_split = 'test'  # 测试数据集分割: 'test' 或 'val'
    
    # 新增保存间隔配置
    save_interval = 3  # 每3个epoch保存一次权重
    
    # 测试配置 - 新增默认测试参数，避免每次输入命令行
    test_config = {
        'model': 'AFFSegNet',  # 测试使用的模型
        'fold': 0,  # 测试使用的fold (0-4)
        'weights': 'results/AFFSegNet_LiTS2017_fold0/checkpoints/epoch_066_dice0.95.pth',  # 测试权重路径
        'mode': 'best',  # 测试权重类型: 'best' 或 'latest'
        'split': 'test',  # 测试数据集分割: 'test' 或 'val'
        'save_vis': True,  # 是否保存可视化结果
        'num_vis': 88  # 保存可视化样本的数量
    }
    
    def __init__(self):
        # 动态设置工作目录
        if self.cross_validation:
            self.work_dir = f'results/{self.network}_{self.datasets_name}_fold{self.current_fold}'
        else:
            self.work_dir = f'results/{self.network}_{self.datasets_name}'
        
        # 动态设置测试权重路径
        if not self.test_weights_path:
            self.test_weights_path = f'results/{self.network}_{self.datasets_name}_fold{self.test_fold}/checkpoints/{self.test_mode}.pth'
    
    pretrained_path = ''
    num_classes = 3
    loss_weight = [0.3, 0.4, 0.3]  # CE, Dice, Focal weights
    criterion = CeDiceFocalLoss(num_classes, loss_weight, ignore_index=-1)  # 使用三重损失函数处理类别不平衡
    z_spacing = 1
    input_channels = 3

    # 分布式训练配置
    distributed = False
    local_rank = -1
    num_workers = 2  # 改为0，避免多进程内存问题
    seed = 2050
    world_size = None
    rank = None
    amp = False

    # 训练参数
    batch_size = 34  # 减小batch size，避免第一次加载太慢
    epochs = 100
    resume_training = False  # 是否恢复训练，False表示重新开始训练
    print_interval = 5  # 减少打印间隔，增加输出频率
    val_interval = 10  # 每1个epoch验证一次，便于测试

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
    def set_fold(cls, fold):
        """设置当前fold并更新相关配置"""
        if fold < 0 or fold >= cls.num_folds:
            raise ValueError(f"Fold必须在0-{cls.num_folds-1}之间")
        cls.current_fold = fold
        print(f"设置当前fold为: {fold}")
    
    @classmethod
    def set_test_config(cls, test_fold=None, test_mode=None, test_weights_path=None, test_dataset_split=None):
        """设置测试相关配置"""
        if test_fold is not None:
            if test_fold < 0 or test_fold >= cls.num_folds:
                raise ValueError(f"测试fold必须在0-{cls.num_folds-1}之间")
            cls.test_fold = test_fold
            print(f"设置测试fold为: {test_fold}")
        
        if test_mode is not None:
            if test_mode not in ['best', 'latest']:
                raise ValueError("测试模式必须是 'best' 或 'latest'")
            cls.test_mode = test_mode
            print(f"设置测试模式为: {test_mode}")
        
        if test_weights_path is not None:
            cls.test_weights_path = test_weights_path
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
            return f'results/{self.network}_{self.datasets_name}_fold{self.test_fold}/checkpoints/{self.test_mode}.pth' 

    @classmethod
    def set_model(cls, model_name):
        """设置使用的模型"""
        if model_name not in cls.available_models:
            raise ValueError(f"模型必须是以下之一: {cls.available_models}")
        cls.network = model_name
        print(f"设置模型为: {model_name}")
        
        # 更新工作目录和测试权重路径
        if cls.cross_validation:
            cls.work_dir = f'results/{cls.network}_{cls.datasets_name}_fold{cls.current_fold}'
        else:
            cls.work_dir = f'results/{cls.network}_{cls.datasets_name}'
        
        # 更新测试权重路径
        if not hasattr(cls, '_custom_test_weights_path') or not cls._custom_test_weights_path:
            cls.test_weights_path = f'results/{cls.network}_{cls.datasets_name}_fold{cls.test_fold}/checkpoints/{cls.test_mode}.pth'
    
    @classmethod
    def set_save_interval(cls, interval):
        """设置保存间隔"""
        if interval <= 0:
            raise ValueError("保存间隔必须大于0")
        cls.save_interval = interval
        print(f"设置保存间隔为每{interval}个epoch保存一次")
        
    @classmethod
    def get_save_info(cls):
        """获取当前保存设置信息"""
        return {
            'save_interval': cls.save_interval,
            'val_interval': cls.val_interval,
            'epochs': cls.epochs
        } 