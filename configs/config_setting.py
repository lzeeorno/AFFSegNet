from PyInstaller.log import level
from albumentations import Sharpen
from torchvision import transforms
from utils import *
from datetime import datetime
import torch
import torchvision.transforms.functional as TF
import math



class setting_config:
    """
    the config of training setting.
    """
    network = 'vmunet'
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        # 'load_ckpt_path': './results/vmunet_isic18_Wednesday_28_August_2024_15h_59m_03s/checkpoints/best-epoch29-loss0.4166.pth',
        # 'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        'load_ckpt_path': './pre_trained_weights/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth',
    }

    datasets = 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 512
    input_size_w = 512
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 2042
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 10 #default: 32 with imgSize 256,
    epochs = 30

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 38
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    '''
    Data Augmentation
    '''
    def make_odd(num):
        num = math.ceil(num)
        if num % 2 == 0:
            num += 1
        return num

    level = 5
    #
    # train_transformer = transforms.Compose([
    #     myNormalize(datasets, train=True),
    #     myToTensor(),
    #     transforms.RandomApply([transforms.ColorJitter(brightness=0.04 * level)], p=0.2 * level),  # Brightness jitter
    #     transforms.RandomApply([transforms.ColorJitter(contrast=0.04 * level)], p=0.2 * level),  # Contrast jitter
    #     transforms.RandomPosterize(bits=math.floor(8 - 0.8 * level), p=0.2 * level),  # Posterize approximation
    #     transforms.RandomAdjustSharpness(sharpness_factor=(1 + 0.04 * level), p=0.2 * level),  # Sharpen approximation
    #     transforms.RandomApply([transforms.GaussianBlur(kernel_size=make_odd(3 + 0.8 * level), sigma=(0.1, 2.0))], p=min(0.2 * level, 1)),  # GaussianBlur approximation
    #     transforms.Lambda(lambda img: img + torch.randn(img.size()) * random.uniform(2 * level, 10 * level)),  # GaussNoise approximation
    #     transforms.RandomRotation(degrees=4 * level, expand=False, center=None, fill=0),  # Rotate approximation
    #     transforms.RandomHorizontalFlip(p=0.2 * level),  # HorizontalFlip
    #     transforms.RandomVerticalFlip(p=0.2 * level),  # VerticalFlip
    #     transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(1 - 0.04 * level, 1 + 0.04 * level), shear=None, fill=0)], p=0.2 * level),  # Affine approximation
    #     transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(0, 2 * level), fill=0)], p=0.2 * level),  # Shear x approximation
    #     transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=(0, 0, 0, 2 * level), fill=0)], p=0.2 * level),  # Shear y approximation
    #     transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.02 * level, 0), fill=0)], p=0.2 * level),  # Translate x approximation
    #     transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0, 0.02 * level), fill=0)], p=0.2 * level),  # Translate y approximation
    #     transforms.RandomChoice([
    #         transforms.Lambda(lambda img: img),  # Placeholder for ElasticTransform approximation
    #         transforms.Lambda(lambda img: img),  # Placeholder for GridDistortion approximation
    #         transforms.Lambda(lambda img: img)   # Placeholder for OpticalDistortion approximation
    #     ], p=[0.1, 0.1, 0.1]),  # RandomChoice with placeholders
    #
    #     myResize(input_size_h, input_size_w)  # Custom resize
    # ])


    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'SGD'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.98 # default: 0 – momentum factor
        weight_decay = 1e-6 # default: 0.05 – weight decay (L2 penalty)
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 100 # – Maximum number of iterations. Cosine function period.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 6e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
