# -*- coding: utf-8 -*-
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import datetime
from unittest import TextTestRunner

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.nn as nn


from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset.dataset import Dataset_synapse_png

from utilities.metrics import dice_coef_synapse, batch_iou, mean_iou, iou_score, hd95_2d
import utilities.losses as losses
from utilities.utils import str2bool, count_params, load_pretrained_weights
import pandas as pd
from net.ASSNet import ASSNet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    # data preprocessing
    parser.add_argument('--upper', default=200)
    parser.add_argument('--lower', default=-200)
    parser.add_argument('--img_size', default=512)

    # mode name on log record
    parser.add_argument('--model_name', default='ASSNet',
                        choices=['Unet', 'AttUnet', 'res_unet_plus', 'R2Unet', 'R2AttU_Net', 'sepnet', 'KiU_Net',
                                 'Unet3D', 'ResT', 'ASSNet'])
    # pre trained
    parser.add_argument('--pretrained', default=False, type=str2bool)
    # dataset name on log record
    parser.add_argument('--dataset', default="Synapse",
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default='medAug')
    parser.add_argument('--loss', default='BCEDiceLoss')

    # training
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=500, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=10, type=int,
                        metavar='N', help='-')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate, Resunet=1e-4, R2Unet=1e-5, Unet=1e-2, ResUformer=0.001')
    parser.add_argument('--momentum', default=0.98, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-6, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='nesterov')
    parser.add_argument('--deepsupervision', default=False)
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))



def train(args, train_loader, model, criterion, optimizer, lr_decay, epoch, index):
    losses = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    dices_3s = AverageMeter()
    dices_4s = AverageMeter()
    dices_5s = AverageMeter()
    dices_6s = AverageMeter()
    dices_7s = AverageMeter()
    dices_8s = AverageMeter()

    model.train()
    l2_reg = 0.5
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # compute gradient and do optimizing step
        # Before backward, use opt change all variable's loss = 0, b/c gradient will accumulate
        optimizer.zero_grad()
        input = input.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).float()

        # Check for NaNs in inputs
        if torch.isnan(input).any() or torch.isnan(target).any():
            print("Input contains NaN")

        #cal iteration
        # num_iterations = len(train_loader) * args.epochs
        # curr_iter = index * len(train_loader) + i + 1
        # print('Iteration [%d/%d]' % (curr_iter, num_iterations))

        # compute output
        # with torch.cuda.amp.autocast():  # Mixed precision training
        outputs = model(input)
        loss = criterion(outputs, target)
        dice_1 = dice_coef_synapse(outputs, target)[0]
        dice_2 = dice_coef_synapse(outputs, target)[1]
        dice_3 = dice_coef_synapse(outputs, target)[2]
        dice_4 = dice_coef_synapse(outputs, target)[3]
        dice_5 = dice_coef_synapse(outputs, target)[4]
        dice_6 = dice_coef_synapse(outputs, target)[5]
        dice_7 = dice_coef_synapse(outputs, target)[6]
        dice_8 = dice_coef_synapse(outputs, target)[7]


        losses.update(loss.item(), input.size(0))
        # ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))
        dices_2s.update(torch.tensor(dice_2), input.size(0))
        dices_3s.update(torch.tensor(dice_3), input.size(0))
        dices_4s.update(torch.tensor(dice_4), input.size(0))
        dices_5s.update(torch.tensor(dice_5), input.size(0))
        dices_6s.update(torch.tensor(dice_6), input.size(0))
        dices_7s.update(torch.tensor(dice_7), input.size(0))
        dices_8s.update(torch.tensor(dice_8), input.size(0))

        # backward to calculate loss
        loss.backward()
        optimizer.step()
    # update learning rate
    lr_decay.step()

    log = OrderedDict([
        ('lr', optimizer.param_groups[0]['lr']),
        ('loss', losses.avg),
        # ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg),
        ('dice_3', dices_3s.avg),
        ('dice_4', dices_4s.avg),
        ('dice_5', dices_5s.avg),
        ('dice_6', dices_6s.avg),
        ('dice_7', dices_7s.avg),
        ('dice_8', dices_8s.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    dices_3s = AverageMeter()
    dices_4s = AverageMeter()
    dices_5s = AverageMeter()
    dices_6s = AverageMeter()
    dices_7s = AverageMeter()
    dices_8s = AverageMeter()
    hd95_s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            with torch.cuda.amp.autocast():
                l2_reg = 0.1
                # compute output
                outputs = model(input)
                loss = criterion(outputs, target)
                iou = iou_score(outputs, target)
                dice_1 = dice_coef_synapse(outputs, target)[0]
                dice_2 = dice_coef_synapse(outputs, target)[1]
                dice_3 = dice_coef_synapse(outputs, target)[2]
                dice_4 = dice_coef_synapse(outputs, target)[3]
                dice_5 = dice_coef_synapse(outputs, target)[4]
                dice_6 = dice_coef_synapse(outputs, target)[5]
                dice_7 = dice_coef_synapse(outputs, target)[6]
                dice_8 = dice_coef_synapse(outputs, target)[7]
                hd95 = hd95_2d(outputs, target)

                losses.update(loss.item(), input.size(0))
                ious.update(iou, input.size(0))
                dices_1s.update(torch.tensor(dice_1), input.size(0))
                dices_2s.update(torch.tensor(dice_2), input.size(0))
                dices_3s.update(torch.tensor(dice_3), input.size(0))
                dices_4s.update(torch.tensor(dice_4), input.size(0))
                dices_5s.update(torch.tensor(dice_5), input.size(0))
                dices_6s.update(torch.tensor(dice_6), input.size(0))
                dices_7s.update(torch.tensor(dice_7), input.size(0))
                dices_8s.update(torch.tensor(dice_8), input.size(0))
                hd95_s.update(torch.tensor(hd95), input.size(0))

            log = OrderedDict([
                ('loss', losses.avg),
                ('iou', ious.avg),
                ('dice_1', dices_1s.avg),
                ('dice_2', dices_2s.avg),
                ('dice_3', dices_3s.avg),
                ('dice_4', dices_4s.avg),
                ('dice_5', dices_5s.avg),
                ('dice_6', dices_6s.avg),
                ('dice_7', dices_7s.avg),
                ('dice_8', dices_8s.avg),
                ('HD95_avg', hd95_s.avg),
            ])

            return log

def get_gamma(epoch, total_epochs):
    return ((1 - (epoch / total_epochs)) ** 0.9)


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s' % (args.dataset, args.model_name)
        else:
            args.name = '%s_%s' % (args.dataset, args.model_name)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('../trained_models/{}_{}/{}'.format(args.dataset, args.model_name, timestamp)):
        os.makedirs('../trained_models/{}_{}/{}'.format(args.dataset, args.model_name, timestamp))
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')


    with open('../trained_models/{}_{}/{}/args.txt'.format(args.dataset,args.model_name, timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, '../trained_models/{}_{}/{}/args.pkl'.format(args.dataset, args.model_name, timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss_synapse().cuda()

    cudnn.benchmark = True
    if args.model_name == 'Unet3D':
        img_paths = glob('../data/train_image3D/*')
        mask_paths = glob('../data/train_mask3D/*')
    else:
        # Data loading code
        img_paths = glob('../data/trainImage_synapse_png/*')
        mask_paths = glob('../data/trainMask_synapse_png/*')

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=seed_value)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))

    # create model
    print("=> creating model %s" % args.model_name)
    if args.model_name == 'Unet':
        model = Unet.U_Net(args)
    if args.model_name == 'ASSNet':
        model = ASSNet(args)
        pretrained_path = '../trained_models/xxx/xxx.pth'
        print('pretrained selected!')

    #multi gpu
    model = torch.nn.DataParallel(model).cuda()

    '''
    pretrain
    '''
    if args.pretrained == True:
        print('Pretrained model loading...')
        # load_pretrained_weights(model, pretrained_path)
        model.load_state_dict(torch.load(pretrained_path))
        print('Pretrained model loaded!')
    else:
        print('No Pretrained')

    print('{} parameters:{}'.format(args.model_name, count_params(model)))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
        print('AdamW optimizer loaded!')
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        #                              betas=(0.9, 0.999),
        #                              weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        print('SGD optimizer loaded!')



    def make_odd(num):
        num = math.ceil(num)
        if num % 2 == 0:
            num += 1
        return num

    level = 5
    transform_ct = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=min(0.2 * level, 1)),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                 crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
        A.OneOf([
            A.ElasticTransform(alpha=0.1 * level, sigma=0.25 * level, alpha_affine=0.25 * level, p=0.1),
            A.GridDistortion(distort_limit=0.05 * level, p=0.1),
            A.OpticalDistortion(distort_limit=0.05 * level, p=0.1)
        ], p=0.2),
        ToTensorV2()
    ], p=1)
    # transform_ct = transforms.ToTensor()


    train_dataset = Dataset_synapse_png(args, train_img_paths, train_mask_paths, transform=transform_ct)
    val_dataset = Dataset_synapse_png(args, val_img_paths, val_mask_paths, transform=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)


    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'dice_1', 'dice_2',
        'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2', 'HD95_avg'
    ])


    best_train_loss = 100
    best_avg_dice = 0
    val_trigger = False


    first_time = time.time()
    #lr decay
    # scheduler_mult = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98)
    # 使用 CosineAnnealingLR 调度器实现余弦退火学习率衰减策略
    scheduler_mult = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=6e-6)

    for i, epoch in enumerate(range(args.epochs)):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        train_log = train(args, train_loader, model, criterion, optimizer, scheduler_mult, epoch, i)
        print('lr %.8f - train_loss %.4f' % (train_log['lr'], train_log['loss']))
        print('dice_1 %.4f - dice_2 %.4f - dice_3 %.4f - dice_4 %.4f'%(train_log['dice_1'], train_log['dice_2'],
                                                                         train_log['dice_3'], train_log['dice_4']))
        print('dice_5 %.4f - dice_6 %.4f - dice_7 %.4f - dice_8 %.4f'%(train_log['dice_5'],train_log['dice_6'],
                                                                       train_log['dice_7'],train_log['dice_8']))


        train_loss = train_log['loss']

        avg_dice = (train_log['dice_1']+train_log['dice_2']+train_log['dice_3']+train_log['dice_4']
                    +train_log['dice_5']+train_log['dice_6']+train_log['dice_7']+train_log['dice_8'])/8
        if train_loss < 0:
            print('Gradient descent not exist!')
            break
        if (train_loss < best_train_loss) and (avg_dice > best_avg_dice):
            val_trigger = True
            best_train_loss = train_loss
            best_avg_dice = avg_dice

        if val_trigger == True:
            print("=> Start Validation...")
            val_trigger = False
            val_log = validate(args, val_loader, model, criterion)
            print('lr %.8f - val_loss %.4f - val_iou %.4f'%(train_log['lr'], val_log['loss'], val_log['iou']))
            print('val_dice_1 %.4f - val_dice_2 %.4f - val_dice_3 %.4f - val_dice_4 %.4f'%(val_log['dice_1'],
                                                                                           val_log['dice_2'],
                                                                                           val_log['dice_3'],
                                                                                           val_log['dice_4']))
            print('val_dice_5 %.4f - val_dice_6 %.4f - val_dice_7 %.4f - val_dice_8 %.4f - val_HD95_avg %.4f'%(
                val_log['dice_5'], val_log['dice_6'], val_log['dice_7'], val_log['dice_8'], val_log['HD95_avg']
            ))


            tmp = pd.Series([
                epoch,
                train_log['lr'],
                train_log['loss'],
                train_log['dice_1'],
                train_log['dice_2'],
                train_log['dice_3'],
                train_log['dice_4'],
                train_log['dice_5'],
                train_log['dice_6'],
                train_log['dice_7'],
                train_log['dice_8'],
                # train_log['HD95'],
                val_log['loss'],
                val_log['iou'],
                val_log['dice_1'],
                val_log['dice_2'],
                val_log['dice_3'],
                val_log['dice_4'],
                val_log['dice_5'],
                val_log['dice_6'],
                val_log['dice_7'],
                val_log['dice_8'],
                val_log['HD95_avg'],
            ], index=['epoch', 'lr', 'loss', 'dice_1', 'dice_2', 'dice_3',
                      'dice_4', 'dice_5', 'dice_6', 'dice_7', 'dice_8',
                      'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2',
                      'val_dice_3', 'val_dice_4','val_dice_5', 'val_dice_6',
                      'val_dice_7', 'val_dice_8','HD95_avg'])

            # 确保 log 的列顺序与 tmp 的索引顺序一致
            log = log.reindex(columns=tmp.index.tolist())
            # log = log._append(tmp, ignore_index=True)
            # 使用 pd.concat 替代 append
            log = pd.concat([log, tmp.to_frame().T], ignore_index=True)
            log.to_csv('../trained_models/{}_{}/{}/validation_{}_{}_{}_batchsize_{}.csv'.format(args.dataset, args.model_name, timestamp, args.model_name,
                                                                                 args.aug, args.loss, args.batch_size),index=True)
            print('save result to csv ->')
            torch.save(model.state_dict(),
                       '../trained_models/{}_{}/{}/epoch{}-val_loss:{:.4f}_model.pth'.format(
                           args.dataset, args.model_name, timestamp, epoch, val_log['loss'])
                       )
            print("=> saved best model .pth")


            # # early stopping
            # if not args.early_stop is None:
            #     if trigger >= args.early_stop:
            #         print("=> early stopping")
            #         break
        else:
            tmp = pd.Series([
                epoch,
                train_log['lr'],
                train_log['loss'],
                train_log['dice_1'],
                train_log['dice_2'],
                train_log['dice_3'],
                train_log['dice_4'],
                train_log['dice_5'],
                train_log['dice_6'],
                train_log['dice_7'],
                train_log['dice_8'],
                # train_log['HD95'],
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
            ], index=['epoch', 'lr', 'loss', 'dice_1', 'dice_2', 'dice_3',
                      'dice_4', 'dice_5', 'dice_6', 'dice_7', 'dice_8',
                      'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2',
                      'val_dice_3', 'val_dice_4','val_dice_5', 'val_dice_6',
                      'val_dice_7', 'val_dice_8','HD95_avg'])

            # # 确保 log 的列顺序与 tmp 的索引顺序一致
            log = log.reindex(columns=tmp.index.tolist())
            # log = log._append(tmp, ignore_index=True)
            # 使用 pd.concat 替代 append
            log = pd.concat([log, tmp.to_frame().T], ignore_index=True)
            log.to_csv('../trained_models/{}_{}/{}/Train_{}_{}_{}_batchsize_{}.csv'.format(args.dataset, args.model_name, timestamp,
                                                                                 args.model_name, args.aug, args.loss,
                                                                                 args.batch_size), index=True)
            print('save result to csv ->')

        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed_value = 20241017
    # seed_value = 2014
    np.random.seed(seed_value)
    random.seed(seed_value)
    # os.environ['PYTHONHASHSEED'] = str(seed_value)  # ban hash random, let experiment reproduceable
    # set cpu seed
    torch.manual_seed(seed_value)
    # set gpu seed(1 gpu)
    torch.cuda.manual_seed(seed_value)
    # multi gpu
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    main()
