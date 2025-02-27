import numpy as np
import torch
import torch.nn.functional as F
import skimage.morphology as morphology
from scipy.ndimage.morphology import generate_binary_structure
import scipy.spatial as spatial
import math
from scipy.ndimage import distance_transform_edt

def mean_iou(y_true_in, y_pred_in, print_table=False):
    if True: #not np.sum(y_true_in.flatten()) == 0:
        labels = y_true_in
        y_pred = y_pred_in

        true_objects = 2
        pred_objects = 2

        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        if print_table:
            print("Thresh\tTP\tFP\tFN\tPrec.")
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            if (tp + fp + fn) > 0:
                p = tp / (tp + fp + fn)
            else:
                p = 0
            if print_table:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
            prec.append(p)

        if print_table:
            print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
        return np.mean(prec)

    # else:
    #     if np.sum(y_pred_in.flatten()) == 0:
    #         return 1
    #     else:
    #         return 0


def batch_iou(output, target):
    output = torch.sigmoid(output).data.cpu().numpy() > 0.5
    target = (target.data.cpu().numpy() > 0.5).astype('int')
    output = output[:,0,:,:]
    target = target[:,0,:,:]

    ious = []
    for i in range(output.shape[0]):
        ious.append(mean_iou(output[i], target[i]))

    return np.mean(ious)


def mean_iou(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    ious = []
    for t in np.arange(0.5, 1.0, 0.05):
        output_ = output > t
        target_ = target > t
        intersection = (output_ & target_).sum()
        union = (output_ | target_).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)

    return np.mean(ious)

'''
RVD stands for Relative Volume Difference. 
evaluate the similarity or dissimilarity between two binary images or segmentation masks.
RVD ranges from -1 to 1, where a value of 0 indicates perfect overlap between the segmented region and the ground truth region
'''
def rvd_score(output, target):
    smooth = 1e-5
    l2_reg = 0.1  # regularization
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    rvd = (intersection - union) / (union + smooth)

    return rvd
'''
Jaccard
Volume Overlap
'''
# def iou_score(output, target):
#     smooth = 1e-5
#     l2_reg = 0.1 # regularization
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)

def iou_score(output, target):
    #gpu
    smooth = 1e-5  # 防止分母为 0

    # 使用 sigmoid 激活函数，并保持在 GPU 上
    output = torch.sigmoid(output)

    # 阈值化操作，保持在 GPU 上
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()

    # 计算交集和并集，保持在 GPU 上
    intersection = (output_ * target_).sum(dim=(1, 2, 3))  # 按批次求和
    union = (output_ + target_ - output_ * target_).sum(dim=(1, 2, 3))  # 并集公式

    # 计算 IoU
    iou = (intersection + smooth) / (union + smooth)

    # 返回每个 batch 的平均 IoU
    return iou.mean().item()

def f1_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    true_positive = (output_ & target_).sum()
    false_positive = (output_ & (~target_)).sum()
    false_negative = ((~output_) & target_).sum()

    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)

    return 2 * precision * recall / (precision + recall + smooth)

'''
FNR (False Negative Rate):
FNR measures the proportion of positive instances that are incorrectly classified as negative.
(FN / (FN + TP))
'''
def fnr_score(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    fnr = (target_ & ~output_).sum() / (target_.sum() + 1e-5)

    return fnr

'''
FPR (False Positive Rate):
FPR measures the proportion of actual negative instances that are incorrectly classified as positive.
 (FP / (FP + TN))
'''
def fpr_score(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    fpr = (~target_ & output_).sum() / (~target_).sum()

    return fpr

'''
ASSD (Average Symmetric Surface Distance):
ASSD measures the average distance
the average of the distances from each point on one surface to the nearest point on the other surface and vice versa.
common performance metrics in medical image segmentation tasks
'''
def assd_score(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dist_map_output = distance_transform_edt(1 - output)
    dist_map_target = distance_transform_edt(1 - target)

    assd_val = np.abs(dist_map_output - dist_map_target).mean()

    return assd_val

'''
RMSD (Root Mean Square Distance):
RMSD measures the root mean square difference
It provides an overall measure of the difference between the two surfaces.
common performance metrics in medical image segmentation tasks
'''
def rmsd_score(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dist_map_output = distance_transform_edt(1 - output)
    dist_map_target = distance_transform_edt(1 - target)

    rmsd_val = np.sqrt(np.mean(np.square(dist_map_output - dist_map_target)))

    return rmsd_val
'''
HD95(Hausdorff Distance 95th percentile): 
Calculates the furthest distance from the true segmentation 
among 95% of the segmented pixels.
'''
def hd95_lits(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    # Calculate distance maps
    distance_output = distance_transform_edt(1 - output_)
    distance_target = distance_transform_edt(1 - target_)

    # Calculate HD95
    #check if array are empty or nan
    if np.any(target_):
        hd95_1 = np.percentile(distance_output[target_], 95)
    else:
        hd95_1 = np.mean(distance_output)
    if np.any(output_):
        hd95_2 = np.percentile(distance_target[output_], 95)
    else:
        hd95_2 = np.mean(distance_target)

    return hd95_1, hd95_2

def hd95_2d(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    distance_output = distance_transform_edt(1-output_)
    distance_target = distance_transform_edt(1-target_)

    distances = np.concatenate([distance_output[target_], distance_target[output_]])
    hd95 = np.percentile(distances, 95)

    return hd95



'''
MSD (Mean Surface Distance):
MSD measures the average distance between corresponding points on the predicted and ground truth surfaces.
common performance metrics in medical image segmentation tasks
'''
def msd_score(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    dist_map_output = distance_transform_edt(1 - output)
    dist_map_target = distance_transform_edt(1 - target)

    msd_val = np.mean(np.square(dist_map_output - dist_map_target))

    return msd_val

# def iou_score(output, target):
#     smooth = 1e-5
#     l2_reg = 0.1 # regularization
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     intersection = (output_ & target_).sum()
#     union = (output_ | target_).sum()
#
#     return (intersection + smooth) / (union + smooth)
def dice_coef_synapse_cpu(output, target):
    smooth = 1e-5  # 防止 union 为 0
    l2_reg = 0.9989

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    dice_scores = []

    for i in range(1, 9):  # 从 1 到 8，忽略背景0
        output_i = output_[:, i, :, :]
        target_i = target_[:, i, :, :]

        intersection = (output_i * target_i).sum()
        union = output_i.sum() + target_i.sum()

        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_i)

    return dice_scores

def dice_coef_synapse(output, target):
    smooth = 1e-5  # 防止 union 为 0

    # 确保输入是张量，并在 GPU 上进行计算
    if torch.is_tensor(output):
        output = torch.sigmoid(output)  # 使用 sigmoid 激活函数
    if torch.is_tensor(target):
        target = target.float()  # 确保 target 是浮点型

    # 阈值化操作
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()

    dice_scores = []

    for i in range(1, 9):  # 从 1 到 8，忽略背景 0
        output_i = output_[:, i, :, :]
        target_i = target_[:, i, :, :]

        # 计算交集和并集
        intersection = (output_i * target_i).sum(dim=(1, 2))  # 按批次求和
        union = output_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))

        # 计算 Dice 系数
        dice_i = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice_i.mean().item())  # 返回每个通道的平均 Dice 系数

    return dice_scores


def dice_coef_lits(output, target):
    smooth = 1e-5  # 防止分母为 0
    l2_reg = 0.9989

    # 使用 sigmoid 激活函数，并保持在 GPU 上
    output = torch.sigmoid(output)

    # 阈值化操作，保持在 GPU 上
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()

    # 分别取出 liver 和 tumor 的通道
    output_1 = output_[:, 1, :, :]  # liver
    output_2 = output_[:, 2, :, :]  # tumor
    target_1 = target_[:, 1, :, :]
    target_2 = target_[:, 2, :, :]

    # 计算交集和并集，保持在 GPU 上
    intersection_1 = (output_1 * target_1).sum(dim=(1, 2))  # 按批次求和
    intersection_2 = (output_2 * target_2).sum(dim=(1, 2))
    union1 = output_1.sum(dim=(1, 2)) + target_1.sum(dim=(1, 2))
    union2 = output_2.sum(dim=(1, 2)) + target_2.sum(dim=(1, 2))

    # 计算 Dice 系数
    dice_1 = (2. * intersection_1 + smooth) / (union1 + smooth)
    dice_2 = (2. * intersection_2 + smooth) / (union2 + smooth)

    # 返回每个 batch 的平均 Dice 系数
    return dice_1.mean().item(), dice_2.mean().item()

# def dice_coef_lits(output, target):
#     smooth = 1e-5 # in case union is 0
#     l2_reg = 0.9989
#     num = output.shape[0]
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#     # print(output_.shape, target_.shape)
#     # exit()
#     output_1 = output_[:,1,:,:] #liver
#     output_2 = output_[:,2,:,:] #tumor
#
#     target_1 = target_[:,1,:,:]
#     target_2 = target_[:,2,:,:]
#
#     intersection_1 = (output_1 * target_1)
#     intersection_2 = (output_2 * target_2)
#
#     union1 = output_1.sum() + target_1.sum()
#     union2 = output_2.sum() + target_2.sum()
#
#     dice_1 = (2. * intersection_1.sum() + smooth) / (union1 + smooth)
#     dice_2 = (2. * intersection_2.sum() + smooth) / (union2 + smooth)
#
#     return dice_1, dice_2

def dice_coef(output, target):
    smooth = 1e-5 # in case union is 0
    l2_reg = 0.9989
    num = output.shape[0]
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    # print(output_.shape, target_.shape)
    # exit()
    output_1 = output_[:,0,:,:]
    output_2 = output_[:,1,:,:]

    target_1 = target_[:,0,:,:]
    target_2 = target_[:,1,:,:]

    intersection_1 = (output_1 * target_1)
    intersection_2 = (output_2 * target_2)

    union1 = output_1.sum() + target_1.sum()
    union2 = output_2.sum() + target_2.sum()

    dice_1 = l2_reg*((2. * intersection_1.sum() + smooth) / (union1 + smooth))
    dice_2 = (2. * intersection_2.sum() + smooth) / (union2 + smooth)

    return dice_1, dice_2


def accuracy(output, target):
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    output = torch.sigmoid(output).reshape(-1).data.cpu().numpy()
    output = (np.round(output)).astype('int')
    # target = target.view(-1).data.cpu().numpy()
    target = target.reshape(-1).data.cpu().numpy()
    target = (np.round(target)).astype('int')
    # print(output.shape)
    # print(target.shape)
    # exit()
    (output == target).sum()

    return (output == target).sum() / len(output)

'''
PPV（Positive Predictive Value） also called Precision
PPV = (intersection + smooth) / (output.sum() + smooth)
common performance metrics in medical image segmentation tasks
'''
def ppv(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    intersection = (output * target).sum()
    return (intersection + smooth) / (output.sum() + smooth)
'''
Sensitivity also called Recall
Sensitivity = (intersection + smooth) / (target.sum() + smooth)
common performance metrics in medical image segmentation tasks
'''
def sensitivity(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (intersection + smooth) / (target.sum() + smooth)


"""
计算基于重叠度和距离等九种分割常见评价指标
"""


class Metirc():

    def __init__(self, real_mask, pred_mask, voxel_spacing):
        """

        :param real_mask: 金标准
        :param pred_mask: 预测结果
        :param voxel_spacing: 体数据的spacing
        """
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    # 下面三个是提取边界和计算最小距离的实用函数
    def get_surface(self, mask, voxel_spacing):
        """

        :param mask: ndarray
        :param voxel_spacing: 体数据的spacing
        :return: 提取array的表面点的真实坐标(以mm为单位)
        """

        # 卷积核采用的是三维18邻域

        kernel = generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask
        # np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
        surface_pts = surface.nonzero()
        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))

        if surface_pts.size == 0:
            surface_pts = [[0, 0, 0]]

        # print(surface_pts) # todo: 这里当遇到没有标签值的空数组时会报错
        # print(np.array(self.voxel_sapcing[::-1]).reshape(1, 3))

        # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
        # # 读出来的数据spacing和shape不是对应的,所以需要反向
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):
        """

        :return: 预测结果表面体素到金标准表面体素的最小距离
        """

        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):
        """

        :return: 金标准表面体素到预测结果表面体素的最小距离
        """
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    # 下面的六个指标是基于重叠度的
    def get_dice_coefficient(self):
        """

        :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
        """

        smooth = 1e-5
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return dice, 2 * intersection, union

    def get_jaccard_index(self):
        """

        :return: 杰卡德系数
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_VOE(self):
        """

        :return: 体素重叠误差 Volumetric Overlap Error
        """

        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """

        :return: 体素相对误差 Relative Volume Difference
        """

        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        """

        :return: 欠分割率 False negative rate
        """
        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):
        """

        :return: 过分割率 False positive rate
        """
        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union

    # 下面的三个指标是基于距离的
    def get_ASSD(self):
        """

        :return: 对称位置平均表面距离 Average Symmetric Surface Distance
        """
        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
            (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        """

        :return: 对称位置表面距离的均方根 Root Mean Square symmetric Surface Distance
        """
        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):
        """

        :return: 对称位置的最大表面距离 Maximum Symmetric Surface Distance
        """
        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
