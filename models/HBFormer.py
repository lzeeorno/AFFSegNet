# ------------------------------------------------------------
# Copyright (c) University of Macau and
# Shenzhen Institutes of Advanced Technology，Chinese Academy of Sciences.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by FuChen Zheng(orcid:https://orcid.org/0009-0001-8589-7026)
# ------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import numpy as np
from functools import reduce, lru_cache
from einops import rearrange
from operator import mul
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ptflops import get_model_complexity_info
# from dataset.dataset import synapse_num_classes, lits_num_classes
import argparse
import math
import cv2

try:
    # 作为包导入时
    from .swin_backbone import SwinBackbone
except ImportError:
    # 直接运行 python models/HBFormer.py 时
    from swin_backbone import SwinBackbone



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--HBFormer', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='Synapse',
                        help='dataset name (default: LiTS2017/Synapse)')

    args = parser.parse_args()

    return args

# ===== Plugin Blocks - 直接集成到文件中 =====

class ECALayer(nn.Module):
    """Efficient Channel Attention"""
    
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class Med_DSPP(nn.Module):
    """动态多尺度特征融合：基于ODConv思想的自适应分支权重 + 深度可分离空洞卷积
    
    参考：
    - ODConv (CVPR 2022): Omni-Dimensional Dynamic Convolution
    - CondConv: Dynamic routing for efficient networks
    
    核心创新：
    1. 每个空洞分支的重要性根据输入内容自适应调整
    2. 卷积核参数可学习线性组合，提升表达灵活性
    3. 显著减少冗余计算，同时保持多尺度感受野
    """
    
    def __init__(self, in_dims, out_dims, rates=(1, 4, 8), num_experts=4):
        super().__init__()
        self.rates = rates
        self.num_experts = num_experts
        self.num_branches = len(rates)
        
        # 分支权重预测网络
        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dims, in_dims // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims // 8, self.num_branches, 1),
            nn.Softmax(dim=1)
        )
        
        # 多专家卷积核权重预测
        self.expert_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dims, in_dims // 8, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dims // 8, self.num_experts, 1),
            nn.Softmax(dim=1)
        )
        
        # 每个分支包含多个专家卷积核
        self.dw_experts = nn.ModuleList()
        self.pw_experts = nn.ModuleList()
        
        for r in rates:
            # 深度可分离专家卷积 - 多个depthwise核
            dw_expert = nn.ModuleList([
                nn.Conv2d(in_dims, in_dims, 3, stride=1, padding=r, dilation=r, 
                         groups=in_dims, bias=False) 
                for _ in range(num_experts)
            ])
            
            # pointwise专家卷积
            pw_expert = nn.ModuleList([
                nn.Conv2d(in_dims, out_dims, 1, bias=False)
                for _ in range(num_experts)  
            ])
            
            self.dw_experts.append(dw_expert)
            self.pw_experts.append(pw_expert)
        
        # 批归一化和激活
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_dims) for _ in rates])
        self.activations = nn.ModuleList([nn.SiLU(inplace=True) for _ in rates])
        
        # ECA注意力用于通道精炼
        self.eca = ECALayer(out_dims * self.num_branches)
        
        # 最终融合
        self.fuse = nn.Conv2d(out_dims * self.num_branches, out_dims, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_dims)

    def dynamic_conv(self, x, dw_experts, pw_experts, expert_weights):
        """动态卷积：根据专家权重组合多个卷积核"""
        B, C, H, W = x.shape
        
        # 深度可分离卷积 - 动态组合专家核
        dw_out = 0
        for i, dw_conv in enumerate(dw_experts):
            dw_out += expert_weights[:, i:i+1, :, :] * dw_conv(x)
            
        # pointwise卷积 - 动态组合专家核  
        pw_out = 0
        for i, pw_conv in enumerate(pw_experts):
            pw_out += expert_weights[:, i:i+1, :, :] * pw_conv(dw_out)
            
        return pw_out

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 预测分支权重和专家权重
        branch_weights = self.branch_attention(x)  # B, num_branches, 1, 1
        expert_weights = self.expert_attention(x)   # B, num_experts, 1, 1
        
        branch_feats = []
        
        # 每个空洞分支的动态卷积
        for i, (dw_experts, pw_experts, bn, act) in enumerate(
            zip(self.dw_experts, self.pw_experts, self.bns, self.activations)
        ):
            # 动态卷积
            feat = self.dynamic_conv(x, dw_experts, pw_experts, expert_weights)
            
            # 批归一化和激活
            feat = act(bn(feat))
            
            # 应用分支权重
            weighted_feat = feat * branch_weights[:, i:i+1, :, :]
            branch_feats.append(weighted_feat)
        
        # 拼接所有分支特征
        concat_feat = torch.cat(branch_feats, dim=1)
        
        # ECA通道注意力
        refined_feat = self.eca(concat_feat)
        
        # 最终融合
        output = self.fuse_bn(self.fuse(refined_feat))
        
        return output

class EFFN(nn.Module):
    """ Multilayer perceptron(MLP) with depthwise separable convolution."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2d(in_channels=hidden_features,
                                        out_channels=hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.pointwise_conv = nn.Conv2d(in_channels=hidden_features,
                                        out_channels=hidden_features, kernel_size=1)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# class SemanticsAligner(nn.Module):
#     def __init__(self, window_size, threshold=3):
#         super(SemanticsAligner, self).__init__()
#         self.window_size = window_size
#         self.threshold = threshold
#         self.sine_pos_embed = PositionEmbeddingSine()
#
#     def calculate_distances(self, coords):
#         relative_coords = coords[:, :, None] - coords[:, None, :]
#         # 确保 relative_coords 是浮点类型
#         relative_coords = relative_coords.float()
#         distances = torch.norm(relative_coords, dim=-1)
#         return distances
#
#     def count_same_distances(self, distances):
#         unique_distances, counts = torch.unique(distances, return_counts=True)
#         return counts
#
#     def forward(self, x):
#         B, H, W, C = x.shape
#         coords_h = torch.arange(H)
#         coords_w = torch.arange(W)
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
#         coords_flatten = torch.flatten(coords, 1).T  # H*W, 2
#
#         distances = self.calculate_distances(coords_flatten)
#         counts = self.count_same_distances(distances)
#
#         if torch.any(counts > self.threshold):
#             pos_embed = self.sine_pos_embed(x)
#         else:
#             pos_embed = None  # Use relative position encoding
#
#         return pos_embed


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class MWA_Block(nn.Module):
    """ MWA_Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = EFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # Initialize the ContextAware_Geometry_Aligner
        # self.CG_Aligner = ContextAware_Geometry_Aligner(dim)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # Apply Aligner
        # Aligner = self.CG_Aligner(x)
        # Combine the original features with the CG Aligner features
        # x = x + Aligner
        x = x.view(B, H * W, C)

        # EFFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list
        mask = torch.zeros_like(x[:, 0, :, :]).bool()
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



class Encoder(nn.Module):
    """ A basic encoder base on MWA_Block for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # Initialize the ContextAware_Geometry_Aligner
        # self.CG_Aligner = ContextAware_Geometry_Aligner(dim*2)

        # build blocks
        self.blocks = nn.ModuleList([
            MWA_Block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        # pos = self.position_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        # pos = pos.flatten(2).transpose(1, 2)

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            x = x.view(B, -1, H // 2, W // 2)
            return x
            # Apply Aligner
            # Aligner = self.CG_Aligner(x)
            # Combine the original features with the CG Aligner features
            # x = x + Aligner
            # return x
        else:
            x = x.view(B, -1, H, W)
            return x
            # Apply Aligner
            # Aligner = self.CG_Aligner(x)
            # Combine the original features with the CG Aligner features
            # x = x + Aligner
            # return x

# class ContextAware_Geometry_Aligner(nn.Module):
#     """Contextual Position Embedding with Geometric Features.
#
#     This module combines traditional position encoding with contextual information
#     and geometric features extracted from the image.
#     """
#
#     def __init__(self, embed_dim, kernel_size=3):
#         super(ContextAware_Geometry_Aligner, self).__init__()
#         self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2, groups=embed_dim)
#         self.proj = nn.Linear(2, embed_dim)
#
#         # Scharr kernels for edge detection
#         self.scharr_x = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
#         self.scharr_y = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False)
#
#         # Define the Scharr operator kernels
#         scharr_kernel_x = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32)
#         scharr_kernel_y = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32)
#
#         # Initialize the Scharr operator weights
#         self.scharr_x.weight.data = scharr_kernel_x.repeat(embed_dim, 1, 1, 1)
#         self.scharr_y.weight.data = scharr_kernel_y.repeat(embed_dim, 1, 1, 1)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#
#         # Generate traditional position encoding
#         coords_h = torch.arange(H, device=x.device)
#         coords_w = torch.arange(W, device=x.device)
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
#         coords = coords.flatten(1).T  # H*W, 2
#         pos_embed = self.proj(coords.float()).view(H, W, C).permute(2, 0, 1)  # C, H, W
#
#         # Extract contextual information
#         context_embed = self.conv(x)
#
#         # Extract geometric features using Scharr operator
#         grad_x = self.scharr_x(x)
#         grad_y = self.scharr_y(x)
#         geo_features = torch.sqrt(grad_x ** 2 + grad_y ** 2)
#
#         # Combine position encoding with contextual information and geometric features
#         combined_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1) + context_embed + geo_features
#
#         return combined_embed




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None


    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww

        # Apply Aligner
        # Aligner = self.CG_Aligner(x)

        # Combine the original features with the CG Aligner features
        # x = x + Aligner

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class ResConvBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm=None):
        super(ResConvBlock, self).__init__()
        if norm is None:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn1 = nn.InstanceNorm2d(planes)
            self.bn2 = nn.InstanceNorm2d(planes)
        else:
            raise KeyError(" the norm is not batch norm and instance norm!!")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, bias=False)
        self.conv_skip = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.stride = stride

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.conv_skip(res)

        out += identity
        out = self.relu(out)

        return out





class ContextAware_Geometry_Aligner(nn.Module):
    def __init__(self, in_planes):
        super(ContextAware_Geometry_Aligner, self).__init__()
        self.in_planes = in_planes

        # Sobel滤波器用于边缘检测
        self.sobel_x = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False, groups=in_planes)
        self.sobel_y = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False, groups=in_planes)

        # 初始化Sobel滤波器参数
        sobel_kernel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.sobel_x.weight.data = sobel_kernel_x.expand(in_planes, 1, -1, -1)
        self.sobel_y.weight.data = sobel_kernel_y.expand(in_planes, 1, -1, -1)

        # 自适应池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 8, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)
        '''1. 边缘检测：使用 Sobel 滤波器进行边缘检测，提取图像的边缘信息。'''
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_features = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        '''2. 特征融合：将边缘特征与原始特征进行融合，以增强边缘信息。'''
        fused_features = x + edge_features

        '''3. 自适应池化和全连接层：使用自适应池化层和全连接层生成特征增强因子。'''
        b, c, _, _ = fused_features.size()
        y = self.avg_pool(fused_features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        '''4. 特征增强：使用生成的增强因子对原始特征进行通道级的增强。'''
        enhanced_features = x * y.expand_as(x)
        # Permute output back to [batch, height, width, channel]
        # enhanced_features = enhanced_features.permute(0, 2, 3, 1)

        return enhanced_features


class LRD_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm=None):
        super(LRD_Block, self).__init__()
        if norm is None:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn1 = nn.InstanceNorm2d(planes)
            self.bn2 = nn.InstanceNorm2d(planes)
        else:
            raise KeyError(" the norm is not batch norm and instance norm!!")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, bias=False)
        self.conv_skip = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.stride = stride

    def forward(self, x):
        res = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.conv_skip(res)

        out += identity
        out = self.relu(out)

        return out

class MFF_Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, upsample_kernel_size, norm_name):
        super(MFF_Decoder, self).__init__()
        self.transp_conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=upsample_kernel_size, stride=upsample_kernel_size)
        self.in_planes = in_planes
        self.out_planes = out_planes
        #LongRangeDependencies(LRD)
        self.LRD_block = LRD_Block(in_planes, out_planes, stride=1, norm=norm_name)
        self.Med_DSPP = Med_DSPP(out_planes, out_planes)
        # self.featurefusion1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.BN = nn.BatchNorm2d(out_planes)
        

    def forward(self, inp, skip):
        input = self.transp_conv(inp)
        WindowAtt_Encoder_skip = torch.cat((input, skip), dim=1)
        x = self.LRD_block(WindowAtt_Encoder_skip)
        output = self.Med_DSPP(x)
        output = self.BN(self.ReLU(output))
        return output




class HBFormer(nn.Module):
    def __init__(
        self,
        args,
        img_size=(512, 512),
        feature_size= 48,
        num_heads=[4, 8, 16, 32],
        norm_name='instance',
        dropout_rate = 0.0,
    ):
        super(HBFormer, self).__init__()
        self.args = args
        # 根据数据集自动推断输入和输出通道数
        in_channels = 3  # 改为3通道，支持伪HDR输入
        if args.dataset == 'LiTS2017':
            out_channels = 3  # 肝脏分割：背景、肝脏、肿瘤
        elif args.dataset == 'Synapse':
            out_channels = 9  # Synapse数据集
        elif args.dataset == 'ACDC':
            out_channels = 3  # 心脏分割：右心室、心肌、左心室（不包括背景）
        else:
            out_channels = 9  # 默认值

        self.patch_size = 2
        self.classification = False
        self.patch_norm = False

        # 原始输入预处理保持不变
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        # 直接使用 SwinBackbone 作为编码器；优先用7，否则选择能整除特征图的窗口
        swin_pretrained = getattr(args, 'swin_pretrained_path', 'pre_trained_weights/swin_tiny_patch4_window7_224.pth')
        grid_size = img_size[0] // 4
        for cand in [7, 8, 6, 5, 4]:
            if grid_size % cand == 0:
                swin_window = cand
                break
        self.MWAencoder = SwinBackbone(pretrained_path=swin_pretrained, img_size=img_size[0], window_size=swin_window)

        # 将 Swin 各 stage 输出映射到原来解码器期望的通道数
        # SwinTiny stages: [96, 192, 384, 768]
        self.proj_enc1 = nn.Conv2d(96,  feature_size * 1, kernel_size=1)
        self.proj_enc2 = nn.Conv2d(192, feature_size * 2, kernel_size=1)
        self.proj_enc3 = nn.Conv2d(384, feature_size * 4, kernel_size=1)
        self.proj_enc4 = nn.Conv2d(768, feature_size * 8, kernel_size=1)

        self.bottleneck = nn.Sequential(
            
        )

        '''MFF Decoder'''
        self.MFF_decoder4 = MFF_Decoder(feature_size * 16, feature_size * 8, 2, None)
        self.MFF_decoder3 = MFF_Decoder(feature_size * 8, feature_size * 4, 2, None)
        self.MFF_decoder2 = MFF_Decoder(feature_size * 4, feature_size * 2, 2, None)
        self.MFF_decoder1 = MFF_Decoder(feature_size * 2, feature_size * 1, 2, None)
        # self.AFF_decoder2 = DecodeBlock(feature_size * 1, feature_size * 1, 2, norm_name)
        self.out = nn.Sequential(
        nn.ConvTranspose2d(feature_size * 1, feature_size * 1, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Conv2d(feature_size * 1, out_channels, 1)
        )

    def forward(self, x_in):
        x_in = self.input_proj(x_in)
        # Swin 编码器：输出4个尺度特征
        feats = self.MWAencoder(x_in)
        c1, c2, c3, c4 = feats

        # 投影到原HBFormer解码器期望的通道维度
        x       = self.proj_enc1(c1)
        enc1    = self.proj_enc2(c2)
        enc2    = self.proj_enc3(c3)  # skip for dec3 (192 channels, 1/16 res)
        enc3_sk = c3                  # skip for dec4 (384 channels, 1/16 res)
        enc4    = c4                  # deepest feature (768 channels, 1/32 res)

        '''MFF Decoder'''
        dec4 = self.MFF_decoder4(enc4, enc3_sk)
        dec3 = self.MFF_decoder3(dec4, F.interpolate(enc2, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = self.MFF_decoder2(dec3, F.interpolate(enc1, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = self.MFF_decoder1(dec2, F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

        out = self.out(dec1)

        return out


'''
检查模型是否能够创建并输出期望的维度
      - Flops:  80.32 GMac
      - Params: 30.05 M
'''
# if __name__ == "__main__":
#     args = parse_args()
#     model = HBFormer(args, img_size=(512, 512), feature_size=48)
#     model.cuda()
#     flops, params = get_model_complexity_info(
#         model, input_res=(3, 512, 512), as_strings=True,
#         print_per_layer_stat=False
#     )
#     print('      - Flops:  ' + flops)
#     print('      - Params: ' + params)
#     x = torch.randn(1, 3, 512, 512).cuda()
#     with torch.no_grad():  # 在不计算梯度的情况下执行前向传播
#         out = model(x)
#     print('Final Output:')
#     print(out.shape)  # 输出预期是与分类头的输出通道数匹配的特征图
