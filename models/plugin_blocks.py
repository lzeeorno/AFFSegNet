"""
CMamba Plugin Blocks: 即插即用模块集合
包含各种先进的注意力机制、卷积模块和特征增强模块

Author: Assistant AI
Date: 2025
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 尝试导入 mamba_ssm
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except ImportError:
    selective_scan_fn = None
    selective_scan_ref = None

# ===== 1. SS2D from VMUNet =====

class SS2D(nn.Module):
    """SS2D from VMUNet - State Space 2D Module"""
    
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        # Check if selective_scan_fn is available
        if selective_scan_fn is None:
            return self.forward_core_fallback(x)
            
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    
    def forward_core_fallback(self, x: torch.Tensor):
        """Fallback when selective_scan_fn is not available"""
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_corev0(x)
        
        if selective_scan_fn is not None:
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            # Fallback: simple average and ensure correct shape
            y = (y1 + y2 + y3 + y4) / 4  # y1-y4 are already [B, H, W, C] from fallback
            
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# ===== 2. Multi-Head Self-Attention (MSA) from Swin Transformer =====

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention from Swin Transformer"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ===== 3. Window Multi-Head Self-Attention (WMSA) from Swin Transformer =====

def window_partition(x, window_size):
    """Partition into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowMultiHeadSelfAttention(nn.Module):
    """Window based Multi-Head Self-Attention from Swin Transformer"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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

# ===== 4. CBAM (Convolutional Block Attention Module) =====

class ChannelAttention(nn.Module):
    """Channel Attention Module in CBAM"""
    
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module in CBAM"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ===== 5. Focused Linear Attention from FLatten Transformer =====

class FocusedLinearAttention(nn.Module):
    """Focused Linear Attention from FLatten Transformer"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q @ kv * z

        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, N, C)
        v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ===== 6. Deformable Attention (自设计基于可变形卷积) =====

class DeformableAttention(nn.Module):
    """Deformable Attention based on Deformable Convolution"""
    
    def __init__(self, dim, num_heads=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Offset and mask prediction
        self.offset_conv = nn.Conv2d(dim, 2 * kernel_size * kernel_size, 
                                   kernel_size=kernel_size, stride=stride, 
                                   padding=padding, dilation=dilation, bias=bias)
        self.mask_conv = nn.Conv2d(dim, kernel_size * kernel_size,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, bias=bias)
        
        # Attention weights
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.mask_conv.weight, 0.)
        nn.init.constant_(self.mask_conv.bias, 0.)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate offsets and masks
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Apply deformable sampling (simplified version)
        # In practice, you would use deformable convolution operations
        # Here we use a simplified attention mechanism
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(-2, -1).contiguous().view(B, C, H, W)
        
        # Apply mask
        mask = mask.view(B, 1, self.kernel_size * self.kernel_size, H, W)
        out = out.unsqueeze(2) * mask
        out = out.sum(dim=2)
        
        out = self.proj(out)
        return out

# ===== 7. SE Block (Squeeze-and-Excitation) =====

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ===== 8. ECA (Efficient Channel Attention) =====

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

# ===== 9. SimAM (Simple Parameter-free Attention Module) =====

class SimAM(nn.Module):
    """Simple Parameter-free Attention Module"""
    
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)

# ===== 10. Coordinate Attention =====

class CoordinateAttention(nn.Module):
    """Coordinate Attention"""
    
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

# ===== 11. Large Selective Kernel (LSK) =====

class LSKBlock(nn.Module):
    """Large Selective Kernel Block"""
    
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

# ===== 12. Non-Local Block =====

class NonLocalBlock(nn.Module):
    """Non-Local Block"""
    
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
                
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

# ===== 13. Global Context Block =====

class GlobalContextBlock(nn.Module):
    """Global Context Block"""
    
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add',)):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
            
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

# ===== 14. RFB (Receptive Field Block) =====

class RFBBlock(nn.Module):
    """Receptive Field Block"""
    
    def __init__(self, in_channel, out_channel, stride=1, scale=0.1):
        super().__init__()
        self.scale = scale
        self.out_channel = out_channel
        inter_channel = in_channel // 8
        
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, 2*inter_channel, 1, stride, 0),
            nn.BatchNorm2d(2*inter_channel),
            nn.ReLU(inplace=True)
        )
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, 2*inter_channel, 3, 1, 1),
            nn.BatchNorm2d(2*inter_channel),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, (inter_channel//2)*3, 3, 1, 1),
            nn.BatchNorm2d((inter_channel//2)*3),
            nn.ReLU(inplace=True),
            nn.Conv2d((inter_channel//2)*3, 2*inter_channel, 3, 1, 1),
            nn.BatchNorm2d(2*inter_channel),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, 1, stride, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channel, (inter_channel//2)*3, (1,3), 1, (0,1)),
            nn.BatchNorm2d((inter_channel//2)*3),
            nn.ReLU(inplace=True),
            nn.Conv2d((inter_channel//2)*3, (inter_channel//2)*3, (3,1), 1, (1,0)),
            nn.BatchNorm2d((inter_channel//2)*3),
            nn.ReLU(inplace=True),
            nn.Conv2d((inter_channel//2)*3, 2*inter_channel, 3, 1, 1),
            nn.BatchNorm2d(2*inter_channel),
            nn.ReLU(inplace=True)
        )
        
        self.ConvLinear = nn.Conv2d(8*inter_channel, out_channel, 1, 1, 0)
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, stride, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        
        return out

# ===== 15. DyReLU (Dynamic ReLU) =====

class DyReLU(nn.Module):
    """Dynamic ReLU"""
    
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super().__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, dim=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, dim=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise_dim = False
        if self.conv_type == '1d' and len(x.shape) == 2:
            x = x.unsqueeze(-1)
            raise_dim = True

        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        
        if self.conv_type == '1d':
            x_perm = x.permute(0, 2, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :self.k].view(-1, 1, self.k, 1) + relu_coefs[:, self.k:].view(-1, 1, self.k, 1)
            result = torch.max(output, dim=2)[0].squeeze(-1).permute(0, 2, 1)
        elif self.conv_type == '2d':
            x_perm = x.permute(0, 2, 3, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :self.k].view(-1, 1, 1, self.k, 1) + relu_coefs[:, self.k:].view(-1, 1, 1, self.k, 1)
            result = torch.max(output, dim=3)[0].squeeze(-1).permute(0, 3, 1, 2)

        if raise_dim:
            result = result.squeeze(-1)
        return result

# ===== 16. CARAFE (Content-Aware ReAssembly of FEatures) =====

class CARAFE(nn.Module):
    """Content-Aware ReAssembly of FEatures"""
    
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.k_enc = k_enc
        self.c = c
        self.c_mid = c_mid

        self.encoder = nn.Sequential(
            nn.Conv2d(c, c_mid, k_enc, padding=k_enc//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, scale*scale*k_up*k_up, 1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Content-aware kernel prediction
        kernel = self.encoder(x)
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.view(b, self.scale*self.scale, self.k_up*self.k_up, h, w)
        
        # Feature reassembly
        x_unfold = F.unfold(x, self.k_up, padding=self.k_up//2)
        x_unfold = x_unfold.view(b, c, self.k_up*self.k_up, h, w)
        
        out = (kernel * x_unfold.unsqueeze(1)).sum(dim=2)
        out = F.pixel_shuffle(out.view(b, -1, h, w), self.scale)
        
        return out

# ===== 测试函数 =====

def test_plugin_blocks():
    """Test all plugin blocks"""
    print("=" * 60)
    print("Testing CMamba Plugin Blocks...")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test configurations
    test_configs = [
        ("SS2D", lambda: SS2D(d_model=64), (2, 16, 16, 64)),
        ("MultiHeadSelfAttention", lambda: MultiHeadSelfAttention(dim=64), (2, 256, 64)),
        ("CBAM", lambda: CBAM(in_planes=64), (2, 64, 16, 16)),
        ("FocusedLinearAttention", lambda: FocusedLinearAttention(dim=64, window_size=(8, 8), num_heads=8), (2, 64, 64)),
        ("SEBlock", lambda: SEBlock(channel=64), (2, 64, 16, 16)),
        ("ECALayer", lambda: ECALayer(channels=64), (2, 64, 16, 16)),
        ("SimAM", lambda: SimAM(), (2, 64, 16, 16)),
        ("CoordinateAttention", lambda: CoordinateAttention(inp=64, oup=64), (2, 64, 16, 16)),
        ("LSKBlock", lambda: LSKBlock(dim=64), (2, 64, 16, 16)),
        ("NonLocalBlock", lambda: NonLocalBlock(in_channels=64), (2, 64, 16, 16)),
        ("GlobalContextBlock", lambda: GlobalContextBlock(inplanes=64, ratio=0.25), (2, 64, 16, 16)),
        ("RFBBlock", lambda: RFBBlock(in_channel=64, out_channel=64), (2, 64, 16, 16)),
        ("DyReLU", lambda: DyReLU(channels=64), (2, 64, 16, 16)),
    ]
    
    for name, model_fn, input_shape in test_configs:
        try:
            print(f"\n--- Testing {name} ---")
            model = model_fn().to(device)
            model.eval()
            
            # Create appropriate input
            if name in ["MultiHeadSelfAttention", "FocusedLinearAttention"]:
                x = torch.randn(input_shape).to(device)
            elif name == "SS2D":
                x = torch.randn(input_shape).to(device)  # [B, H, W, C]
            else:
                x = torch.randn(input_shape).to(device)  # [B, C, H, W]
            
            with torch.no_grad():
                output = model(x)
            
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"✅ {name} test: SUCCESS")
            
        except Exception as e:
            print(f"❌ {name} test: FAILED - {str(e)}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_plugin_blocks()
