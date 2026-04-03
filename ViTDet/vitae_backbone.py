"""Standalone ViTAE-Small backbone for object detection.

Extracted from the ViTDet mmdetection repo to work without mmcv/mmdet.
Produces 4-scale feature maps (strides 4, 8, 16, 32) compatible with FPN.
"""

import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─── Window Attention utilities ───────────────────────────────────────────────


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def calc_rel_pos_spatial(attn, q, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    B, n_head, q_N, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)
    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)
    return attn


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        q_size = window_size[0]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        B_, N, C = x.shape
        x = x.reshape(B_, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        x = window_partition(x, self.window_size[0])
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)
        B_w = x.shape[0]
        N_w = x.shape[1]
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B_, H * W, C)
        return x


# ─── Token modules ────────────────────────────────────────────────────────────


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8
        self.drop_path = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb), nn.GELU(),
            nn.Linear(1 * self.emb, self.emb), nn.Dropout(dp2),
        )
        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)
        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)
        y = v + self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _TokenTransformerAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=None, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = in_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        v = v.permute(0, 2, 1, 3).view(B, N, self.in_dim).contiguous()
        x = v + x
        return x


class Token_transformer(nn.Module):
    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _TokenTransformerAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = _Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio),
                         out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ─── Normal & Reduction Cells ────────────────────────────────────────────────


class _NCAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None, window_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.window_size = window_size
        q_size = window_size[0]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NormalCell(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, group=64, tokens_type='transformer',
                 img_size=224, window=False, window_size=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.img_size = img_size
        self.window_size = window_size
        self.tokens_type = tokens_type
        assert tokens_type == 'transformer'
        if not window:
            self.attn = _NCAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.PCM = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group), nn.BatchNorm2d(mlp_hidden_dim), nn.SiLU(inplace=True),
            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group), nn.BatchNorm2d(dim), nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, 1, group), nn.SiLU(inplace=True),
        )
        self.H = 0
        self.W = 0

    def forward(self, x):
        b, n, c = x.shape
        H, W = self.H, self.W
        x_2d = x.view(b, H, W, c).permute(0, 3, 1, 2).contiguous()
        convX = self.drop_path(self.PCM(x_2d).permute(0, 2, 3, 1).contiguous().view(b, n, c))
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PRM(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4,
                 dilations=[1, 6, 12], in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.patch_shape = (self.img_size[0] // downsample_ratio, self.img_size[1] // downsample_ratio)
        self.convs = nn.ModuleList()
        for dilation in self.dilations:
            padding = math.ceil(((self.kernel_size - 1) * dilation + 1 - self.stride) / 2)
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=self.kernel_size,
                          stride=self.stride, padding=padding, dilation=dilation),
                nn.GELU()
            ))
        self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, H, W = x.shape
        padding = math.ceil(((self.kernel_size - 1) * self.dilations[0] + 1 - self.stride) / 2)
        padding = [padding, padding]
        extra_padding = math.ceil(self.stride / 2) - 1
        wP = hP = False
        if H % self.downsample_ratio != 0:
            hP = True
            padding[0] += extra_padding
        if W % self.downsample_ratio != 0:
            wP = True
            padding[1] += extra_padding
        y = F.gelu(F.conv2d(x, self.convs[0][0].weight, self.convs[0][0].bias,
                             self.stride, tuple(padding), self.dilations[0])).unsqueeze(dim=-1)
        for i in range(1, len(self.dilations)):
            pad = math.ceil(((self.kernel_size - 1) * self.dilations[i] + 1 - self.stride) / 2)
            pad = [pad, pad]
            if hP:
                pad[0] += extra_padding
            if wP:
                pad[1] += extra_padding
            _y = F.gelu(F.conv2d(x, self.convs[i][0].weight, self.convs[i][0].bias,
                                  self.stride, tuple(pad), self.dilations[i])).unsqueeze(dim=-1)
            y = torch.cat((y, _y), dim=-1)
        B, C, H, W, N = y.shape
        y = y.permute(0, 4, 1, 2, 3).flatten(3).reshape(B, N * C, W * H).permute(0, 2, 1).contiguous()
        return y, (H, W)


class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64,
                 downsample_ratios=4, kernel_size=7, num_heads=1,
                 dilations=[1, 2, 3, 4], tokens_type='performer', group=1,
                 attn_drop=0., drop_path=0., mlp_ratio=1.0, window_size=(14, 14)):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        PCMStride = []
        residual = downsample_ratios // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None
        self.tokens_type = tokens_type
        drop = 0.
        if tokens_type == 'pooling':
            PCMStride = [1, 1, 1]
            self.pool = nn.MaxPool2d(downsample_ratios, stride=downsample_ratios, padding=0)
            tokens_type = 'transformer'
            downsample_ratios = 1
        self.PCM = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, 3, PCMStride[0], 1, groups=group), nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 3, PCMStride[1], 1, groups=group), nn.BatchNorm2d(embed_dims), nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, token_dims, 3, PCMStride[2], 1, groups=group), nn.SiLU(inplace=True),
        )
        self.PRM = PRM(img_size=img_size, kernel_size=kernel_size,
                        downsample_ratio=downsample_ratios, dilations=self.dilations,
                        in_chans=in_chans, embed_dim=embed_dims)
        in_chans = self.PRM.out_chans
        self.patch_shape = self.PRM.patch_shape
        if tokens_type == 'performer':
            self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)
        elif tokens_type == 'performer_less':
            self.attn = None
            self.PCM = None
        elif tokens_type == 'transformer':
            self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads,
                                           mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path)

    def forward(self, x, size):
        H, W = size
        if len(x.shape) < 4:
            B, N, C = x.shape
            x = x.reshape(B, H, W, C).contiguous().permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.PRM(x)
        H, W = math.ceil(H / self.downsample_ratios), math.ceil(W / self.downsample_ratios)
        B, N, C = PRM_x.shape
        assert N == H * W, f"N={N}, H={H}, W={W}, shape={shortcut.shape}"
        if self.attn is None:
            return PRM_x, (H, W)
        convX = self.PCM(shortcut)
        x = self.attn.attn(self.attn.norm1(PRM_x))
        convX = convX.permute(0, 2, 3, 1).reshape(*x.shape).contiguous()
        x = x + self.attn.drop_path(convX)
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))
        return x, (H, W)


# ─── Main backbone ────────────────────────────────────────────────────────────


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = to_2tuple(img_size)
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, 3, 2, 1, bias=False), nn.BatchNorm2d(inter_channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.patch_shape = (img_size[0] // 4, img_size[1] // 4)

    def forward(self, x, size):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x, (h, w)


class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64,
                 downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6,
                 dilations=[1, 2, 3, 4], RC_tokens_type='performer',
                 NC_tokens_type='transformer', RC_group=1, NC_group=64,
                 NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0, attn_drop=0., norm_layer=nn.LayerNorm,
                 window_size=(14, 14), use_checkpoint=False,
                 globalBlock=[2, 5, 8, 13]):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.use_checkpoint = use_checkpoint
        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims // 2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                                     RC_heads, dilations, tokens_type=RC_tokens_type, group=RC_group)
        else:
            self.RC = nn.Identity()
        full_image_size = self.RC.patch_shape
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr,
                       norm_layer=norm_layer, group=NC_group, tokens_type=NC_tokens_type,
                       img_size=img_size // downsample_ratios, window_size=window_size if i not in globalBlock else full_image_size,
                       window=True if i not in globalBlock else False)
            for i in range(NC_depth)
        ])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        for nc in self.NC:
            nc.H = h
            nc.W = w
            if self.use_checkpoint and hasattr(nc, 'globalBlock') and nc.globalBlock:
                x = checkpoint.checkpoint(nc, x)
            else:
                x = nc(x)
        return x, (h, w)


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ViTAE_S(nn.Module):
    """Standalone ViTAE-Small backbone (no mmdet dependency)."""

    def __init__(self, img_size=224, in_chans=3, stages=4,
                 embed_dims=64, token_dims=64,
                 downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 3, 3, 3],
                 RC_heads=[1, 1, 1, 1], NC_heads=4,
                 dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                 RC_tokens_type='transformer', NC_tokens_type='transformer',
                 RC_group=[1, 1, 1, 1], NC_group=[1, 32, 64, 64],
                 NC_depth=[2, 2, 6, 2], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 window_size=(14, 14), out_indices=(0, 1, 2, 3),
                 frozen_stages=-1, use_checkpoint=False, load_ema=True,
                 globalBlock=[2, 5, 8, 13]):
        super().__init__()
        self.stages = stages
        self.load_ema = load_ema
        rep = lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)]
        self.embed_dims = rep(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = rep(downsample_ratios, stages)
        self.kernel_size = rep(kernel_size, stages)
        self.RC_heads = rep(RC_heads, stages)
        self.NC_heads = rep(NC_heads, stages)
        self.dilaions = rep(dilations, stages)
        self.RC_tokens_type = rep(RC_tokens_type, stages)
        self.NC_tokens_type = rep(NC_tokens_type, stages)
        self.RC_group = rep(RC_group, stages)
        self.NC_group = rep(NC_group, stages)
        self.NC_depth = rep(NC_depth, stages)
        self.mlp_ratio = rep(mlp_ratio, stages)
        self.qkv_bias = rep(qkv_bias, stages)
        self.qk_scale = rep(qk_scale, stages)
        self.drop = rep(drop_rate, stages)
        self.attn_drop = rep(attn_drop_rate, stages)
        self.norm_layer = rep(norm_layer, stages)
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.pos_drop = nn.Dropout(p=drop_rate)
        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        Layers = []
        for i in range(stages):
            startDpr = 0 if i == 0 else self.NC_depth[i - 1]
            Layers.append(BasicLayer(
                img_size, in_chans, self.embed_dims[i], self.tokens_dims[i],
                self.downsample_ratios[i], self.kernel_size[i], self.RC_heads[i],
                self.NC_heads[i], self.dilaions[i], self.RC_tokens_type[i],
                self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i],
                self.NC_depth[i], dpr[startDpr:self.NC_depth[i] + startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i],
                qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], window_size=window_size,
                use_checkpoint=use_checkpoint, globalBlock=globalBlock,
            ))
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)
        self.num_layers = len(Layers)
        embed_dim = self.tokens_dims[-1]
        self.norm = norm_layer(embed_dim)
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2), Norm2d(embed_dim), nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)
        self.depth = depth + stages
        self.out_channels = embed_dim  # for FPN integration
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x):
        b, _, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
        x = self.norm(x)
        xp = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        return OrderedDict([(str(i), op(xp)) for i, op in enumerate(ops)])

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def get_num_layers(self):
        return self.depth


# ─── Config presets ───────────────────────────────────────────────────────────


def vitae_small_config(img_size=512):
    """Return kwargs matching the ViTDet-ViTAE-Small-100e config."""
    return dict(
        img_size=img_size,
        stages=3,
        embed_dims=[64, 64, 192],
        token_dims=[96, 192, 384],
        downsample_ratios=[4, 2, 2],
        kernel_size=[7, 3, 3],
        RC_heads=[1, 1, 1],
        NC_heads=[1, 1, 6],
        dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2]],
        RC_tokens_type=['performer', 'performer', 'performer_less'],
        NC_tokens_type=['transformer', 'transformer', 'transformer'],
        RC_group=[1, 1, 1],
        NC_group=[1, 1, 96],
        NC_depth=[0, 0, 14],
        mlp_ratio=3.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        window_size=(14, 14),
        use_checkpoint=False,
        load_ema=True,
        globalBlock=[2, 5, 8, 13],
    )


def build_vitae_small(img_size=512, pretrained=None):
    """Build ViTAE-Small and optionally load pretrained weights."""
    cfg = vitae_small_config(img_size)
    model = ViTAE_S(**cfg)
    model.init_weights()
    if pretrained is not None:
        load_vitae_pretrained(model, pretrained, img_size)
    return model


def load_vitae_pretrained(model, checkpoint_path, img_size=512):
    """Load pretrained weights from mmdet-style checkpoint.

    Handles:
    - Stripping 'backbone.' prefix from state_dict keys
    - Interpolating relative position embeddings when img_size differs from 1024
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Extract backbone state dict
    if "state_dict_ema" in ckpt:
        sd = ckpt["state_dict_ema"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # Strip 'backbone.' prefix
    backbone_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v

    if not backbone_sd:
        # Maybe the checkpoint IS the backbone already
        backbone_sd = sd

    # Interpolate rel_pos weights if img_size != 1024
    model_sd = model.state_dict()
    for k in list(backbone_sd.keys()):
        if "rel_pos" in k and k in model_sd:
            src = backbone_sd[k]
            dst = model_sd[k]
            if src.shape != dst.shape:
                # Interpolate along spatial dimension (dim 0)
                src_2d = src.unsqueeze(0).permute(0, 2, 1)  # (1, head_dim, src_len)
                dst_2d = F.interpolate(src_2d, size=dst.shape[0], mode="linear", align_corners=False)
                backbone_sd[k] = dst_2d.permute(0, 2, 1).squeeze(0)
                print(f"  Interpolated {k}: {src.shape} → {backbone_sd[k].shape}")

    msg = model.load_state_dict(backbone_sd, strict=False)
    print(f"[ViTAE_S] Loaded pretrained from {checkpoint_path}")
    print(f"  missing : {len(msg.missing_keys)} keys")
    print(f"  unexpected: {len(msg.unexpected_keys)} keys")
    if msg.missing_keys:
        print(f"  missing examples: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"  unexpected examples: {msg.unexpected_keys[:5]}")
    return msg
