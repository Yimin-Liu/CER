""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from .pooling import build_pooling_layer
import copy


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class PatchMerging_1(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging_row(nn.Module):
    def __init__(self, input_resolution, dim, row, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)

        # self.reduction.bias.requires_grad_(False)
        # self.reduction.apply(weights_init_kaiming)
        # self.pooling = build_pooling_layer('avg')
        # self.pooling = nn.AdaptiveAvgPool2d((16, 1))
        self.pooling = GeneralizedMeanPooling(output_size=(16, 1))
        self.row = row

    def selected_area_row(self, x):
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, C, H, W)

        x0 = x[:, :, :, 0::8]
        x1 = x[:, :, :, 1::8]
        x2 = x[:, :, :, 2::8]
        x3 = x[:, :, :, 3::8]
        x4 = x[:, :, :, 4::8]
        x5 = x[:, :, :, 5::8]
        x6 = x[:, :, :, 6::8]
        x7 = x[:, :, :, 7::8]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -3).squeeze(-1).permute(0, 2,
                                                                                1)  # B H/2 W/2 4*C     64, 384, 16, 8

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        return x

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape

        global_feature = x[:, 0].unsqueeze(1)
        part_feature = x[:, 1:]
        part_feature = part_feature.permute(0, 2, 1).view(B, C, 16, 8)

        part_feature = self.pooling(part_feature).squeeze(-1).permute(0, 2, 1)

        # part_feature = self.selected_area_row(part_feature)

        # part_feature = self.norm(part_feature)
        part_feature = self.reduction(part_feature)

        x = torch.cat((global_feature, part_feature), dim=1)

        # x = torch.cat((global_feature, part_feature), dim=1)
        # x = self.norm(x)
        # x = self.reduction(x)

        return x


class PatchMerging_column_2(nn.Module):
    def __init__(self, input_resolution, dim, column, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim, dim, bias=False)
        self.norm = norm_layer(dim)
        # self.reduction.bias.requires_grad_(False)
        # self.reduction.apply(weights_init_kaiming)
        self.pooling = nn.AdaptiveAvgPool2d((column, 1))

        # self.pooling = GeneralizedMeanPooling(output_size=(column, 1))
        self.column = column

    def selected_area_column(self, x):
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.squeeze(-1)

        x0 = x[:, :, 0::8]
        x1 = x[:, :, 1::8]
        x2 = x[:, :, 2::8]
        x3 = x[:, :, 3::8]
        x4 = x[:, :, 4::8]
        x5 = x[:, :, 5::8]
        x6 = x[:, :, 6::8]
        x7 = x[:, :, 7::8]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -2).permute(0, 2, 1)  # B H/2 W/2 4*C     64, 384, 16, 8
        return x

    def forward(self, x):
        B, L, C = x.shape
        global_feature = x[:, 0].unsqueeze(1)
        part_feature = x[:, 1:]
        # part_feature = part_feature.permute(0, 2, 1).view(B, C, 16, 8)

        part_feature = part_feature.permute(0, 2, 1).unsqueeze(-1)
        part_feature = self.pooling(part_feature).squeeze(-1).permute(0, 2, 1)
        global_feature = self.norm(global_feature)
        # part_feature = self.selected_area_column(part_feature)
        # x = torch.cat((global_feature, part_feature), dim=1)
        # x = self.norm(x)
        # x = self.reduction(x)

        part_feature = self.norm(part_feature)
        # part_feature = self.reduction(part_feature)
        x = torch.cat((global_feature, part_feature), dim=1)

        return x


class PatchMerging_column_4(nn.Module):
    def __init__(self, input_resolution, dim, column, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim, dim, bias=False)
        # self.reduction.bias.requires_grad_(False)
        # self.reduction.apply(weights_init_kaiming)
        self.norm = norm_layer(dim)

        self.pooling = nn.AdaptiveAvgPool2d((column, 1))

        # self.pooling = GeneralizedMeanPooling(output_size=(column, 1))
        self.column = column

    def selected_area_column(self, x):
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.squeeze(-1)

        x0 = x[:, :, 0::4]
        x1 = x[:, :, 1::4]
        x2 = x[:, :, 2::4]
        x3 = x[:, :, 3::4]

        x = torch.cat([x0, x1, x2, x3], -2).permute(0, 2, 1)
        return x

    def forward(self, x):
        B, L, C = x.shape
        global_feature = x[:, 0].unsqueeze(1)
        part_feature = x[:, 1:]
        part_feature = part_feature.permute(0, 2, 1).unsqueeze(-1)  # .view(B, C, 16, 8)  # .unsqueeze(-1)

        part_feature = self.pooling(part_feature).squeeze(-1).permute(0, 2, 1)
        # part_feature = self.selected_area_column(part_feature)
        # x = torch.cat((global_feature, part_feature), dim=1)
        # x = self.norm(x)
        # x = self.reduction(x)
        global_feature = self.norm(global_feature)
        part_feature = self.norm(part_feature)
        # part_feature = self.reduction(part_feature)
        x = torch.cat((global_feature, part_feature), dim=1)

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class PatchEmbed_VOLO(nn.Module):
    """
    Image to Patch Embedding.
    Different with ViT use 1 conv layer, we use 4 conv layers to do patch embedding
    """

    def __init__(self, img_size=224, stem_conv=False, stem_stride=1,
                 patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384):
        super().__init__()
        assert patch_size in [4, 8, 16]
        img_size = to_2tuple(img_size)
        self.num_x = img_size[1] // patch_size
        self.num_y = img_size[0] // patch_size
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.stem_conv = stem_conv
        if stem_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride,
                          padding=3, bias=False),  # 112x112
                #  nn.BatchNorm2d(hidden_dim),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                #  nn.BatchNorm2d(hidden_dim),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.proj = nn.Conv2d(hidden_dim,
                              embed_dim,
                              kernel_size=patch_size // stem_stride,
                              stride=patch_size // stem_stride)
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)  # B, C, H, W  64, 384, 16, 8
        x = x.flatten(2).permute(0, 2, 1)
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, pretrained_path='', hw_ratio=1, conv_stem=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.in_planes = self.embed_dim = embed_dim
        if conv_stem:
            self.patch_embed = PatchEmbed_VOLO(img_size=img_size, stem_conv=True, stem_stride=2,
                                               patch_size=patch_size, in_chans=in_chans, hidden_dim=64, embed_dim=384)
            print('Using convolution stem')
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)
            print('Using standard patch embedding')

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.camera_nums = 8

        self.sie_embed = nn.Parameter(torch.zeros(self.camera_nums, 1, embed_dim))  # 修改
        trunc_normal_(self.sie_embed, std=.02)
        print('camera number is : {}'.format(6))
        print('using SIE_Lambda is : {}'.format(1.0))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])  # depth

        self.gap = build_pooling_layer('avg')

        self.patch_partition_row = PatchMerging_row(input_resolution=2, dim=384, row=8)

        self.patch_partition_column = PatchMerging_column_2(input_resolution=2, dim=384, column=2)

        # self.patch_partition_column_1 = PatchMerging_column_4(input_resolution=2, dim=384, column=3, norm_layer=nn.functional.normalize())

        self.patch_partition_column_1 = PatchMerging_column_4(input_resolution=2, dim=384, column=3)

        # self.blocks10_blocks0 = Block(
        #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[11], norm_layer=norm_layer)

        #
        self.blocks11_blocks0 = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[11], norm_layer=norm_layer)

        self.blocks11_blocks1 = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[11], norm_layer=norm_layer)

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        self.bottleneck = nn.BatchNorm1d(num_features=self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.BatchNorm2d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)

        self.part_bns = nn.ModuleList([
            self.make_bnneck(self.in_planes, weights_init_kaiming) for i in range(6)
        ])

        # --------------------------------------------
        # Part split settings
        self.num_parts = 5
        self.fmap_h = img_size[0] // 16
        self.fmap_w = img_size[1] // 16
        self.has_head = False
        self.granularities = [2, 3]
        self.has_ours = True

        # Two different granularity branches
        if self.has_head:
            block = self.blocks[-1]
            layer_norm = self.norm
            self.b1 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b2 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            for i, g in enumerate(self.granularities):
                setattr(self, 'b{}_pool'.format(i + 1), nn.AvgPool2d(kernel_size=(self.fmap_h // g, self.fmap_w),
                                                                     stride=(self.fmap_h // g,)))

        if self.has_ours:
            block = self.blocks[-1]
            layer_norm = self.norm
            self.b1 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )
            self.b2 = nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)
            )

            block_1 = self.blocks[-2]
            self.b3 = nn.Sequential(
                copy.deepcopy(block_1),
                copy.deepcopy(layer_norm)
            )
            self.b_base_pool_avg = nn.AvgPool2d(kernel_size=(1, self.fmap_w),
                                            stride=(1,))

            self.b_base_pool_max = nn.MaxPool2d(kernel_size=(1, self.fmap_w),
                                            stride=(1,))
            for i, g in enumerate(self.granularities):
                setattr(self, 'b{}_pool_avg'.format(i + 1), nn.AvgPool2d(kernel_size=(self.fmap_h // g, 1),
                                                                     stride=(self.fmap_h // g,)))

            for i, g in enumerate(self.granularities):
                setattr(self, 'b{}_pool_max'.format(i + 1), nn.MaxPool2d(kernel_size=(self.fmap_h // g, 1),
                                                                     stride=(self.fmap_h // g,)))

        print('num_parts={}, branch_parts={}'.format(self.num_parts, self.granularities))

        # Global bottleneck
        self.bottleneck = self.make_bnneck(self.in_planes, weights_init_kaiming)

        # Part bottleneck
        self.part_bns = nn.ModuleList([
            self.make_bnneck(self.in_planes, weights_init_kaiming) for i in range(self.num_parts)
        ])
        self.gem = GeneralizedMeanPooling()
        # ------------------------------------

        self.load_param(pretrained_path, hw_ratio)

    def make_bnneck(self, dims, init_func):
        bn = nn.BatchNorm1d(dims)
        bn.bias.requires_grad_(False)  # disable bias update
        bn.apply(init_func)
        return bn

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_multi_branch(self, x, label=None, cam_label=None, view_label=None):
        """
        ViT 1st ~ (L-1)-th layers, duplicated L-th layers as projection heads for two branches.
        """

        B = x.shape[0]
        x = self.patch_embed(x)  # [64,3,256,128]
        # [64, 128, 384]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # [64, 129, 384]

        x = x + self.pos_embed + 1.0 * self.sie_embed[cam_label]

        x = self.pos_drop(x)
        for blk in self.blocks[:-1]:
            x = blk(x)  # output before last layer

        # gf = self.gem(x[:, 1:].permute(0, 2, 1)).squeeze()
        # x = x[:, 0] + gf
        # B = x.size(0)
        # Split after head
        # branch 1
        x_b1 = self.b1(x)  # (B, L, C)
        x_b1_glb = x_b1[:, 0, :]  # (B, C)
        x_b1_patch = x_b1[:, 1:, :]  # (B, L-1, C)
        x_b1_patch = x_b1_patch.permute(0, 2, 1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b1_patch = self.b1_pool(x_b1_patch).squeeze()  # (B, C, P1)

        # branch 2
        x_b2 = self.b2(x)
        x_b2_glb = x_b2[:, 0, :]
        x_b2_patch = x_b2[:, 1:, :]
        x_b2_patch = x_b2_patch.permute(0, 2, 1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_b2_patch = self.b2_pool(x_b2_patch).squeeze()  # (B, C, P2)

        # Mean global feature
        x_glb = 0.5 * (x_b1_glb + x_b2_glb)  # (B, C)
        # if self.global_feature_type == 'mean':
        #     x_glb = 0.5 * (x_b1_glb + x_b2_glb)  # (B, C)
        # elif self.global_feature_type == 'b1':
        #     x_glb = x_b1_glb
        # elif self.global_feature_type == 'b2':
        #     x_glb = x_b2_glb
        # else:
        #     raise ValueError('Invalid global feature type: {}'.format(self.global_feature_type))

        # Stack two branch part features
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2)  # (B, C, P), P = P1 + P2

        return x_glb, x_part

    def forward_ours(self, x, label=None, cam_label=None, view_label=None):
        """
        ViT 1st ~ (L-1)-th layers, duplicated L-th layers as projection heads for two branches.
        """

        B = x.shape[0]
        x = self.patch_embed(x)  # [64,3,256,128]
        # [64, 128, 384]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # [64, 129, 384]

        # x = x + self.pos_embed
        x = x + self.pos_embed + 1.0 * self.sie_embed[cam_label]

        x = self.pos_drop(x)
        for blk in self.blocks[:-2]:
            x = blk(x)  # output before last layer

        # gf = self.gem(x[:, 1:].permute(0, 2, 1)).squeeze()
        # x = x[:, 0] + gf
        # B = x.size(0)
        # Split after head
        # branch 1

        # x = self.b_base_pool(x)
        # x = self.b3(x)
        x_glb = x[:, 0, :]  # (B, C)
        x_patch = x[:, 1:, :]  # (B, L-1, C)
        x_patch = x_patch.permute(0, 2, 1).reshape((B, self.in_planes, self.fmap_h, self.fmap_w))
        x_patch = (self.b_base_pool_avg(x_patch) + self.b_base_pool_max(x_patch)) / 2
        x = torch.cat((x_glb.unsqueeze(2), x_patch.squeeze()), dim=2).permute(0, 2, 1)

        x = self.b3(x)

        # x_b1 = self.b1(x)  # (B, L, C)
        x_b1_glb = x[:, 0, :]  # (B, C)
        x_b1_patch = x[:, 1:, :]  # (B, L-1, C)
        x_b1_patch = x_b1_patch.permute(0, 2, 1).reshape((B, self.in_planes, 16, 1))
        x_b1_patch = (self.b1_pool_avg(x_b1_patch).squeeze() + self.b1_pool_max(x_b1_patch).squeeze()) / 2   # (B, C, P1)
        x_b1 = torch.cat((x_b1_glb.unsqueeze(2), x_b1_patch.squeeze()), dim=2).permute(0, 2, 1)
        x_b1 = self.b1(x_b1)  # (B, L, C)
        x_b1_glb = x_b1[:, 0, :]  # (B, C)
        x_b1_patch = x_b1[:, 1:, :].permute(0, 2, 1)  # (B, L-1, C)

        # branch 2
        # x_b2 = self.b2(x)
        x_b2_glb = x[:, 0, :]
        x_b2_patch = x[:, 1:, :]
        x_b2_patch = x_b2_patch.permute(0, 2, 1).reshape((B, self.in_planes, 16, 1))
        x_b2_patch = (self.b2_pool_avg(x_b2_patch).squeeze() + self.b2_pool_max(x_b2_patch).squeeze()) / 2  # (B, C, P2)
        x_b2 = torch.cat((x_b2_glb.unsqueeze(2), x_b2_patch.squeeze()), dim=2).permute(0, 2, 1)
        x_b2 = self.b2(x_b2)
        x_b2_glb = x_b2[:, 0, :]
        x_b2_patch = x_b2[:, 1:, :].permute(0, 2, 1)

        # Mean global feature
        x_glb = 0.5 * (x_b1_glb + x_b2_glb)  # (B, C)
        # if self.global_feature_type == 'mean':
        #     x_glb = 0.5 * (x_b1_glb + x_b2_glb)  # (B, C)
        # elif self.global_feature_type == 'b1':
        #     x_glb = x_b1_glb
        # elif self.global_feature_type == 'b2':
        #     x_glb = x_b2_glb
        # else:
        #     raise ValueError('Invalid global feature type: {}'.format(self.global_feature_type))

        # Stack two branch part features
        x_part = torch.cat([x_b1_patch, x_b2_patch], dim=2)  # (B, C, P), P = P1 + P2

        return x_glb, x_part

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [64,3,256,128]
        # [64, 128, 384]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # [64, 129, 384]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        use_part = True
        if use_part:
            # x = self.patch_partition_row(x)
            # x = self.blocks10_blocks0(x)
            #
            # x_1 = self.patch_partition_column(x)
            # x_1 = self.blocks11_blocks0(x_1)
            #
            # x_2 = self.patch_partition_column_1(x)
            # x_2 = self.blocks11_blocks1(x_2)

            x_1 = self.blocks11_blocks0(x)
            x_1 = self.patch_partition_column(x_1)

            # x_1 = self.patch_partition_column(x)
            x_2 = self.blocks11_blocks1(x)
            x_2 = self.patch_partition_column_1(x_2)

            # x_2 = self.patch_partition_column_1(x_1)
            # x_2 = self.blocks11_blocks2(x_2)

        # x_1 = self.norm(x_1)
        # x_2 = self.norm(x_2)

        x_part = torch.cat([x_1[:, 1:].permute(0, 2, 1), x_2[:, 1:].permute(0, 2, 1)], dim=2)

        x_glb = self.bottleneck((x_1[:, 0] + x_2[:, 0]) / 2)
        x_part = torch.stack([self.part_bns[i](x_part[:, :, i]) for i in range(x_part.size(2))], dim=2)

        x_glb = F.normalize(x_glb, dim=1)
        x_part = F.normalize(x_part, dim=1)

        x_part = x_part.permute(0, 2, 1)
        x_part_1 = x_part[:, :2, :]
        x_part_2 = x_part[:, -3:, :]

        return x_glb, x_part_1, x_part_2

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # x, x_part1, x_part2 = self.forward_features(x)
        # # bn_x = feat = self.bottleneck(x)
        # # bn_x = F.normalize(bn_x)
        # # bn_x_part1 = self.bottleneck(x_part1.permute(0, 2, 1)).permute(0, 2, 1)
        # # bn_x_part1 = F.normalize(bn_x_part1, dim=2)
        # # bn_x_part2 = self.bottleneck(x_part2.permute(0, 2, 1)).permute(0, 2, 1)
        # # bn_x_part2 = F.normalize(bn_x_part2, dim=2)
        #
        # bn_x_part = torch.cat((x_part1, x_part2), dim=1)
        x_glb, x_part = self.forward_ours(x, label, cam_label, view_label) # forward
        x_glb = self.bottleneck(x_glb)
        x_part = torch.stack([self.part_bns[i](x_part[:, :, i]) for i in range(x_part.size(2))], dim=2)

        x_glb = F.normalize(x_glb, dim=1)
        x_part = F.normalize(x_part, dim=1).permute(0, 2, 1)

        return x_glb, x_part

    def load_param(self, model_path, hw_ratio):
        param_dict = torch.load(model_path, map_location='cpu')
        count = 0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'teacher' in param_dict:  ### for dino
            obj = param_dict["teacher"]
            print('Convert dino model......')
            newmodel = {}
            for k, v in obj.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if not k.startswith("backbone."):
                    continue
                old_k = k
                k = k.replace("backbone.", "")
                newmodel[k] = v
                param_dict = newmodel
        for k, v in param_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            if k.startswith('base'):
                k = k.replace('base.', '')
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
                continue
            if 'fc.' in k or 'classifier' in k or 'bottleneck' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x,
                                     hw_ratio=hw_ratio)
            try:
                self.state_dict()[k].copy_(v)

                # if k.startswith("blocks.11."):
                #     for i in range(3):
                #         self.state_dict()["blocks11_blocks" + str(i) + k[9:]].copy_(v)

                # if k.startswith("blocks.10."):
                #     self.state_dict()["blocks10_blocks" + str(0) + k[9:]].copy_(v)
                # else:
                #     self.state_dict()[k].copy_(v)

                # if k.startswith("blocks.11."):
                #     for i in range(2):
                #         self.state_dict()["blocks11_blocks" + str(i) + k[9:]].copy_(v)
                # else:
                #     self.state_dict()[k].copy_(v)

                count += 1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))
        print('Load %d / %d layers.' % (count, len(self.state_dict().keys())))


def resize_pos_embed(posemb, posemb_new, hight, width, hw_ratio):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old_h = int(math.sqrt(len(posemb_grid) * hw_ratio))
    gs_old_w = gs_old_h // hw_ratio
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_part(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        # nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
