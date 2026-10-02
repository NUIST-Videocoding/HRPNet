from torch._tensor import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from collections import OrderedDict
import time
from einops import rearrange, repeat
from functools import partial
from pdb import set_trace as stx
from typing import Optional, Callable
import math
import numbers
import sys
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.GELU()

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class ffn(nn.Module):
    def __init__(self, num_feat, ffn_expand=2, activation=nn.GELU, use_attention=True):
        super(ffn, self).__init__()
        self.dw_channel = num_feat * ffn_expand
        self.conv1 = nn.Conv2d(num_feat, self.dw_channel, kernel_size=1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=self.dw_channel)
        self.conv3 = nn.Conv2d(self.dw_channel // 2, num_feat, kernel_size=1, padding=0, stride=1)
        self.activation = activation()
        self.use_attention = use_attention
        if use_attention:
            self.eca = ECAAttention(kernel_size=3)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = self.activation(x1) * x2
        if self.use_attention:
            x = self.eca(x)
        x = self.conv3(x)
        return x



class SFTLayer(nn.Module):
    def __init__(self, in_channels, condition_channels):
        """
        SFT Layer to modulate image features with semantic condition features.
        
        Args:
            in_channels (int): Channel number of the image feature map to be modulated.
            condition_channels (int): Channel number of the semantic condition feature map.
        """
        super(SFTLayer, self).__init__()
        mid_channels = in_channels // 2
        
        self.condition_processor = nn.Sequential(
            nn.Conv2d(condition_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels * 2, kernel_size=3, padding=1)
        )
        
    def forward(self, image_feature, condition_feature):
        """
        Args:
            image_feature (torch.Tensor): The feature map from the main U-Net branch.
            condition_feature (torch.Tensor): The semantic prior feature map (e.g., from DINOv2).
        
        Returns:
            torch.Tensor: The modulated feature map.
        """
        gamma_beta = self.condition_processor(condition_feature)
        
        if gamma_beta.size()[-2:] != image_feature.size()[-2:]:
            gamma_beta = F.interpolate(
                gamma_beta, 
                size=image_feature.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            
        gamma, beta = gamma_beta.chunk(2, dim=1)
        

        
        return modulated_feature

class MultiScaleSpatialEnhance(nn.Module):
    def __init__(self, nc, expand_ratio=2):
        super(MultiScaleSpatialEnhance, self).__init__()
        
        self.norm_in = nn.BatchNorm2d(nc)

        expand_nc = int(nc * expand_ratio)
        self.expand_conv = nn.Conv2d(nc, expand_nc, kernel_size=1)
        
        self.split_channels = expand_nc // 4
        
        self.branch3 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1, groups=self.split_channels)
        self.branch5 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=5, padding=2, groups=self.split_channels)
        self.branch_dil = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=2, dilation=2, groups=self.split_channels)
        

        self.activation = nn.GELU()

        self.compress_conv = nn.Conv2d(expand_nc, nc, kernel_size=1)

    def forward(self, x):
        identity = x
        x = self.norm_in(x)
        x = self.expand_conv(x)
        x_splits = list(torch.split(x, self.split_channels, dim=1))
        x_splits[1] = self.branch3(x_splits[1])
        x_splits[2] = self.branch5(x_splits[2])
        x_splits[3] = self.branch_dil(x_splits[3])
        out = torch.cat(x_splits, dim=1)
        out = self.activation(out)
        out = self.compress_conv(out)
        out = out + identity
        
        return out

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))
        self.ca = ChannelAttention(nc)
        self.spatial_enhance = MultiScaleSpatialEnhance(nc)

    def forward(self, x):
        identity = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        mag = self.ca(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out = x_out + identity
        out = self.spatial_enhance(x_out)

        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)

        diff_h = bridge.size(2) - up.size(2)
        diff_w = bridge.size(3) - up.size(3)

        up = F.pad(up, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        out = torch.cat([up, bridge], dim=1)
        out = self.conv_block(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


class SFTFreBlock(nn.Module):

    def __init__(self, in_channels, condition_channels):
        super(SFTFreBlock, self).__init__()
        self.sft = SFTLayer(in_channels=in_channels, condition_channels=condition_channels)
        self.fremlp = FreMLP(nc=in_channels)

    def forward(self, x, condition_feature):
        x = self.sft(x, condition_feature)
        x = self.fremlp(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_chn=3, wf=16, up_mode='upsample', conv1=conv):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')

        dino_feat_dim = 384 
        self.repeat_times_encoder = 1 
        self.repeat_times_decoder = 2 
        self.layer0 = conv1(in_chn, wf, kernel_size=3, stride=1)
        self.layer1 = UNetConvBlock(in_chans=wf, out_chans=32)
        self.layer2 = UNetConvBlock(in_chans=32, out_chans=64)
        self.layer3 = UNetConvBlock(in_chans=64, out_chans=128)
        self.enc_dual_blocks_0 = nn.ModuleList([
            SFTFreBlock(in_channels=wf, condition_channels=dino_feat_dim) 
            for _ in range(self.repeat_times_encoder)
        ])
        self.enc_dual_blocks_1 = nn.ModuleList([
            SFTFreBlock(in_channels=32, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_encoder)
        ])
        self.enc_dual_blocks_2 = nn.ModuleList([
            SFTFreBlock(in_channels=64, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_encoder)
        ])
        self.enc_dual_blocks_3 = nn.ModuleList([
            SFTFreBlock(in_channels=128, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_encoder)
        ])
        self.layer4 = UNetConvBlock(in_chans=128, out_chans=256)
        self.up_3 = UNetUpBlock(in_chans=256, out_chans=128, up_mode=up_mode)
        self.up_2 = UNetUpBlock(in_chans=128, out_chans=64, up_mode=up_mode)
        self.up_1 = UNetUpBlock(in_chans=64, out_chans=32, up_mode=up_mode)
        self.up_0 = UNetUpBlock(in_chans=32, out_chans=16, up_mode=up_mode)
        self.dec_dual_blocks_3 = nn.ModuleList([
            SFTFreBlock(in_channels=128, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_decoder)
        ])
        self.dec_dual_blocks_2 = nn.ModuleList([
            SFTFreBlock(in_channels=64, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_decoder)
        ])
        self.dec_dual_blocks_1 = nn.ModuleList([
            SFTFreBlock(in_channels=32, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_decoder)
        ])
        self.dec_dual_blocks_0 = nn.ModuleList([
            SFTFreBlock(in_channels=16, condition_channels=dino_feat_dim)
            for _ in range(self.repeat_times_decoder)
        ])
        self.last = conv1(wf, in_chn, kernel_size=3, stride=1)

    def forward(self, x, semantic_priors):

        blocks = []
        x0_img = self.layer0(x) 
        x0_modulated = x0_img # 初始化
        for block in self.enc_dual_blocks_0:
            x0_modulated = block(x0_modulated, semantic_priors[0]) # 循环更新
        blocks.append(x0_modulated)
        x1_img = self.layer1(x0_modulated)
        x1_modulated = x1_img # 初始化
        for block in self.enc_dual_blocks_1:
            x1_modulated = block(x1_modulated, semantic_priors[1]) # 循环更新
        blocks.append(x1_modulated)
        x2_img = self.layer2(x1_modulated)
        x2_modulated = x2_img # 初始化
        for block in self.enc_dual_blocks_2:
            x2_modulated = block(x2_modulated, semantic_priors[2]) # 循环更新
        blocks.append(x2_modulated)
        x3_img = self.layer3(x2_modulated)
        x3_modulated = x3_img # 初始化
        for block in self.enc_dual_blocks_3:
            x3_modulated = block(x3_modulated, semantic_priors[3]) # 循环更新
        blocks.append(x3_modulated)
        x4 = self.layer4(x3_modulated)
        x3_up = self.up_3(x4, blocks[-1])
        x3_modulated_dec = x3_up # 初始化
        for block in self.dec_dual_blocks_3:
            x3_modulated_dec = block(x3_modulated_dec, semantic_priors[3]) # 循环更新
        x2_up = self.up_2(x3_modulated_dec, blocks[-2])
        x2_modulated_dec = x2_up # 初始化
        for block in self.dec_dual_blocks_2:
            x2_modulated_dec = block(x2_modulated_dec, semantic_priors[2]) # 循环更新
        x1_up = self.up_1(x2_modulated_dec, blocks[-3])
        x1_modulated_dec = x1_up # 初始化
        for block in self.dec_dual_blocks_1:
            x1_modulated_dec = block(x1_modulated_dec, semantic_priors[1]) # 循环更新
        x0_up = self.up_0(x1_modulated_dec, blocks[-4])
        x0_modulated_dec = x0_up # 初始化
        for block in self.dec_dual_blocks_0:
            x0_modulated_dec = block(x0_modulated_dec, semantic_priors[0]) # 循环更新
        residual = self.last(x0_modulated_dec)
        output = x + residual 
        
        return torch.sigmoid(output)