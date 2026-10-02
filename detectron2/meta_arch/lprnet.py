import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .vssd import VMAMBA2Block

class CropLayer(nn.Module):
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]
class VCRBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=None):
        super(VCRBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        center_offset = padding - kernel_size // 2
        ver_pad = (center_offset + 1, center_offset)
        hor_pad = (center_offset, center_offset + 1)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                                 stride=1, padding=padding)
        self.conv_ver = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                                  stride=1, padding=(padding, 0), padding_mode='zeros') 
        self.conv_hor = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                                  stride=1, padding=(0, padding), padding_mode='zeros')

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()
        self.ver_pad_val = ver_pad
        self.hor_pad_val = hor_pad

    def forward(self, x):
        out_3x3 = self.conv3x3(x)
        out_ver = self.conv_ver(x)
        out_hor = self.conv_hor(x)
        out = out_3x3 + out_ver + out_hor
        return self.relu(self.bn(out))
class SemanticsRefinementBranch(nn.Module):
    def __init__(self, channels=384, num_blocks=3):
        super(SemanticsRefinementBranch, self).__init__()
        
        layers = []
        for _ in range(num_blocks):
            layers.append(VCRBlock(channels))
        
        self.body = nn.Sequential(*layers)
        self.tail = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.body(x)
        residual = self.tail(residual)
        return x + residual

class AsymmetricConvBranch(nn.Module):
    """ 非对称卷积分支: Nx1 -> 1xN """
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), 
                               padding=(pad, 0), groups=channels) # Depthwise
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), 
                               padding=(0, pad), groups=channels) # Depthwise
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.conv2(self.act(self.conv1(x)))
class MyVCRNet(nn.Module):
    def __init__(self, in_channels=384):
        super(MyVCRNet, self).__init__()
        
        self.branches = nn.ModuleList([
            SemanticsRefinementBranch(channels=in_channels, num_blocks=3), # 对应第1个语义特征
            SemanticsRefinementBranch(channels=in_channels, num_blocks=3), # 对应第2个语义特征
            SemanticsRefinementBranch(channels=in_channels, num_blocks=3), # 对应第3个语义特征
            SemanticsRefinementBranch(channels=in_channels, num_blocks=3)  # 对应第4个语义特征
        ])

    def forward(self, x_list):

        out_list = []
        for x, branch in zip(x_list, self.branches):
            if x is not None:
                out = branch(x)
                out_list.append(out)
            else:
                out_list.append(None)
                
        return out_list
class GlobalContextBranch(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_out = self.norm(x_flat + attn_out)
        return x_out.transpose(1, 2).view(B, C, H, W)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class RealFourierMambaBranch(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.freq_mamba = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            VMAMBA2Block(dim=channels * 2),
            nn.Conv2d(channels * 2, channels * 2, 1), 
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x_pre = self.pre_conv(x)
        ffted = torch.fft.rfft2(x_pre.float(), norm='ortho')
        real_part = ffted.real
        imag_part = ffted.imag
        freq_feat = torch.cat([real_part, imag_part], dim=1)
        freq_feat_refined = self.freq_mamba(freq_feat)
        c = x.shape[1]
        real_refined, imag_refined = torch.split(freq_feat_refined, c, dim=1)
        ffted_refined = torch.complex(real_refined, imag_refined)
        output_spatial = torch.fft.irfft2(ffted_refined, s=(H, W), norm='ortho')
        global_feature = self.post_conv(output_spatial)
        
        return global_feature

class MultiScaleGLAttention(nn.Module):

    def __init__(self, n_feats):
        super().__init__()
        
        self.n_feats = n_feats
        half_feats = n_feats // 2
        self.pre_conv = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, bias=False),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.RFM = RealFourierMambaBranch(half_feats)
        self.VCR3 = VCRBlock(half_feats, kernel_size=3)
        self.VCR5 = VCRBlock(half_feats, kernel_size=5)
        self.VCR7 = VCRBlock(half_feats, kernel_size=7)
        self.proj_last = nn.Conv2d(n_feats * 3, n_feats, 1, 1, 0)
        
    def forward(self, x):

        x_pre = self.pre_conv(x)
        x_global, x_local = torch.chunk(x_pre, 2, dim=1)        
        global_feat = self.RFM(x_global)
        out_1 = torch.cat([global_feat, self.VCR3(x_local)], dim=1)
        out_2 = torch.cat([global_feat, self.VCR5(x_local)], dim=1)
        out_3 = torch.cat([global_feat, self.VCR7(x_local)], dim=1)
        
        out1 = channel_shuffle(out_1, groups=2) + x_pre
        out2 = channel_shuffle(out_2, groups=2) + x_pre
        out3 = channel_shuffle(out_3, groups=2) + x_pre
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.proj_last(out) * self.scale + x
        
        return out
class GLI_Block(nn.Module):

    def __init__(self, dim, num_blocks=3):
        super().__init__()
        
        layers = []
        for _ in range(num_blocks):
            layers.append(MultiScaleGLAttention(dim))
        
        self.body = nn.Sequential(*layers)
        self.tail = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.body(x)
        residual = self.tail(residual)
        return x + residual
class GL_Net(nn.Module):
    def __init__(self, dim=384, num_blocks=3):
        super().__init__()
        self.stage_12 = GLI_Block(dim, num_blocks=num_blocks)
        self.stage_9 = GLI_Block(dim, num_blocks=num_blocks)
        self.stage_6 = GLI_Block(dim, num_blocks=num_blocks)
        self.stage_3 = GLI_Block(dim, num_blocks=num_blocks)
        
    def forward(self, features):
        f3, f6, f9, f12 = features
        r12 = self.stage_12(f12)
        r9 = self.stage_9(f9)  
        r6 = self.stage_6(f6)
        r3 = self.stage_3(f3)
        
        return [r3, r6, r9, r12]




