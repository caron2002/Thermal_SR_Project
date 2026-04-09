# LDASRNet Model 구현

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 3x3 Convolutional Block
def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

'''
bias 란
Y = W * x + b
bias - 편향값
'''

# 1x1 Convolutional Block
def conv1x1(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

# Shallow Feature Extractor
class ShallowFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShallowFeatureExtractor, self).__init__()
        self.conv3x3 = conv3x3(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv3x3(x)
        return x

# Dynamic Weights Block
class DynamicWeightsBlock(nn.Module):
    def __init__(self, in_channels, gamma=2, beta=1):
        super(DynamicWeightsBlock, self).__init__()
        self.t = int(abs(math.log2(in_channels) / gamma) + (beta / gamma))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=self.t, padding=(self.t - 1)//2, bias=False)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(in_channels, 2)
        self.softmax = nn.Softmax(dim=1) # why dim=1?

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.ReLU(x)
        x = self.fc(x.squeeze(1))
        x = self.softmax(x)
        return x

# Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, k_size=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origin_x = x
        x = self.avg_pool(x) # (batch, channel, 1, 1)
        x = x.squeeze(-1).permute(0, 2, 1) #(batch, 1, channel)
        x = self.conv1d(x) #(batch, 1, channel)
        x = self.sigmoid(x) #(batch, channel, 1, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1) #(batch, channel, 1, 1)
        x = origin_x * x
        return x

# Pixel Attention Block
class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1x1 = conv1x1(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        origin_x = x
        x = self.conv1x1(x) #(batch, channel, height, width)
        x = self.sigmoid(x) #(batch, channel, height, width)
        x = origin_x * x + origin_x
        return x

# Attention Branch
class AttentionBranch(nn.Module):
    def __init__(self, in_channels, k_size=5):
        super(AttentionBranch, self).__init__()
        self.channel_attention = ChannelAttention(k_size)
        self.pixel_attention = PixelAttention(in_channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.pixel_attention(x)
        return x


# Non-attention Branch
class NonAttentionBranch(nn.Module):
    def __init__(self, in_channels):
        super(NonAttentionBranch, self).__init__()
        self.conv = conv3x3(in_channels, in_channels)

    def forward(self, x):
        x = self.conv(x)
        return x

# Dynamic Attention Block
class DynamicAttentionBlock(nn.Module):
    def __init__(self, in_channels, gamma=2, beta=1, kernel_size=5):
        super(DynamicAttentionBlock, self).__init__()
        self.dynamic_weights = DynamicWeightsBlock(in_channels, gamma, beta)
        self.attention_branch = AttentionBranch(in_channels, kernel_size)
        self.non_attention_branch = NonAttentionBranch(in_channels)
        self.conv1x1 = conv1x1(in_channels, in_channels)
    
    def forward(self, x):
        weights = self.dynamic_weights(x) #(batch, 2)
        att_w = weights[:, 0].view(-1, 1, 1, 1)
        non_att_w = weights[:, 1].view(-1, 1, 1, 1)

        # Attention Branch 와 Non-Attention Branch
        x0 = self.conv1x1(x)
        att_x = self.attention_branch(x0) #(batch, channel, height, width)
        non_att_x = self.non_attention_branch(x0) #(batch, channel, height, width)
        
        out = att_w * att_x + non_att_w * non_att_x
        out = self.conv1x1(out)
        out = out + x
        return out

# FRec Block
class FRec_x2(nn.Module):
    def __init__(self, inchannels):
        super(FRec_x2, self).__init__()
        self.NN_inter = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv3x3 = conv3x3(inchannels, inchannels)
        self.pixel_attention = PixelAttention(inchannels)
        
    def forward(self, x):
        x = self.NN_inter(x)
        x = self.conv3x3(x)
        x = self.pixel_attention(x)
        x = self.conv3x3(x)
        return x
    
class FRecBlock_x8(nn.Module):
    def __init__(self, in_channels):
        super(FRecBlock_x8, self).__init__()
        self.frec_x2 = FRec_x2(in_channels)
        self.conv3x3 = conv3x3(in_channels, 1)
    
    def forward(self, x):
        x = self.frec_x2(x)
        x = self.frec_x2(x)
        x = self.frec_x2(x)
        x = self.conv3x3(x)
        return x

# LDASRNet 모델 구축
class LDASRNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,  # Shallow Feature 수
                 gamma=2, 
                 beta=1, 
                 kernel_size=5,
                 num_dab=16):
        super(LDASRNet, self).__init__()
        # Shallow Feature Extractor
        self.Shallow_Feature_Extractor = ShallowFeatureExtractor(in_channels, out_channels)
        
        # Dynamic Attention Block 16개 생성
        self.DABs = nn.Sequential(*[DynamicAttentionBlock(out_channels, gamma, beta, kernel_size) for _ in range(num_dab)])

        # FRecBlcok_x8 생성
        self.FRec_Block_x8 = FRecBlock_x8(out_channels)
        
    def forward(self, x):
        
        origin_x = x
        base = F.interpolate(origin_x, scale_factor=8, mode='bilinear')

        x = self.Shallow_Feature_Extractor(x)
        x = self.DABs(x)
        x = self.FRec_Block_x8(x)

        out = x + base
        return out