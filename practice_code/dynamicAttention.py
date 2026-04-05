import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# 연습 데이터 불러오기
img = cv2.imread('Dataset/thermal/train/LR_x8/003_02_D1_th.bmp', 0) # 이미지 불러오기
img = img.astype(np.float32) / 255.0 # 정규화

# Convolution 연습
img = torch.from_numpy(img)
img = img.unsqueeze(0).unsqueeze(0) # 채널 차원추가
print(img.shape)

# GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img = img.to(device)


def conv1x1(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


class ShallowFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShallowFeatureExtractor, self).__init__()
        self.conv3x3 = conv3x3(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv3x3(x)
        return x


# Channel Attention 연습
class ChannelAttention(nn.Module):
    def __init__(self, k_size=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        print('Working on Channel Attention')
        origin_x = x #(batch, channel, height, width)
        x = self.avg_pool(x) #(batch, channel, 1, 1)
        x = x.squeeze(-1) #(batch, channel, 1)
        x = self.conv1d(x.permute(0, 2, 1)) #(batch, 1, channel)
        x = self.Sigmoid(x).permute(0, 2, 1) #(batch, 1, channel)
        x = x.unsqueeze(-1)
        x = origin_x * x
        return x
    
class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1x1 = conv1x1(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print('Working on Pixel Attention')
        origin_x = x
        x = self.conv1x1(x) #(batch, channel, height, width)
        x = self.sigmoid(x) #(batch, channel, height, width)
        x = origin_x * x
        x = origin_x + x
        return x
    
class AttentionBranch(nn.Module):
    def __init__(self, in_channels, k_size=5):
        super(AttentionBranch, self).__init__()
        self.channel_attention = ChannelAttention(k_size)
        self.pixel_attention = PixelAttention(in_channels)

    def forward(self, x):
        print('Working on Dynamic Attention Block')
        x = self.channel_attention(x)
        x = self.pixel_attention(x)
        return x
    
class DynamicWeightBlock(nn.Module):
    def __init__(self, in_channels, gamma=2, beta=1):
        super(DynamicWeightBlock, self).__init__()
        self.t = int(abs(math.log2(in_channels)/gamma) + (beta/gamma))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.t, padding=(self.t-1)//2, bias=False)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(in_channels, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg_pool(x) #(1, 32, 1, 1)
        x = x.squeeze(-1) # (1, 32, 1)
        print(x.shape)
        x = self.conv1d(x.permute(0, 2, 1)) # input shape: (batch, channel, length)??
        print(x.shape)
        x = self.ReLU(x)
        x = x.squeeze(1) # (1, 32)
        x = self.fc(x) # input shape: (batch, features)
        x = self.softmax(x)
        return x
    
class DynamicAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=5, gamma=2, beta=1):
        super(DynamicAttentionBlock, self).__init__()
        self.ShallowFeatureExtractor = ShallowFeatureExtractor(in_channels, out_channels)
        self.dynamic_weigth_block = DynamicWeightBlock(out_channels, gamma, beta)
        self.AttentionBranch = AttentionBranch(in_channels=out_channels, k_size=k_size)
        self.conv3x3 = conv3x3(out_channels, out_channels)
        self.conv1x1 = conv1x1(out_channels, out_channels)

    def forward(self, x):
        x = self.ShallowFeatureExtractor(x)

        weights = self.dynamic_weigth_block(x) # (batch, 2)
        att_weights = weights[:, 0].view(-1, 1, 1, 1)       # (B, 1, 1, 1)
        non_att_weights = weights[:, 1].view(-1, 1, 1, 1)   # (B, 1, 1, 1)

        x0 = self.conv1x1(x)
        att_x = self.AttentionBranch(x0)
        non_att_x = self.conv3x3(x0)
        out = att_weights * att_x + non_att_weights * non_att_x
        out = self.conv1x1(out)
        out = out + x
        return out
    

# Attention Blcok 출력

model = DynamicAttentionBlock(in_channels=1, out_channels=64, k_size=5, gamma=2, beta=1).to(device)
output = model(img) # (batch, channel, height, width) 형태로 인 풋되어야 함
print(output.shape)


output = output.detach().cpu().numpy()
output = output.squeeze(0)

# 이미지 출력
for i in range(output.shape[0]):
    plt.subplot(8, 8, i+1)
    plt.imshow(output[i], cmap='gray')
    plt.axis('off')
plt.show()