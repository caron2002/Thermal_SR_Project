import torch
import torch.nn as nn
import torch.nn.functional as F

# 모델 불러오기
from lib.model import LDASRNet

# 데이터 증강 함수 불러오기
from lib.transfomer import data_Agumentation

import cv2
import glob
import os
from tqdm import tqdm
import numpy as np

# 데이터 불러오기
root_path = 'Dataset/thermal'

# train data path
train_LR_path = os.path.join(root_path, 'train/LR_x8')
train_GT_path = os.path.join(root_path, 'train/GT')
train_LR_path_list = sorted(glob.glob(os.path.join(train_LR_path, '*.bmp')))
train_GT_path_list = sorted(glob.glob(os.path.join(train_GT_path, '*.bmp')))

# val data path
val_LR_path = os.path.join(root_path, 'val/LR_x8')
val_GT_path = os.path.join(root_path, 'val/GT')
val_LR_path_list = sorted(glob.glob(os.path.join(val_LR_path, '*.bmp')))
val_GT_path_list = sorted(glob.glob(os.path.join(val_GT_path, '*.bmp')))

#=======================================#

# Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 500

model = LDASRNet(in_channels=1, out_channels=40).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.L1Loss()

# Dataset & DataLoader
from torch.utils.data import Dataset, DataLoader
import cv2


class ThermalSRDataset(Dataset):
    def __init__(self, lr_list, hr_list):
        super(ThermalSRDataset, self).__init__()
        self.lr_path_list = lr_list
        self.hr_path_list = hr_list
        self.dataAgument = data_Agumentation()

    def __len__(self):
        return len(self.lr_path_list)
    
    def Agumentation(self, lr_img, hr_img):
        imgs = self.dataAgument(image=lr_img, mask=hr_img)
        lr = imgs['image']
        hr = imgs['mask']
        return lr, hr

    def __getitem__(self, idx):
        lr = cv2.imread(self.lr_path_list[idx], 0)
        hr = cv2.imread(self.hr_path_list[idx], 0)

        lr, hr = self.Agumentation(lr, hr)

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        lr = torch.from_numpy(lr).unsqueeze(0).float()
        hr = torch.from_numpy(hr).unsqueeze(0).float()

        return lr, hr
    

train_dataset = ThermalSRDataset(train_LR_path_list, train_GT_path_list)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

# import matplotlib.pyplot as plt

# lr, hr = train_dataset[0]

# plt.subplot(1, 2, 1)
# plt.imshow(lr.squeeze(0), cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(hr.squeeze(0), cmap='gray')
# plt.show()

import time
save_root = "checkpoints"
lt = time.localtime()
save_dir = os.path.join(save_root, f'checkpoints_{time.strftime("%Y%m%d_%H%M")}')
os.makedirs(save_dir, exist_ok=True)

# Full Image Training
for epoch in range(epochs):
    model.train()
    pbar = pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0.0

    for step, (lr, hr) in enumerate(pbar, 1):
        
        lr = lr.to(device)
        hr = hr.to(device)


        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), avg_loss=epoch_loss / step)

    torch.save(model.state_dict(), os.path.join(save_dir, f"ldasrnet_epoch_{epoch+1}.pth"))
    


