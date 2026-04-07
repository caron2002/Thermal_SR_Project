import torch
import cv2
from torch.utils.data import DataLoader

# 모델 불러오기
from lib.model import LDASRNet
from lib.dataload import ThermalSRDataset

import glob
import os
from tqdm import tqdm
import numpy as np
import time

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

# Train & Val Data 및 저장 폴더 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = ThermalSRDataset(train_LR_path_list, train_GT_path_list)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

# val data
val_dataset = ThermalSRDataset(val_LR_path_list, val_GT_path_list, training=False)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

# 저장 폴더 경로 설정
save_root = "checkpoints"
lt = time.localtime()
save_dir = os.path.join(save_root, f'checkpoints_{time.strftime("%Y%m%d_%H%M")}')
os.makedirs(save_dir, exist_ok=True)


# 모델 파라미터 및 기본 설정
epochs = 2000

model = LDASRNet(in_channels=1, out_channels=40).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.L1Loss()

# Full Image Training
for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
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

    if (epoch % 10 == 0) or (epoch == epochs - 1):
        torch.save(model.state_dict(), os.path.join(save_dir, f"ldasrnet_epoch_{epoch+1}.pth"))
    


