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

#=======================================#

# PSNR & SSIM
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.image import structural_similarity_index_measure


#=======================================#

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#=======================================#

# Train & Val Data 및 저장 폴더 설정
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
epochs = 10

model = LDASRNet(in_channels=1, out_channels=40).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.L1Loss()

max_score = 0.0

#=======================================#

# Tensorboard

from torch.utils.tensorboard import SummaryWriter
log_dir = os.path.join("runs", f"ldasrnet_{time.strftime('%Y%m%d_%H%M')}")
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

#=======================================#

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

    
    

    # validation
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    epoch_val_loss = 0.0
    val_count = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)
            sr = torch.clamp(sr, 0.0, 1.0)
            hr = torch.clamp(hr, 0.0, 1.0)

            psnr = peak_signal_noise_ratio(sr, hr, data_range=1.0)
            ssim = structural_similarity_index_measure(sr, hr, data_range=1.0)
            loss = criterion(sr, hr)

            bs = lr.size(0) # batch_size 불러오기
            val_psnr += psnr.item() * bs
            val_ssim += ssim.item() * bs
            val_count += bs

            epoch_val_loss += loss.item()
        
        val_psnr /= val_count
        val_ssim /= val_count

        print(f"Epoch [{epoch+1}/{epochs}] "
          f"train_loss={epoch_loss/len(train_loader):.6f} "
          f"val_psnr={val_psnr:.4f} "
          f"val_ssim={val_ssim:.4f}")
        
        # Tensorboard 에 저장
        train_loss = epoch_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("PSNR/val", val_psnr, epoch + 1)
        writer.add_scalar("SSIM/val", val_ssim, epoch + 1)
        

        if ((epoch+1) % 10 == 0) or (epoch == epochs - 1):
            torch.save(model.state_dict(), os.path.join(save_dir, f"ldasrnet_epoch_{epoch + 1}.pth"))
        
        score = val_psnr + 10 * val_ssim
        
        if max_score < score:
            max_score = score
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))

        

            


