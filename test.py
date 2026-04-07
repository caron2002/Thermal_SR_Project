import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model import LDASRNet

import cv2
import glob
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 데이터 로더
class ThermalsDataset(Dataset):
    def __init__(self, lr_list):
        super(ThermalsDataset, self).__init__()
        self.lr_path_list = lr_list
    
    def __len__(self):
        return len(self.lr_path_list)
    
    def __getitem__(self, idx):
        lr = cv2.imread(self.lr_path_list[idx], 0)
        
        lr = lr.astype(np.float32) / 255.0

        lr = torch.from_numpy(lr).unsqueeze(0).float()

        return lr
    
test_path = 'Dataset/thermal/test/sisr_x8/LR_x8'
test_path_list = sorted(glob.glob(os.path.join(test_path, '*.bmp')))

test_dataset = ThermalsDataset(test_path_list)
test_loader = DataLoader(test_dataset, batch_size=1)

# Test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LDASRNet(in_channels=1, out_channels=40).to(device)

state_dict = torch.load('checkpoints/checkpoints_20260406_2017/ldasrnet_epoch_500.pth', map_location=device)
# 가중치 load 해주기
model.load_state_dict(state_dict)

pbar = tqdm(zip(test_loader, test_path_list), total=len(test_path_list))
for lr, name in pbar:
    with torch.no_grad():
        lr = lr.to(device)
        sr = model(lr)

    sr = sr.squeeze().cpu().clamp(0, 1).numpy()
    sr = (sr * 255.0).round().astype(np.uint8)

    base_name = os.path.splitext(os.path.basename(name))[0]
    save_path = os.path.join("outputs", f"{base_name}.bmp")

    ok = cv2.imwrite(save_path, sr)


    




