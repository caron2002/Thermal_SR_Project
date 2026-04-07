from lib.transfomer import data_Agumentation
import torch
from torch.utils.data import Dataset


class ThermalSRDataset(Dataset):
    def __init__(self, lr_list, hr_list, training=True):
        super(ThermalSRDataset, self).__init__()
        self.lr_path_list = lr_list
        self.hr_path_list = hr_list
        self.training = training
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

        if self.training:
            lr, hr = self.Agumentation(lr, hr)

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        lr = torch.from_numpy(lr).unsqueeze(0).float()
        hr = torch.from_numpy(hr).unsqueeze(0).float()

        return lr, hr