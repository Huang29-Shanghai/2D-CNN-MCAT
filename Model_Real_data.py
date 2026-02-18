import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import glob
import mne
from mne.datasets import sample
import time






class NeuroDataset(Dataset):
    def __init__(self,data_dir: str, sampling_rate: float = 512,  #
                 total_duration: float = 1.0,  #
                 ):
        self.data_dir = data_dir
        self.sample_ids = self._scan_sample_ids()
        print(f"成功加载 {len(self.sample_ids)} 个有效样本")

    # ---------- 内部工具 ----------
    def _scan_sample_ids(self):
        valid_ids = []
        for eeg_file in glob.glob(os.path.join(self.data_dir, "eeg_std_data_*.npy")):
            code = os.path.basename(eeg_file).replace(".npy", "").split("_")[-1]
            required = [
                f"eeg_data_{code}.npy",
                f"meg_data_{code}.npy",
                f"grad_data_{code}.npy",
                f"meg_std_data_{code}.npy",
                f"eeg_estimate_{code}.npy",
                f"meg_estimate_{code}.npy"
            ]
            missing = [f for f in required
                       if not os.path.exists(os.path.join(self.data_dir, f))]
            if not missing:
                valid_ids.append(code)
            else:
                print(f"code={code} 缺失 {missing}")
        valid_ids.sort(key=int)
        return valid_ids

    def _load(self, prefix, code):
        path = os.path.join(self.data_dir, f"{prefix}_{code}.npy")
        data = np.load(path)
        if data.shape[-1] != 300:
            raise ValueError(f"{path} 时间点不是 512")
        return torch.tensor(data, dtype=torch.float64).unsqueeze(0)   # [1, n_channels, 512]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        code = self.sample_ids[idx]
        return {
            "meg": self._load("meg_data", code),   # [1, 102,  300]
            "grad": self._load("grad_data", code),  # [1, 102,  300]
            "eeg": self._load("eeg_data", code),  # [1, 60,  300]
            "eeg_std": self._load("eeg_std_data", code),  # [1, 60,   300]
            "meg_std": self._load("meg_std_data", code),  # [1, 102,  300]
            "eeg_loc" : self._load("eeg_estimate",  code),   # [1, 7498, 300]
            "meg_loc" : self._load("meg_estimate",  code),   # [1, 7498, 300]
        }









class DualModalityCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super(DualModalityCNN, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # EEG处理分支（输入形状：[batch, 1, 60, 300]）
        self.eeg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [32, 30, 150]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [64, 15, 75]
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [128, 7, 37]
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [256, 3, 18]
        )

        # MEG处理分支（输入形状：[batch, 1, 204, 300]）
        self.meg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),  # [32, 102, 150]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2), # [64, 51, 75]
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2), # [B, 128, 25, 37]
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2)  # [B,256, 12, 18]
        )

        # 特征投影层
        self.hidden_dim = 300
        self.linear_eeg = nn.Linear(54 , self.hidden_dim)
        self.linear_meg = nn.Linear(216 , self.hidden_dim)


    def forward(self, eeg, meg):
        # EEG特征提取
        eeg_feat = self.eeg_conv(eeg)  # [20, 128, 1 , 128],[B,C,H,W]
        eeg_feat = eeg_feat.view(eeg_feat.size(0), eeg_feat.size(1), -1)  #  [20, 300 , 3*18]
        eeg_feat = self.linear_eeg(eeg_feat) # [B, 256, 300]
        eeg_feat = self.dropout(eeg_feat)
        # MEG特征提取
        meg_feat = self.meg_conv(meg)  # [20, 128, 12, 18]
        meg_feat = meg_feat.view(meg_feat.size(0), meg_feat.size(1), -1)  #  [20, 300 , 6*18]
        meg_feat = self.linear_meg(meg_feat) # [B, 256, 300]
        meg_feat = self.dropout(meg_feat)

        return eeg_feat, meg_feat




# 互交叉注意力模块
class MutualCrossAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 定义线性层

        self.attention_A_2 = nn.Linear(300, 7498)
        self.attention_B_2 = nn.Linear(300, 7498)


    def forward(self, x1, x2):

        d = x1.size(-1)
        B = x1.size(0)
        scores_A = torch.matmul(x1, x2.transpose(1, 2)) / math.sqrt(d)

        # 线性变化
        attention_A_1 = scores_A.transpose(1, 2)  # [B, 300meg, 300eeg]
        attention_A_2 = self.attention_A_2(attention_A_1)  # [B, 300meg, 7498eeg]
        attention_A_3 = attention_A_2.transpose(1, 2)  # [B, 7498eeg,300meg]
        attention_A_4 = attention_A_3.view(B, 1, 7498, 300)
        attention_A = F.Tanh(attention_A_4)
        attention_A = self.dropout(attention_A)

        scores_B = torch.matmul(x2, x1.transpose(1, 2)) / math.sqrt(d)

        # 线性变化
        attention_B_1 = scores_B.transpose(1, 2)  # [B, 300, 300]
        attention_B_2 = self.attention_B_2(attention_B_1)  # [B, 300, 7498]
        attention_B_3 = attention_B_2.transpose(1, 2)  # [B, 7498,300]
        attention_B_4 = attention_B_3.view(B, 1, 7498, 300)
        attention_B = F.Tanh(attention_B_4)
        attention_B = self.dropout(attention_B)


        weights_A = attention_A
        weights_B = attention_B


        return weights_A, weights_B,scores_A, scores_B







