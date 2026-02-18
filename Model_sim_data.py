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
    def __init__(self, data_dir: str, sampling_rate: float = 512,
                 total_duration: float = 10.0,
                 num_segments: int = 20,
                 ):

        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.total_duration = total_duration
        self.num_segments = num_segments

        # 计算时间点相关参数
        self.total_points = int(round(sampling_rate * total_duration))
        self.points_per_segment, self.remainder = divmod(self.total_points, num_segments)

        # 自动扫描有效样本文件
        self.sample_ids = self._scan_sample_ids()
        print(f"成功加载 {len(self.sample_ids)} 个有效样本")

    def _scan_sample_ids(self):
        """验证文件完整性并获取有效样本 ID 列表"""
        valid_ids = []
        # 扫描所有 EEG 数据文件
        for eeg_file in glob.glob(os.path.join(self.data_dir, "stc_data_*.npy")):
            # 提取样本 ID
            base_name = os.path.basename(eeg_file)
            try:
                code = base_name.split("_")[2].split(".")[0]
                if len(code) != 4:
                    continue
            except IndexError:
                continue

            # 检查配套文件是否存在
            required_files = [
                #f"eeg_data_{code}.npy",
                #f"grad_data_{code}.npy",
                f"eeg_std_data_{code}.npy",
                f"meg_std_data_{code}.npy",
                #f"stc_data_{code}.npy",
                f"eeg_estimate_{code}.npy",
                f"meg_estimate_{code}.npy"
            ]
            if all(os.path.exists(os.path.join(self.data_dir, f)) for f in required_files):
                valid_ids.append(code)

        # 按自然顺序排序
        valid_ids.sort(key=lambda x: int(x))
        return valid_ids

    def __len__(self):
        """总数据量为样本数 * 时间窗口数"""
        return len(self.sample_ids) * self.num_segments

    def _load_npy(self, prefix, code):
        """安全加载数据文件并进行维度验证"""
        path = os.path.join(self.data_dir, f"{prefix}_{code}.npy")
        data = np.load(path)

        # 维度验证
        if data.shape[-1] != self.total_points:
            raise ValueError(
                f"文件{os.path.basename(path)}的时间点数不匹配！"
                f"期望{self.total_points}点，实际{data.shape[-1]}点"
            )

        # 转换为 Tensor 并添加通道维度 （1xH×W 格式）
        return torch.tensor(data, dtype=torch.float64).unsqueeze(0) # [1, H, W]

    def __getitem__(self, idx):

        # 计算样本索引和时间片索引
        sample_idx = idx // self.num_segments
        segment_idx = idx % self.num_segments

        # 获取样本 ID
        sample_code = self.sample_ids[sample_idx]

        # 动态加载所有模态数据
        grad_data = self._load_npy("grad_data", sample_code)
        #meg_data = self._load_npy("meg_data", sample_code)
        eeg_data = self._load_npy("eeg_data", sample_code)
        eeg_std_data = self._load_npy("eeg_std_data", sample_code)
        meg_std_data = self._load_npy("meg_std_data", sample_code)  # [1, 102, 5120]
        source = self._load_npy("stc_data", sample_code)  # [1, 7498, 5120]
        eeg_loc = self._load_npy("eeg_estimate", sample_code)  # [1, 7498, 5120]
        meg_loc = self._load_npy("meg_estimate", sample_code)  # [1, 7498, 5120]


        # 计算时间切片范围（处理余数）
        start_point = segment_idx * self.points_per_segment
        end_point = start_point + self.points_per_segment

        # 最后一个切片包含余数
        if segment_idx == self.num_segments - 1 and self.remainder != 0:
            end_point += self.remainder

        # 提取时间片段
        sample = {
            "grad": grad_data[:, :, start_point:end_point],  # (通道, 传感器数, 时间片段)
            #"meg": meg_data[:, :, start_point:end_point],  # (通道, 传感器数, 时间片段)
            "eeg": eeg_data[:, :, start_point:end_point],
            "eeg_std": eeg_std_data[:, :, start_point:end_point],
            "meg_std": meg_std_data[:, :, start_point:end_point],  # (通道, 传感器数, 时间片段)
            "source": source[:, :, start_point:end_point],  # (通道, 源点数, 时间片段)
            "eeg_loc": eeg_loc[:, :, start_point:end_point],  # (通道, 源点数, 时间片段)
            "meg_loc": meg_loc[:, :, start_point:end_point]  # (通道, 源点数, 时间片段)
        }

        # 关键检查点 (单个样本维度正确性)
        assert sample["eeg_std"].shape == (
        1, 60, self.points_per_segment + (self.remainder if segment_idx == self.num_segments - 1 else 0)), \
            f"EEG 切片错误! 当前形状: {sample['eeg_std'].shape}, 预期(1,60,{self.points_per_segment + (self.remainder if segment_idx == self.num_segments - 1 else 0)})"

        assert sample["source"].shape == (
        1, 7498, self.points_per_segment + (self.remainder if segment_idx == self.num_segments - 1 else 0)), \
            f"Source 切片必须保留源点维度! 当前形状: {sample['source'].shape}"

        return sample


class DualModalityCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super(DualModalityCNN, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # EEG处理分支（输入形状：[batch, 1, 60, 256]）
        self.eeg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [32, 30, 128]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [64, 15, 64]
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [128, 7, 32]
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # [256, 3, 16]
        )

        # MEG处理分支（输入形状：[batch, 1, 204, 256]）
        self.meg_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),  # [32, 102, 128]
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),  # [64, 51, 64]
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2),  # [B, 128, 25, 32]
            nn.Conv2d(128, 256, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d((2, 2), stride=2)  # [B,256, 12, 16]
        )

        # 特征投影层
        self.hidden_dim = 256
        self.linear_eeg = nn.Linear(48, self.hidden_dim)
        self.linear_meg = nn.Linear(192, self.hidden_dim)


    def forward(self, eeg, meg):
        # EEG特征提取
        eeg_feat = self.eeg_conv(eeg)  # [20, 256, 3 , 32],[B,C,H,W]
        eeg_feat = eeg_feat.view(eeg_feat.size(0), eeg_feat.size(1), -1)  # [20, 256 , 3*16]
        eeg_feat = self.linear_eeg(eeg_feat)  # [B, 256, 256]
        eeg_feat = self.dropout(eeg_feat)
        # MEG特征提取
        meg_feat = self.meg_conv(meg)  # [20, 256, 6, 32]
        meg_feat = meg_feat.view(meg_feat.size(0), meg_feat.size(1), -1)  # [20, 256 , 12*16]
        meg_feat = self.linear_meg(meg_feat)  # [B, 256, 256]
        meg_feat = self.dropout(meg_feat)

        return eeg_feat, meg_feat


# 互交叉注意力模块
class MutualCrossAttention(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 定义线性层
        #self.scores_A = nn.Linear(128, 512)
        #self.scores_B = nn.Linear(128, 512)
        self.attention_A_2 = nn.Linear(256, 7498)
        self.attention_B_2 = nn.Linear(256, 7498)

    def forward(self, x1, x2):

        d = x1.size(-1)
        B = x1.size(0)
        scores_A = torch.matmul(x1, x2.transpose(1, 2)) / math.sqrt(d)

        # 线性变化

        attention_A_1 = scores_A.transpose(1, 2)  # [B, 512meg, 512eeg]
        attention_A_2 = self.attention_A_2(attention_A_1)  # [B, 512meg, 7498eeg]
        attention_A_3 = attention_A_2.transpose(1, 2)  # [B, 7498eeg,512meg]
        attention_A_4 = attention_A_3.view(B, 1, 7498, 256)
        attention_A = F.tanh(attention_A_4)
        attention_A = self.dropout(attention_A)

        scores_B = torch.matmul(x2, x1.transpose(1, 2)) / math.sqrt(d)

        # 线性变化

        attention_B_1 = scores_B.transpose(1, 2)  # [B, 512, 512]
        attention_B_2 = self.attention_B_2(attention_B_1)  # [B, 512, 7498]
        attention_B_3 = attention_B_2.transpose(1, 2)  # [B, 7498,512]
        attention_B_4 = attention_B_3.view(B, 1, 7498, 256)
        attention_B = F.tanh(attention_B_4)
        attention_B = self.dropout(attention_B)



        weights_A = attention_A
        weights_B = attention_B


        return  scores_A, scores_B, weights_A,weights_B

