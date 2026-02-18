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

# 字体
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 或 'Noto Sans CJK SC'
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

torch.set_default_dtype(torch.float64)  #
temperature = 0.1#nn.Parameter(torch.tensor(1.0))

def compute_metrics(fused_estimate, stc):

        B = stc.size(0)
        # 获取估计的源活动峰值
        fused_channel_abs_mean = fused_estimate.abs().mean(dim=-1)
        fused_abs, _ = torch.max(fused_channel_abs_mean, dim=-1)
        #max_fused_value, max_fused_row_index = torch.max(torch.abs(max_fused_value1), dim=-1)

        # 获取真实源活动峰值
        stc_channel_abs_mean = stc.abs().mean(dim=-1)
        stc_abs, _ = torch.max(stc_channel_abs_mean, dim=-1)
        #max_value, max_stc_row_index = torch.max(torch.abs(max_stc_value1), dim=-1)

        # 使用Softmax生成可导权重
        fused_weights = F.softmax(fused_channel_abs_mean / temperature, dim=-1)
        stc_weights = F.softmax(stc_channel_abs_mean/ temperature, dim=-1)

        # print("权重 fused形状:", fused_weights.shape)
        # print("权重 stc形状:", stc_weights.shape)
        # print("权重 fused均值:", fused_weights.mean().item())
        # print("权重 stc均值:", stc_weights.mean().item())
        #print("溫度:", temperature.item())

        # 3. 提取子集坐标（隔离索引操作，保留坐标张量的梯度）
        with torch.no_grad():
            subset_coords = all_vertices_coords_tensor[vertex_indices]  # (7498, 3)
        subset_coords_expanded = subset_coords.unsqueeze(0).expand(B, -1, -1)  # (B, 7498, 3)

        # 4. 加权平均坐标（可导）
        pos_fused = 1000 * (fused_weights.unsqueeze(-1) * subset_coords_expanded.unsqueeze(1)).sum(dim=-2)
        pos_true = 1000 * (stc_weights.unsqueeze(-1) * subset_coords_expanded.unsqueeze(1)).sum(dim=-2)

        # pos_fused = torch.bmm(fused_weights, subset_coords_expanded).squeeze(1) * 1000  # (B, 3)
        # pos_true = torch.bmm(stc_weights, subset_coords_expanded).squeeze(1) * 1000     # (B, 3)


        # 5. 计算空间偏差
        spatial_deviation = torch.norm(pos_fused - pos_true, p=2, dim=1).mean()

        return spatial_deviation


def Rebuilt(fused_loc):
    grad_rebuilt = G_grad @ fused_loc  # (n_meg, n_times)
    eeg_rebuilt = G_eeg @ fused_loc  # (n_eeg, n_times)

    return grad_rebuilt,eeg_rebuilt

def pearsonr(x, y):
    """
    x, y: Tensor, 相同 shape (C, T)  或 (B, C, T)
    return: 标量 —— 所有通道 |r| 的均值
    """
    # 统一变成 3-D： (B, C, T)  若原来是 2-D 则 B=1

    x = x.squeeze(1)
    y = y.squeeze(1)

    # 1. 去均值
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    # 2. 协方差 & 标准差
    cov = (x * y).sum(dim=-1)                  # (B, C)
    std_x = (x * x).sum(dim=-1).sqrt()
    std_y = (y * y).sum(dim=-1).sqrt()

    r = cov / (std_x * std_y )           # (B, C)

    return r.abs().mean(),r.abs().var()  # 标量

# 自定义数据集类
class NeuroDataset(Dataset):
    def __init__(self, data_dir: str, sampling_rate: float = 512,  # 原始采样率（Hz）
                 total_duration: float = 10.0,  # 样本总时长（秒）
                 num_segments: int = 20,  # 切片片段数
                 ):
        """
        参数注释：
        - data_dir: 数据文件目录路径
        - sampling_rate: 数据采集率（样本点/秒）
        - total_duration: 单个样本的时长（秒）
        - num_segments: 将每个样本分割的片段数量
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.total_duration = total_duration
        self.num_segments = num_segments

        # 计算时间点相关参数
        self.total_points = int(round(sampling_rate * total_duration))  # 总样本点数（5120）
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
        """核心数据加载逻辑"""
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

class DualModalityLSTM(nn.Module):
    def __init__(self,
                   # 输入通道数（CNN输出特征数）
                 output_sources=7498,  # 输出脑源数
                 hidden_size = 1024,# LSTM隐藏层维度
                 num_layers=2,  # LSTM层数
                 dropout=0.1,  # Dropout率
                 bidirectional=False):  # 是否使用双向LSTM
        super(DualModalityLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # LSTM层：处理时序数据，输入格式[seq_len, batch, input_size]
        self.weghts_A = nn.LSTM(
            input_size=256,
            hidden_size=1024,
            num_layers=2,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.weghts_B = nn.LSTM(
            input_size=256,
            hidden_size=1024,
            num_layers=2,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        # 计算LSTM输出维度（双向则翻倍）
        #lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # 线性映射层：将LSTM输出映射到目标脑源数
        self.weghts_A_fc = nn.Linear(1024, 7498)
        self.weghts_B_fc = nn.Linear(1024, 7498)


    def forward(self, weghts_A , weghts_B):
        """
        前向传播
        Args:
            x: 输入张量，形状[B, 300, 300] (batch, channels, time_points)
        Returns:
            output: 输出张量，形状[B, 7498, 300] (batch, sources, time_points)
            :param meg:
            :param eeg:
        """


        # 调整维度：[B, 300c, 300] → [300, B, 300c] (seq_len, batch, feature)
        weghts_A = weghts_A.squeeze(1)
        weghts_A = weghts_A.permute(2, 0, 1)
        weghts_B = weghts_B.squeeze(1)
        weghts_B = weghts_B.permute(2, 0, 1)

        # LSTM前向传播
        weghts_A_lstm_out, (hn, cn) = self.weghts_A(weghts_A)
        weghts_B_lstm_out, (hn, cn) = self.weghts_B(weghts_B)


        # 线性映射：[300, B, hidden_size] → [300, B, 7498]
        weghts_A = self.weghts_A_fc(weghts_A_lstm_out)
        weghts_B = self.weghts_B_fc(weghts_B_lstm_out)


        # 调整维度回原格式：[300, B, 7498] → [B, 1, 7498, 300]
        weghts_A = weghts_A.permute(1, 2, 0).unsqueeze(1)
        weghts_B = weghts_B.permute(1, 2, 0).unsqueeze(1)

        return  weghts_A , weghts_B

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


        return  scores_A, scores_B, weights_A,weights_B #scores_A, scores_B

# 训练函数
def train_and_validate(cnn_model, attention_model,LSTM_model, train_loader, val_loader, optimizer, device, epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    improvement_threshold = 0.1
    early_stopping_patience = 3
    early_stopping_counter = 0
    improvement_counter = 0  # 用于记录连续改进小于阈值的次数

    for epoch in range(epochs):
        # 训练阶段
        cnn_model.train()
        attention_model.train()
        LSTM_model.train()

        epoch_train_loss = 0.0
        epoch_train_loss1 = 0
        epoch_train_loss2 = 0
        epoch_train_loss3 = 0
        epoch_train_loss4 = 0
        epoch_train_loss5 = 0
        epoch_train_loss6 = 0
        epoch_train_loss7 = 0

        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')
        batch_idx = 1
        for batch in train_loader_tqdm:
            optimizer.zero_grad()
            grad = batch['grad'].to(device)
            eeg = batch['eeg'].to(device)
            eeg_std = batch['eeg_std'].to(device)
            meg_std = batch['meg_std'].to(device)
            source = batch['source'].to(device)
            eeg_loc = batch['eeg_loc'].to(device)
            meg_loc = batch['meg_loc'].to(device)

            eeg_feat, meg_feat = cnn_model(eeg_std, meg_std)

            # print("eeg_feat形状:", eeg_feat.shape)
            # print("meg_feat形状:", meg_feat.shape)

            scores_A, scores_B ,_ ,_ = attention_model(
                eeg_feat.permute(0,2,1),
                meg_feat.permute(0,2,1)
            )



            weights_A, weights_B = LSTM_model(scores_A, scores_B)

            print("权重 A均值:", weights_A.mean().item(), "方差:", weights_A.var().item())
            print("权重 B均值:", weights_B.mean().item(), "方差:", weights_B.var().item())

            fused_loc = ((weights_A) * meg_loc) + ((weights_B) * eeg_loc)

            #fused_loc = (meg_loc+eeg_loc)/2

            # print(" eeg_loc均值:", torch.mean(abs(eeg_loc)))
            # print(" meg_loc均值:", torch.mean(abs(meg_loc)))
            # print(" eeg_stc均值:", torch.mean(abs(eeg_stc)))
            # print(" meg_stc均值:", torch.mean(abs(meg_stc)))



            grad_pred = (G_grad @ fused_loc)  # (n_meg, n_times)
            eeg_pred = (G_eeg @ fused_loc)  # (n_eeg, n_times)
            #
            # print("fused_loc 均值:", torch.mean(abs(fused_loc)))
            # print("source 均值:", torch.mean(abs(source*2.5)))
            print("grad_pred 均值:", torch.mean(abs(grad_pred)))
            print("grad 均值:", torch.mean(abs(grad)))
            print("eeg_pred 均值:", torch.mean(abs(eeg_pred)))
            print("eeg 均值:", torch.mean(abs(eeg)))

            r1, _ = pearsonr(grad * 1e12, grad_pred)
            r2, _ = pearsonr(eeg* 1e6, eeg_pred)

            # loss9 = compute_metrics((weights_B) * eeg_loc * 1e10, (weights_B) * fused_loc * 1e10)
            # loss8 = compute_metrics((weights_A ) * meg_loc * 1e10, (weights_B) * fused_loc * 1e10)
            # loss7 = compute_metrics((weights_A) * meg_loc* 1e10, (weights_B) * eeg_loc* 1e10)
            loss6 =  F.mse_loss(eeg_pred , eeg * 1e6)
            loss5 =  F.mse_loss(grad_pred, grad * 1e12)

            # loss2 = F.mse_loss(fused_loc * 1e10, source*1e10)
            loss1 = compute_metrics(fused_loc * 1e10, source * 1e10)

            loss =   loss6 + loss5 #+ loss7 #loss8 + loss9#++
            loss.backward()

            epoch_train_loss1 += loss1.item()
            # epoch_train_loss2 += loss2.item()

            epoch_train_loss3 += r1.item()
            epoch_train_loss4 += r2.item()

            epoch_train_loss5 += loss5.item()
            epoch_train_loss6 += loss6.item()


            epoch_train_loss += loss.item()


            total_grad_norm = 0
            for name, param in cnn_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if grad_norm < 1e-7:
                        print(f"⚠️参数 {name} 梯度消失!")
            print(f"总梯度范数: {total_grad_norm:.4f}")

            optimizer.step()
            # 更新进度条信息loss.item()

            train_loader_tqdm.set_postfix({'deviation loss': f'{epoch_train_loss1 / batch_idx:.4f}',
                                           'MSE MEG ': f'{epoch_train_loss5 / batch_idx:.4f}',
                                           'MSE EEG ': f'{epoch_train_loss6 / batch_idx:.4f}',

                                           'R MEG ': f'{epoch_train_loss3 / batch_idx:.4f}',
                                           'R EEG ': f'{epoch_train_loss4 / batch_idx:.4f}',

                                           # 'Source R': f'{epoch_train_loss7 / batch_idx:.4f}',
                                           'Source MSE ': f'{epoch_train_loss2 / batch_idx:.4f}'
                                           })
            batch_idx += 1
        # 记录损失值
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        cnn_model.eval()
        attention_model.eval()
        LSTM_model.eval()

        epoch_val_loss = 0.0
        epoch_val_loss1 = 0
        epoch_val_loss2 = 0
        epoch_val_loss3 = 0
        epoch_val_loss4 = 0
        epoch_val_loss5 = 0
        epoch_val_loss6 = 0
        epoch_val_loss7 = 0

        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation')
        batch_idx = 1

        with torch.no_grad():

            for batch in val_loader_tqdm:
                grad = batch['grad'].to(device)
                #meg = batch['meg'].to(device)
                eeg = batch['eeg'].to(device)
                eeg_std = batch['eeg_std'].to(device)
                meg_std = batch['meg_std'].to(device)
                source = batch['source'].to(device)
                eeg_loc = batch['eeg_loc'].to(device)
                meg_loc = batch['meg_loc'].to(device)

                eeg_feat, meg_feat = cnn_model(eeg_std, meg_std)

                scores_A, scores_B ,_ ,_ = attention_model(
                    eeg_feat.permute(0, 2, 1),
                    meg_feat.permute(0, 2, 1)
                )

                weights_A, weights_B = LSTM_model(scores_A, scores_B)

                fused_loc = ((weights_A) * meg_loc) + ((weights_B) * eeg_loc)


                grad_pred = (G_grad @ fused_loc)  # (n_meg, n_times)
                eeg_pred = (G_eeg @ fused_loc) # (n_eeg, n_times)

                r1, _ = pearsonr(grad * 1e12, grad_pred)
                r2, _ = pearsonr(eeg* 1e6, eeg_pred)
                #r3, _ = pearsonr(fused_loc * 1e10+zero_add, source*2.5e10+zero_add)

                #val_loss7 = compute_metrics((weights_A+0.5) * meg_loc* 1e10, (weights_B+0.5) * eeg_loc* 1e10)
                val_loss6 = F.mse_loss(eeg_pred, eeg*1e6)
                val_loss5 = F.mse_loss(grad_pred, grad*1e12)

                val_loss2 = F.mse_loss(fused_loc * 1e10, source * 1e10)
                val_loss1 = compute_metrics(fused_loc * 1e10, source * 1e10)
                val_loss = val_loss6 + val_loss5

                epoch_val_loss1 += val_loss1.item()
                # epoch_val_loss2 += val_loss2.item()

                epoch_val_loss3 += r1.item()
                epoch_val_loss4 += r2.item()
                epoch_val_loss5 += val_loss5.item()
                epoch_val_loss6 += val_loss6.item()


                epoch_val_loss += val_loss.item()
                # 更新进度条信息loss.item()

                val_loader_tqdm.set_postfix({'deviation loss': f'{epoch_val_loss1 / batch_idx:.4f}',

                                             'MSE MEG ': f'{epoch_val_loss5 / batch_idx:.4f}',
                                             'MSE EEG ': f'{epoch_val_loss6 / batch_idx:.4f}',

                                             'R MEG': f'{epoch_val_loss3 / batch_idx:.4f}',
                                             'R EEG': f'{epoch_val_loss4 / batch_idx:.4f}',

                                             #'Source R': f'{epoch_val_loss7 / batch_idx:.4f}',
                                             'Source MSE': f'{epoch_val_loss2 / batch_idx:.4f}'
                                              })
                batch_idx += 1
        # 记录损失值
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # 早停机制
        if avg_val_loss < best_val_loss:
            improvement = best_val_loss - avg_val_loss
            if improvement < improvement_threshold:
                improvement_counter += 1
            else:
                improvement_counter = 0
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            improvement_counter = 0

        if early_stopping_counter >= early_stopping_patience or improvement_counter >= 5:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # if epoch_val_loss2/len(val_loader)<900 :
        #     print(f"Early stopping triggered after {epoch + 1} epochs.")
        #     break


        #学习率调整
        # scheduler.step(avg_val_loss)

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Deviation Loss')
    plt.title('LOSS CURVE')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    return train_losses, val_losses

def check_batch_dim(batch):
        print("Batch 数据检查:")
        print(f"EEG 形状: {batch['eeg_std'].shape} (预期: [B,1,60,512])")
        print(f"MEG 形状: {batch['meg_std'].shape} (预期: [B,1,102,512])")
        print(f"Source 形状: {batch['source'].shape} (必须为 [B,1,7498,512])")
        print(f"EEG_Loc 形状: {batch['eeg_loc'].shape} (需与 Source 对齐)")
        print(f"MEG_Loc 形状: {batch['meg_loc'].shape} (需与 Source 对齐)")

        assert batch['source'].shape[2] == 7498, "源点维度丢失!"

if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 10
    EPOCHS = 6
    DATA_DIR = "./check_data"
    # 加载数据集并划分训练集/验证集（1:1）
    full_dataset = NeuroDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    fwd_fname = str(sample.data_path()) + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'  # 替换为你的前向模型文件
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']
    # 将左右半球的顶点坐标拼接在一起
    lh_coords = src[0]['rr'][src[0]['vertno']]
    rh_coords = src[1]['rr'][src[1]['vertno']]
    all_vertices_coords = np.concatenate((lh_coords, rh_coords))

    # 将 NumPy 数组转换为 PyTorch 张量
    all_vertices_coords_tensor = torch.from_numpy(all_vertices_coords).float().to(device)
    all_vertices_coords_tensor.requires_grad_(True)  # 启用梯度

    # 假设输入顶点是全局顶点的前7498个
    vertex_indices = torch.arange(0, 7498, dtype=torch.long)
    vertex_indices = vertex_indices.to(device)  # 确保与模型在同一设备

    # 1. 读 forward
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, verbose=False, force_fixed=True)

    # 2. 构造 info（方便挑通道）
    info = fwd['info']

    # 3. 取增益矩阵 G
    G = fwd['sol']['data']  # (n_ch, n_dipoles*3)  XYZ 方向
    n_dip = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    # 在 fwd['info'] 中找对应索引
    meg_idx = mne.pick_types(fwd['info'], meg=True, eeg=False, exclude=[])
    eeg_idx = mne.pick_types(fwd['info'], meg=False, eeg=True, exclude=[])
    grad_idx = mne.pick_types(fwd['info'], meg='grad', eeg=False, exclude=[])

    G_meg = torch.from_numpy(G[meg_idx, :]).to(device)
    G_eeg = torch.from_numpy(G[eeg_idx, :]*1e6).to(device)
    G_grad = torch.from_numpy(G[grad_idx, :]*1e12).to(device)

    G_meg = G_meg.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    G_eeg = G_eeg.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    G_grad = G_grad.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)


    #G_meg = G_meg.double()

    G_grad = G_grad.double()
    G_eeg = G_eeg.double()



    # 在训练循环前执行一次
    first_batch = next(iter(train_loader))
    check_batch_dim(first_batch)
    # 在数据加载部分添加检查
    print("训练集首样本索引:", train_dataset.indices[:5])
    print("验证集首样本索引:", val_dataset.indices[:5])

    # 打印数据统计量对比
    # print("训练集 EEG 均值:", torch.mean(train_dataset[0]["eeg"]))
    # print("训练集 MEG 均值:", torch.mean(train_dataset[0]["meg"]))
    # print("训练集 EEG 最大值:", torch.max(train_dataset[0]["eeg"]))
    # print("训练集 MEG 最大值:", torch.max(train_dataset[0]["meg"]))

    # 初始化模型
    cnn_model = DualModalityCNN().to(device)
    attention_model = MutualCrossAttention().to(device)
    LSTM_model = DualModalityLSTM().to(device)

    # 优化器
    optimizer = optim.Adam(
        list(cnn_model.parameters()) + list(attention_model.parameters()),
        lr =1e-3 ,
        weight_decay = 5e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    time_start = time.time()

    # 开始训练
    train_and_validate(
        cnn_model,
        attention_model,
        LSTM_model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=EPOCHS
    )

    time_end = time.time()
    print('')
    print('time cost:', time_end - time_start)

# ----------------------
# 6. 模型保存与测试
# ----------------------

# 保存模型
    torch.save({'cnn_model': cnn_model.state_dict(), 'attention_model': attention_model.state_dict()}, './fusionloc.pth')
