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

# 1. 全局默认 double + 设备
torch.set_default_dtype(torch.float64)  # 关键：dtype 不再是 Tensor 类型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




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
    def __init__(self,data_dir: str, sampling_rate: float = 512,  # 原始采样率（Hz）
                 total_duration: float = 1.0,  # 样本总时长（秒）
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

    # ---------- Dataset 接口 ----------
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
        self.eeg_lstm = nn.LSTM(
            input_size=60,
            hidden_size=1024,
            num_layers=num_layers,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

        self.meg_lstm = nn.LSTM(
            input_size=204,
            hidden_size=1024,
            num_layers=4,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )


        self.weghts_A = nn.LSTM(
            input_size=300,
            hidden_size=1024,
            num_layers=2,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.weghts_B = nn.LSTM(
            input_size=300,
            hidden_size=1024,
            num_layers=2,
            batch_first=False,  # 设为False，输入格式为[seq_len, batch, feature]
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 计算LSTM输出维度（双向则翻倍）
        #lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # 线性映射层：将LSTM输出映射到目标脑源数
        # self.eeg_fc = nn.Linear(1024, output_sources)
        # self.meg_fc1 = nn.Linear(2048, 4096)
        # self.meg_fc2 = nn.Linear(4096, output_sources)

        self.weghts_A_fc = nn.Linear(2048, 7498)
        self.weghts_B_fc = nn.Linear(2048, 7498)

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
        # eeg = eeg.squeeze(1)
        # eeg = eeg.permute(2, 0, 1)
        # meg = meg.squeeze(1)
        # meg = meg.permute(2, 0, 1)
        weghts_A = weghts_A.squeeze(1)
        weghts_A = weghts_A.permute(2, 0, 1)
        weghts_B = weghts_B.squeeze(1)
        weghts_B = weghts_B.permute(2, 0, 1)

        # LSTM前向传播
        # eeg_lstm_out, (hn, cn) = self.eeg_lstm(eeg)  # lstm_out: [300, B, hidden_size]
        # meg_lstm_out, (hn, cn) = self.meg_lstm(meg)

        weghts_A_lstm_out, (hn, cn) = self.weghts_A(weghts_A)
        weghts_B_lstm_out, (hn, cn) = self.weghts_B(weghts_B)

        # 线性映射：[300, B, hidden_size] → [300, B, 7498]
        # eeg_out = self.eeg_fc(eeg_lstm_out)
        # meg_out = self.meg_fc(meg_lstm_out)

        weghts_A = self.weghts_A_fc(weghts_A_lstm_out)
        weghts_B = self.weghts_B_fc(weghts_B_lstm_out)

        # 调整维度回原格式：[300, B, 7498] → [B, 1, 7498, 300]
        # eeg_stc = eeg_out.permute(1, 2, 0).unsqueeze(1)
        # meg_stc = meg_out.permute(1, 2, 0).unsqueeze(1)
        weghts_A = weghts_A.permute(1, 2, 0).unsqueeze(1)
        weghts_B = weghts_B.permute(1, 2, 0).unsqueeze(1)

        return weghts_A , weghts_B

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
        # attention_A = F.Tanh(attention_A_4)
        attention_A = self.dropout(attention_A_4)

        scores_B = torch.matmul(x2, x1.transpose(1, 2)) / math.sqrt(d)

        # 线性变化
        attention_B_1 = scores_B.transpose(1, 2)  # [B, 300, 300]
        attention_B_2 = self.attention_B_2(attention_B_1)  # [B, 300, 7498]
        attention_B_3 = attention_B_2.transpose(1, 2)  # [B, 7498,300]
        attention_B_4 = attention_B_3.view(B, 1, 7498, 300)
        # attention_B = F.Tanh(attention_B_4)
        attention_B = self.dropout(attention_B_4)


        weights_A = attention_A
        weights_B = attention_B


        return scores_A, scores_B






# 训练函数
def train_and_validate(cnn_model, attention_model,LSTM_model,train_loader, val_loader, optimizer, device, epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    improvement_threshold = 0.1
    early_stopping_patience = 3
    early_stopping_counter = 0
    improvement_counter = 3  # 用于记录连续改进小于阈值的次数

    for epoch in range(epochs):
        # 训练阶段
        cnn_model.train()
        attention_model.train()
        LSTM_model.train()
        epoch_train_loss = 0.0
        epoch_train_loss1 = 0.0
        epoch_train_loss2 = 0.0
        epoch_train_r1 = 0
        epoch_train_r2 = 0


        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training')
        batch_idx = 1
        for batch in train_loader_tqdm:
            optimizer.zero_grad()
            eeg = batch['eeg'].to(device)
            meg = batch['meg'].to(device)
            grad = batch['grad'].to(device)
            eeg_std = batch['eeg_std'].to(device)
            meg_std= batch['meg_std'].to(device)
            eeg_loc = batch['eeg_loc'].to(device)
            meg_loc = batch['meg_loc'].to(device)

            eeg_feat, meg_feat = cnn_model(eeg_std, meg_std)

            # print("eeg_feat形状:", eeg_feat.shape)
            # print("meg_feat形状:", meg_feat.shape)

            scores_A, scores_B = attention_model(
                eeg_feat.permute(0,2,1),
                meg_feat.permute(0,2,1)
            )



            # print("权重 A均值:", weights_A.mean().item(), "方差:", weights_A.var().item())
            # print("权重 B均值:", weights_B.mean().item(), "方差:", weights_B.var().item())
            weights_A, weights_B = LSTM_model(scores_A, scores_B)

            fused_loc = ((weights_A) * meg_loc) + ((weights_B) * eeg_loc)


            #fused_loc = (eeg_loc +  meg_loc)/2
            # print(" fused_loc均值:", torch.mean(abs(fused_loc)))

            grad_pred = (G_grad @ fused_loc) # (n_meg, n_times)
            eeg_pred = (G_eeg @ fused_loc) # (n_eeg, n_times)

            # print("eeg_loc  均值:", abs(eeg_loc).mean().item(), )
            # print("meg_loc  均值:", abs(meg_loc).mean().item(), )
            #print("fused_loc  均值:", abs(fused_loc).mean().item(), )
            print("MEG rebuilt  均值:",abs(grad_pred).mean().item(),)
            # print("MEG real  均值:",abs(grad).mean().item())
            # print("EEG real  均值:", abs(eeg).mean().item(), )
            print("EEG rebuilt  均值:", abs(eeg_pred).mean().item(), )


            mse1 = F.mse_loss(grad*1e12, grad_pred)
            mse2 = F.mse_loss(eeg*1e6, eeg_pred)

            r1,_ = pearsonr(grad*1e12, grad_pred)
            r2,_ = pearsonr(eeg*1e6, eeg_pred)


            loss = mse1+mse2
            loss.backward()

            total_grad_norm = 0

            # for name, param in LSTM_model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.norm().item())

            # for name, param in cnn_model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         total_grad_norm += grad_norm
            #         if grad_norm < 1e-7:
            #             print(f"⚠️参数 {name} 梯度消失!")
            #print(f"总梯度范数: {total_grad_norm:.4f}")

            # meg_stc.retain_grad()
            # print("∂loss/∂meg_stc:",
            #       meg_stc.grad.abs().mean().item() if meg_stc.grad is not None else None)
            optimizer.step()

            epoch_train_loss += loss.item()

            epoch_train_loss1 += mse1.item()
            epoch_train_loss2 += mse2.item()

            epoch_train_r1 += r1.item()
            epoch_train_r2 += r2.item()


            # 更新进度条信息loss.item()

            train_loader_tqdm.set_postfix({'MSE meg loss': f'{epoch_train_loss1 / batch_idx:.4f}',
                                           'MSE eeg loss': f'{epoch_train_loss2 / batch_idx:.4f}',
                                           'R meg loss': f'{epoch_train_r1 / batch_idx:.4f}',
                                           'R eeg loss': f'{epoch_train_r2 / batch_idx:.4f}'})
            batch_idx += 1
        # 记录损失值
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        cnn_model.eval()
        attention_model.eval()
        LSTM_model.eval()

        epoch_val_loss = 0.0
        epoch_val_loss1 = 0.0
        epoch_val_loss2 = 0.0
        epoch_val_r1 = 0
        epoch_val_r2 = 0
        epoch_val_r1_var = 0
        epoch_val_r2_var = 0
        max_val_r1_var = 0
        max_val_r2_var = 0


        val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation')
        batch_idx = 1

        with torch.no_grad():

            for batch in val_loader_tqdm:
                meg = batch['meg'].to(device)
                grad = batch['grad'].to(device)
                eeg = batch['eeg'].to(device)
                eeg_std = batch['eeg_std'].to(device)
                meg_std = batch['meg_std'].to(device)
                eeg_loc = batch['eeg_loc'].to(device)
                meg_loc = batch['meg_loc'].to(device)

                eeg_feat, meg_feat = cnn_model(eeg_std, meg_std)

                scores_A, scores_B = attention_model(
                    eeg_feat.permute(0, 2, 1),
                    meg_feat.permute(0, 2, 1)
                )

                weights_A, weights_B = LSTM_model(scores_A, scores_B)
                #fused_loc = (eeg_loc + meg_loc) / 2
                #meg_stc, eeg_stc = LSTM_model(meg_std, eeg_std)
                fused_loc = ((weights_A) * meg_loc) + ((weights_B) * eeg_loc)

                grad_pred = G_grad @ fused_loc# (n_meg, n_times)
                eeg_pred = G_eeg @ fused_loc  # (n_eeg, n_times)




                val_mse1 = F.mse_loss(grad*1e12, grad_pred)
                val_mse2 = F.mse_loss(eeg*1e6, eeg_pred)

                val_r1,val_r1_var = pearsonr(grad*1e12, grad_pred)
                val_r2,val_r2_var = pearsonr(eeg*1e6, eeg_pred)

                epoch_val_r1_var += val_r1_var.item()
                epoch_val_r2_var += val_r2_var.item()

                if val_r1_var.item() > max_val_r1_var:
                    max_val_r1_var = val_r1_var

                if val_r2_var.item() > max_val_r2_var:
                    max_val_r2_var = val_r2_var

                # print('r_meg 方差均值、最大值:',val_r1_var/batch_idx,max_val_r1_var)
                # print('r_eeg 方差均值、最大值:', val_r2_var/batch_idx,max_val_r2_var)

                val_loss = val_mse1 + val_mse2  # +r1+r2#loss1+loss2+r1+r2
                epoch_val_loss += val_loss.item()
                epoch_val_loss1 += val_mse1.item()
                epoch_val_loss2 += val_mse2.item()

                epoch_val_r1 += val_r1.item()
                epoch_val_r2 += val_r2.item()
                # 更新进度条信息loss.item()

                val_loader_tqdm.set_postfix({  'MSE meg loss': f'{epoch_val_loss1 / batch_idx:.4f}',
                                               'MSE eeg loss': f'{epoch_val_loss2 / batch_idx:.4f}',
                                               'R meg loss': f'{epoch_val_r1 / batch_idx:.4f}',
                                               'R eeg loss': f'{epoch_val_r2 / batch_idx:.4f}'})
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


        #学习率调整
        # scheduler.step(avg_val_loss)

    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Real_data LOSS CURVE')
    plt.legend()
    plt.savefig('loss_curve_real_data.png')
    plt.show()

    return train_losses, val_losses

def check_batch_dim(batch):
        print("Batch 数据检查:")
        print(f"EEG 形状: {batch['eeg_std'].shape} (预期: [B,1,60,512])")
        print(f"MEG 形状: {batch['meg_std'].shape} (预期: [B,1,204,512])")
        print(f"EEG_Loc 形状: {batch['eeg_loc'].shape} (需与 Source 对齐)")
        print(f"MEG_Loc 形状: {batch['meg_loc'].shape} (需与 Source 对齐)")


if __name__ == "__main__":

    BATCH_SIZE = 10
    EPOCHS = 10
    DATA_DIR = "./Real_data/MNE"
    # 加载数据集并划分训练集/验证集
    full_dataset = NeuroDataset(DATA_DIR)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    fwd_fname = str(sample.data_path()) + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'  # 替换为你的前向模型文件
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    # 1. 读 forward
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, verbose=False, force_fixed=True)

    # 2. 构造 info（方便挑通道）
    info = fwd['info']

    # 3. 取增益矩阵 G
    G = fwd['sol']['data']  # (n_ch, n_dipoles*3)  XYZ 方向
    n_dip = fwd['src'][0]['nuse'] + fwd['src'][1]['nuse']

    # 在 fwd['info'] 中找对应索引
    meg_idx = mne.pick_types(fwd['info'], meg=True, eeg=False,exclude=[])
    grad_idx = mne.pick_types(fwd['info'], meg='grad', eeg=False, exclude=[])
    eeg_idx = mne.pick_types(fwd['info'], meg=False, eeg=True,exclude=[])

    G_meg = torch.from_numpy(G[meg_idx, :]).to(device)
    G_eeg = torch.from_numpy(G[eeg_idx, :]*1e6).to(device)
    G_grad = torch.from_numpy(G[grad_idx, :]*1e12).to(device)

    G_meg = G_meg.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    G_grad = G_grad.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    G_eeg = G_eeg.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, -1, -1, -1)
    G_meg = G_meg.double()
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


    cnn_model = DualModalityCNN()
    attention_model = MutualCrossAttention()
    LSTM_model = DualModalityLSTM()
    # 将模型移动到 GPU
    cnn_model.to(device)
    #cnn_model.double()
    attention_model.to(device)
    #attention_model.double()
    LSTM_model.to(device)
    LSTM_model.double()
    # 优化器
    optimizer = torch.optim.Adam(
        list(cnn_model.parameters()) + list(attention_model.parameters())+list(LSTM_model.parameters()),
        lr = 1e-3,
        weight_decay=1e-4 )

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
    torch.save({'cnn_model': cnn_model.state_dict(),
                    'attention_model': attention_model.state_dict(),
                    'LSTM_model':LSTM_model.state_dict()}, './Real_data/MNE-real-data.pth')
