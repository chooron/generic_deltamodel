import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticFeatureAttention(nn.Module):
    """静态特征 Attention"""

    def __init__(self, n_feat, reduction=4):
        super().__init__()
        hidden = max(1, n_feat // reduction)
        self.fc1 = nn.Linear(n_feat, hidden)
        self.fc2 = nn.Linear(hidden, n_feat)

    def forward(self, x):
        # x: (B, F)
        w = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))  # gating
        return x * w


class DynamicParamHead(nn.Module):
    """LSTM 输出映射到动态参数"""

    def __init__(self, hidden_dim, param_dim, hru_num):
        super().__init__()
        self.out_proj = nn.Linear(hidden_dim, param_dim * hru_num)
        self.param_dim = param_dim
        self.hru_num = hru_num

    def forward(self, h):
        # h: (B, N, H)
        B, N, _ = h.shape
        p = self.out_proj(h)  # (B, N, P*Hru)
        p = p.view(B, N, self.param_dim, self.hru_num)  # (B, N, P, Hru)
        return torch.sigmoid(p)  # 保证物理可用


class HRUAttention(nn.Module):
    """根据动态特征和静态特征输出 HRU 权重"""

    def __init__(self, hidden_dim, hru_num, reduction=2):
        super().__init__()
        self.hru_num = hru_num
        hidden = max(8, hru_num // reduction)
        # TODO 需要修改
        self.fc1 = nn.Linear(hidden_dim + hru_num, hidden)
        self.fc2 = nn.Linear(hidden, hru_num)

    def forward(self, lstm_out, mlp_out):
        """
        h: (B, N, P1, H)   LSTM 输出
        mlp_out: (B, P2, H) 静态特征
        return: (B, N, Hru) 权重
        """
        B, N, P, H = lstm_out.shape
        s = mlp_out.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, P2, Hru)
        feat = torch.cat([lstm_out, s], dim=-1)  # (B, N, P1+P2, Hru)

        w = F.relu(self.fc1(feat))  # (B, N, hidden)
        w = self.fc2(w)  # (B, N, Hru)
        w = F.softmax(w, dim=-1)  # basin 内 HRU 权重归一化
        return w


class DualAttnLstmV2(nn.Module):
    """
    1) 静态特征 Attention + MLP 提取流域静态因子
    2) LSTM 提取时序动态特征
    3) DynamicParamHead 输出动态参数
    4) HRUAttention 输出 HRU 权重
    """

    def __init__(self, static_dim, seq_input_dim,
                 lstm_hidden_dim=16, mlp_hidden_dim=16,
                 param_dim=4, hru_num=16,  # param_dim 表示每个 HRU 的参数数量
                 mlp_dr=0.5, lstm_dr=0.5):
        super().__init__()
        self.hru_num = hru_num
        self.static_att = StaticFeatureAttention(static_dim)

        # 静态特征 MLP
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, hru_num, bias=True),  # 输出 HRU 相关特征
            nn.Sigmoid()
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=seq_input_dim + static_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True,
                            dropout=lstm_dr,
                            bidirectional=False)

        # 两个 Head
        self.param_head = DynamicParamHead(hidden_dim=lstm_hidden_dim,
                                           param_dim=param_dim,
                                           hru_num=hru_num)
        self.hru_attn = HRUAttention(hidden_dim=lstm_hidden_dim,
                                     hru_num=hru_num)

    def forward(self, x_seq, x_static):
        """
        x_seq: (B, T, F) 时序输入
        x_static: (B, F_static) 静态输入
        return:
            dynamic_params: (B, T, Param, Hru)
            hru_weights: (B, T, Hru)
        """
        B, T, _ = x_seq.shape

        # 1) 静态特征 Attention + MLP
        x_static_att = self.static_att(x_static)  # (B, F_static)
        mlp_out = self.mlp(x_static_att).view(B, -1, self.hru_num)  # (B, Param, Hru)

        # 2) 拼接静态特征到时序输入
        x_seq_aug = torch.cat(
            [x_seq, x_static_att.unsqueeze(1).repeat(1, T, 1)],
            dim=-1
        )  # (B, T, seq_dim + static_dim)

        # 3) LSTM
        h, _ = self.lstm(x_seq_aug)  # (B, T, hidden)

        # 4) 动态参数
        dynamic_params = self.param_head(h).view(B, T, -1, self.hru_num)  # (B, T, Param, Hru)

        # 5) HRU 权重
        hru_weights = self.hru_attn(dynamic_params, mlp_out)  # (B, T, Hru)

        return dynamic_params, mlp_out, hru_weights
