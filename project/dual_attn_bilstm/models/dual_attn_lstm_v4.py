import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.neural_networks.lstm import LstmModel


# --- 1. 静态特征注意力 ---
class StaticFeatureAttention(nn.Module):
    def __init__(self, n_feat, reduction=4):
        super().__init__()
        hidden = max(8, n_feat // reduction)
        self.fc1 = nn.Linear(n_feat, hidden)
        self.fc2 = nn.Linear(hidden, n_feat)

    def forward(self, x):
        # x: (B, F)
        w = F.relu(self.fc1(x))
        w = torch.sigmoid(self.fc2(w))  # feature-wise gating
        return x * w


class StaticGatedAttention(nn.Module):
    """
    静态门控调节注意力层 (Static-Gated Attention)
    -------------------------------------------------
    动态特征负责生成基础得分，静态特征通过门控信号进行调节修正。
    最终通过 Softmax 得到每个 HRU 的注意力权重。
    """

    def __init__(self, dynamic_params_dim, static_params_dim, attention_hidden_dim):
        """
        Args:
            dynamic_params_dim (int): 动态特征维度 (P_d)
            static_params_dim (int): 静态特征维度 (P_s)
            attention_hidden_dim (int): 内部隐藏维度，用于生成分数
        """
        super().__init__()

        # 动态得分生成器（MLP）
        self.dynamic_scorer = nn.Sequential(
            nn.Linear(dynamic_params_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )

        # 静态门控生成器（MLP）
        self.static_gater = nn.Sequential(
            nn.Linear(static_params_dim, attention_hidden_dim),
            nn.ReLU(),
            nn.Linear(attention_hidden_dim, 1)
        )

        # 是否启用温度控制 (可选)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, dynamic_features, static_features_expanded):
        """
        Args:
            dynamic_features: (..., P_d)
            static_features_expanded: (..., P_s)
        Returns:
            attention_weights: (...,)
        """
        # 基础得分：仅依赖动态特征
        base_scores = self.dynamic_scorer(dynamic_features).squeeze(-1)  # (...,)

        # 静态门控信号：仅依赖静态特征
        # 2*sigmoid -> 范围在 (0, 2)，可实现放大或缩小效果
        correction_gate = 2 * torch.sigmoid(self.static_gater(static_features_expanded)).squeeze(-1)  # (...,)

        # 应用门控修正
        corrected_scores = base_scores * correction_gate

        # 温度缩放 softmax (提升可控性)
        attention_weights = F.softmax(corrected_scores / self.temperature, dim=-1)

        return attention_weights


class DualAttnLstmV4(nn.Module):
    """
    在LSTM前后加入注意力机制，
    第一个注意力机制是特征提取的，
    第二个注意力机制是根据动态参数的预测结果（时序特征和统计特征），加上静态参数得到一个注意力权重，然后用于物理水文模型加权输出的
    """

    def __init__(self, static_dim, seq_input_dim,
                 lstm_hidden_dim=16, mlp_hidden_dim=16, lstm_out_dim=16, mlp_out_dim=16,
                 mlp_dr=0.5, lstm_dr=0.5, hru_num=16):
        super().__init__()
        self.hru_num = hru_num
        self.static_att = StaticFeatureAttention(static_dim)
        # self.lstm = nn.LSTM(input_size=seq_input_dim + static_dim,
        #                     hidden_size=lstm_hidden_dim,
        #                     batch_first=True,
        #                     dropout=lstm_dr,
        #                     bidirectional=False)
        # self.lstm_fc = nn.Sequential(
        #     nn.Linear(lstm_hidden_dim, lstm_out_dim),
        #     nn.Sigmoid()
        # )
        self.lstm_model = LstmModel(
            nx=seq_input_dim + static_dim,
            ny=lstm_out_dim,
            hidden_size=lstm_hidden_dim,
            dr=lstm_dr,
        )
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, mlp_out_dim, bias=True),
            nn.Sigmoid()
        )
        self.feature_att = StaticGatedAttention(dynamic_params_dim=int(lstm_out_dim / hru_num),
                                               static_params_dim=int(mlp_out_dim / hru_num),
                                               attention_hidden_dim=mlp_hidden_dim,)
        # self.feature_att = nn.Linear(lstm_hidden_dim, lstm_out_dim)

    def forward(self, x_seq, x_static):
        B, N, _ = x_seq.shape
        # 1) 静态属性 attention
        x_static_att = self.static_att(x_static)  # (B, F)
        mlp_out = self.mlp(x_static_att)  # (B, H)
        static_params, route_params = mlp_out[:, :-2].view(N, self.hru_num, -1), mlp_out[:, -2:]

        # 拼接到每个时间步输入
        x_seq_aug = torch.cat([x_seq, x_static_att.unsqueeze(0).repeat(x_seq.size(0), 1, 1)], dim=-1)

        # 2) LSTM
        # h, _ = self.lstm_model(x_seq_aug)  # (B, T, hidden)
        dynamic_params = self.lstm_model(x_seq_aug).view(B, N, self.hru_num, -1)

        # 3) 时序特征 attention
        att_out = self.feature_att(dynamic_params, static_params)  # (B, T, out_dim)
        return (torch.permute(dynamic_params, (0, 1, 3, 2)),
                torch.permute(static_params, (0, 2, 1)),
                route_params, att_out)
