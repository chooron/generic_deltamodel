import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BahdanauAttention(nn.Module):
    """
    候选注意力层
    """
    def __init__(self, dynamic_dim, static_dim, attention_hidden_dim):
        """
        Args:
            dynamic_dim (int): 动态参数的数量 (P_d)
            static_dim (int): 静态参数的数量 (P_s)
            attention_hidden_dim (int): 注意力机制的内部隐藏维度
        """
        super().__init__()
        # 这个线性层接收 P_d 维的动态特征，输出 attention_hidden_dim 维
        self.W_d = nn.Linear(dynamic_dim, attention_hidden_dim, bias=False)

        # 这个线性层接收 P_s 维的静态特征，同样输出 attention_hidden_dim 维
        self.W_s = nn.Linear(static_dim, attention_hidden_dim, bias=False)

        # 最终的打分器v，接收 attention_hidden_dim 维的融合特征
        self.v = nn.Linear(attention_hidden_dim, 1, bias=False)

    def forward(self, dynamic_features, static_features_expanded):
        # dynamic_features 的 shape: (..., P_d)
        # static_features_expanded 的 shape: (..., P_s)

        # 经过 W_d 变换后，shape 变为 (..., attention_hidden_dim)
        proj_dynamic = self.W_d(dynamic_features)

        # 经过 W_s 变换后，shape 也变为 (..., attention_hidden_dim)
        proj_static = self.W_s(static_features_expanded)

        # --- 现在它们的最后一个维度完全相同，可以安全地相加了！ ---
        combined = torch.tanh(proj_dynamic + proj_static)  # shape: (..., attention_hidden_dim)

        # 通过 v 计算最终分数
        scores = self.v(combined).squeeze(-1)  # shape: (...)

        # 在HRU维度上应用Softmax
        attention_weights = torch.softmax(scores, dim=-1)  # 假设倒数第一个维度是 HRU

        return attention_weights

class DynamicWeightAttention(nn.Module):
    """
    针对 (Batch/Time, Basin, Param, HRU) 维度布局的注意力层。

    此模块首先通过维度重排将数据整理为标准的 (..., 实体, 特征) 格式，
    然后执行特征融合、评分和归一化，以计算每个HRU的动态权重。
    """

    def __init__(self, dynamic_params_dim, static_params_dim, attention_hidden_dim):
        """
        初始化注意力层。
        Args:
            dynamic_params_dim (int): 动态参数的数量 (P_d)
            static_params_dim (int): 静态参数的数量 (P_s)
            attention_hidden_dim (int): 评分网络MLP的隐藏层维度
        """
        super().__init__()
        # MLP的输入维度是融合后的参数数量
        self.combined_dim = dynamic_params_dim + static_params_dim
        self.layer_norm = nn.LayerNorm(self.combined_dim, eps=1e-4)

        # 评分网络
        self.scoring_mlp = nn.Sequential(
            nn.Linear(self.combined_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )

    def forward(self, dynamic_features, static_features):
        """
        前向传播。

        Args:
            dynamic_features: 动态参数, shape: (B_T, N, P_d, H)
            static_features: 静态参数, shape: (N, P_s, H)

        Returns:
            attention_weights: 计算出的注意力权重, shape: (B_T, N, H)
        """
        # 获取维度信息
        B_T, N, H, _ = dynamic_features.shape

        # --- 步骤 1: 构建统一的特征表示 ---
        # a) 广播静态特征
        # (N, H, P_s) -> (1, N, H, P_s) -> (B_T, N, H, P_s)
        static_features_expanded = static_features.unsqueeze(0).expand(B_T, -1, -1, -1)

        # b) 融合特征
        # 拼接后的 shape: (B_T, N, H, P_d + P_s)
        fused_features = self.layer_norm(torch.cat([dynamic_features, static_features_expanded], dim=-1))

        # --- 步骤 2: 计算重要性分数 ---
        # MLP输入 (B_T, N, H, P_d + P_s)，输出 (B_T, N, H, 1)
        scores = self.scoring_mlp(fused_features)

        # 去掉多余的维度 -> (B_T, N, H)
        scores = scores.squeeze(-1)

        # 在HRU维度(最后一个维度)上应用Softmax
        scores_stabilized = scores - torch.max(scores, dim=-1, keepdim=True)[0]

        attention_weights = torch.softmax(scores_stabilized, dim=-1)

        return attention_weights


class DualAttnBiLstmV2(nn.Module):
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
        self.lstm = nn.LSTM(input_size=seq_input_dim + static_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True,
                            dropout=lstm_dr,
                            bidirectional=False)
        self.lstm_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_out_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=True),
            nn.Dropout(mlp_dr),
            nn.Linear(mlp_hidden_dim, mlp_out_dim, bias=True),
            nn.Sigmoid()
        )
        self.feature_att = DynamicWeightAttention(dynamic_params_dim=int(lstm_out_dim / hru_num),
                                                  static_params_dim=int(mlp_out_dim / hru_num),
                                                  attention_hidden_dim=mlp_hidden_dim)
        # self.feature_att = nn.Linear(lstm_hidden_dim, lstm_out_dim)

    def forward(self, x_seq, x_static):
        B, N, _ = x_seq.shape
        # 1) 静态属性 attention
        x_static_att = self.static_att(x_static)  # (B, F)
        mlp_out = self.mlp(x_static_att) # (B, H)
        static_params, route_params = mlp_out[:, :-2].view(N, self.hru_num, -1), mlp_out[:, -2:]

        # 拼接到每个时间步输入
        x_seq_aug = torch.cat([x_seq, x_static_att.unsqueeze(0).repeat(x_seq.size(0), 1, 1)], dim=-1)

        # 2) LSTM
        h, _ = self.lstm(x_seq_aug)  # (B, T, hidden)
        dynamic_params = self.lstm_fc(h).view(B, N, self.hru_num, -1)

        # 3) 时序特征 attention
        att_out = self.feature_att(dynamic_params, static_params)  # (B, T, out_dim)
        return (torch.permute(dynamic_params, (0, 1, 3, 2)),
                torch.permute(static_params, (0, 2, 1)),
                route_params, att_out)