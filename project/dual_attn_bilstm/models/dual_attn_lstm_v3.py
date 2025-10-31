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


class SelfAttentionWeight(nn.Module):
    def __init__(self, dynamic_params_dim:int, static_params_dim:int, attention_hidden_dim:int, hru_num:int):
        super().__init__()
        self.combined_dim = dynamic_params_dim + static_params_dim
        self.hru_num = hru_num
        self.feature_dim = dynamic_params_dim + static_params_dim  # 即 P_total

        # 自注意力层
        # embed_dim 必须等于 feature_dim
        # num_heads 是超参数，例如 2 或 4
        self.self_attention = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=1, batch_first=True)

        # 用于稳定性的层归一化和前馈网络 (这是Transformer的标准结构)
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 4, self.feature_dim)
        )

        # 最终的评分MLP
        self.scoring_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )

    def forward(self, dynamic_features, static_features):
        B_T, N_basin, H_hru, P_dyn = dynamic_features.shape
        static_features_expanded = static_features.unsqueeze(0).expand(B_T, -1, -1, -1)

        # 1. 特征融合
        fused_features = torch.cat([dynamic_features, static_features_expanded], dim=-1)

        # 2. 变形以适应MultiheadAttention
        # (B_T, N, H, P) -> (B_T * N, H, P)
        # 现在，批次大小是 B_T * N，序列长度是 H
        reshaped_features = fused_features.view(-1, H_hru, self.feature_dim)

        # 3. 应用自注意力机制（Transformer Block的核心部分）
        # 让每个HRU关注流域内其他HRU，得到上下文感知的特征
        attn_output, _ = self.self_attention(reshaped_features, reshaped_features, reshaped_features)

        # 添加残差连接和层归一化
        context_features = self.norm1(reshaped_features + attn_output)

        # 前馈网络
        ffn_output = self.ffn(context_features)

        # 再次添加残差连接和层归一化
        processed_features = self.norm2(context_features + ffn_output)

        # 4. 评分
        scores = self.scoring_mlp(processed_features)  # shape: (B_T * N, H, 1)

        # 5. 变形回原始维度并归一化
        scores = scores.view(B_T, N_basin, H_hru)  # shape: (B_T, N, H)

        scores_stabilized = scores - torch.max(scores, dim=-1, keepdim=True)[0]
        attention_weights = torch.softmax(scores_stabilized, dim=-1)

        return attention_weights


class DualAttnLstmV3(nn.Module):
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
        self.feature_att = SelfAttentionWeight(dynamic_params_dim=int(lstm_out_dim / hru_num),
                                               static_params_dim=int(mlp_out_dim / hru_num),
                                               attention_hidden_dim=mlp_hidden_dim,
                                               hru_num=hru_num)
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
