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

class GatedSpatioTemporalAttention(nn.Module):
    """
    Gated Spatio-Temporal Attention
    --------------------------------
    在时间维 (T) 和 HRU维 (H) 上分别计算注意力，
    最后通过门控乘性融合，实现动态-静态交互修正。
    """
    def __init__(self, dynamic_params_dim, static_params_dim, attention_hidden_dim, num_heads=4):
        super().__init__()

        # 时间维注意力（LSTM输出的动态序列上）
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dynamic_params_dim,
            num_heads=num_heads,
            batch_first=False  # 因为输入是 (T, N, D)
        )

        # 空间维注意力（HRU之间）
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dynamic_params_dim + static_params_dim,
            num_heads=num_heads,
            batch_first=True  # 这里我们会用 (N, H, D)
        )

        # 门控修正层：用空间上下文调节时间动态
        self.gate_proj = nn.Linear(dynamic_params_dim + static_params_dim, dynamic_params_dim)

    def forward(self, dynamic_seq, static_features):
        """
        Args:
            dynamic_seq: (T, N, H, P_d)
            static_features: (N, H, P_s)
        Returns:
            fused_features: (T, N, H)
        """
        T, N, H, P_d = dynamic_seq.shape
        P_s = static_features.shape[-1]

        # === Step 1. 时间维注意力（针对每个HRU独立执行）===
        # 先重排维度，把 (T, N, H, P_d) -> (T, N*H, P_d)
        dyn_reshaped = dynamic_seq.reshape(T, N * H, P_d)
        temp_out, _ = self.temporal_attn(dyn_reshaped, dyn_reshaped, dyn_reshaped)
        # 时间维上下文取平均作为时间聚合特征
        temp_context = temp_out.mean(dim=0).reshape(N, H, P_d)  # (N, H, P_d)

        # === Step 2. 空间维注意力（跨HRU，融合静态信息）===
        spatial_input = torch.cat([temp_context, static_features], dim=-1)  # (N, H, P_d+P_s)
        spatial_out, _ = self.spatial_attn(spatial_input, spatial_input, spatial_input)
        spatial_context = spatial_out.mean(dim=1, keepdim=True)  # (N, 1, P_d+P_s)

        # === Step 3. 门控乘性修正 ===
        gate = torch.tanh(self.gate_proj(spatial_context))  # (N, 1, P_d)
        gated_dynamic = temp_context * (1 + gate)  # (N, H, P_d)

        # === Step 4. 输出权重（HRU聚合注意力）===
        scores = gated_dynamic.mean(dim=-1)  # (N, H)
        attn_weights = F.softmax(scores, dim=-1)  # (N, H)
        fused_features = attn_weights.unsqueeze(0).expand(T, N, H)  # (T, N, H)

        return fused_features


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
        self.feature_att = GatedSpatioTemporalAttention(dynamic_params_dim=int(lstm_out_dim / hru_num),
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
