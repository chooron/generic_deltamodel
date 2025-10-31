import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.gamma


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                if i + 1 < embedding_dim:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * i + 1) / embedding_dim))
                    )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_length = x.size(1)
        pe = self.pe[:, :seq_length].to(x.device)
        x = x + pe
        return self.dropout(x)


class MHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, causal=True):
        super().__init__()
        self.dim_head = (dim // heads) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.causal = causal
        self.to_qkv = nn.Linear(dim, _dim * 3, bias=False)
        self.W_out = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head**-0.5

    def set_causal(self, causal):
        self.causal = causal

    def forward(self, x):
        qkv = self.to_qkv(x)
        q, k, v = tuple(
            rearrange(qkv, "b n (d k h) -> k b h n d", k=3, h=self.heads)
        )
        scaled_dot_prod = (
            torch.einsum("b h i d, b h j d -> b h i j", q, k)
            * self.scale_factor
        )
        i, j = scaled_dot_prod.shape[2], scaled_dot_prod.shape[3]
        if self.causal:
            causal_mask = (
                torch.ones(i, j, device=x.device).triu_(j - i + 1).bool()
            )
            scaled_dot_prod = scaled_dot_prod.masked_fill(
                causal_mask, float("-inf")
            )
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.W_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=None,
        causal=False,
        dim_linear_block=512,
        dropout=0.1,
    ):
        super().__init__()
        self.mhsa = MHSelfAttention(dim, heads, dim_head, causal)
        self.drop = nn.Dropout(dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x + self.drop(self.mhsa(x)))
        return self.norm2(y + self.linear(y))


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        dim,
        enc_in,
        num_experts,
        num_layers=3,
        heads=4,
        dim_head=None,
        max_seq_len=512,
        causal=True,
    ):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, heads, dim_head, causal)
                for _ in range(num_layers)
            ]
        )
        self.linear_in = nn.Linear(enc_in * num_experts, dim)
        self.linear_out = nn.Linear(dim, enc_in * num_experts)

    def forward(self, x):
        # x: (B, F, E, S)
        B, Fe, E, S = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, S, Fe * E)
        x = self.linear_in(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear_out(x)
        return x.reshape(B, Fe, E, S)


class MovingAvg2(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2
        )

    def forward(self, x):
        B, Fe, E, S = x.shape
        x = x.view(B * Fe * E, 1, S)
        x = self.avg(x)
        return x.view(B, Fe, E, S)


class MoeLayer(nn.Module):
    """
    Mixture of Experts layer for time-varying gating weight calculation.
    输入: experts_outputs: (Seq, Batch, NumExperts, Feature)
    输出:
        gating_weights: (Seq, Batch, NumExperts, 1)
    """

    def __init__(
        self,
        enc_in,
        num_experts,
        target_points,
        embed_dim=128,
        smoothing_k=5,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.num_experts = num_experts
        self.target_points = target_points

        # gating network = Transformer-based
        self.transformer_gating = SimpleTransformer(
            dim=embed_dim,
            enc_in=enc_in,
            num_experts=num_experts,
            num_layers=2,
            heads=4,
            max_seq_len=target_points,
        )
        self.moving_avg = MovingAvg2(kernel_size=smoothing_k)

    def forward(self, experts_outputs):
        """
        experts_outputs: (Seq, Batch, NumExperts, Feature)
        """
        S, B, E, _ = experts_outputs.shape

        # 转换到 SimpleTransformer 需要的形状
        x = experts_outputs.permute(1, 2, 0, 3)  # (B, E, S, F)
        x = x.permute(0, 3, 1, 2)  # (B, F, E, S)

        # gating network
        gating_logits = self.transformer_gating(x)  # (B, F, E, S)
        gating_smooth = self.moving_avg(gating_logits)
        gating_weights = F.softmax(gating_smooth, dim=2)  # softmax over experts

        # reshape 为 (Seq, Batch, NumExperts, 1) 便于广播
        gating_weights_for_sum = gating_weights.permute(3, 0, 2, 1).mean(
            -1, keepdim=True
        )

        return gating_weights_for_sum


if __name__ == "__main__":
    S, B, E, Fe = 730, 100, 16, 4  # seq_len, batch, experts, feature
    x = torch.randn(S, B, E, Fe)

    moe = MoeLayer(enc_in=Fe, num_experts=E, target_points=S, smoothing_k=11)
    weights = moe(x)

    print(weights.shape)  # -> (Seq, Batch, NumExperts, 1)
