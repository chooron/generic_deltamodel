import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    标准的 Transformer 位置编码。
    `batch_first` 为 True 的版本。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model) for batch_first=True
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
        """
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class VanillaTransformer(nn.Module):
    """
    一个简化的、独立的 Transformer 模型，用于时间序列预测。
    该实现提取了您提供的代码的核心逻辑，并移除了 neuralhydrology 的依赖。
    """

    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward,
                 dropout=0.1, output_dim=1, seq_len=100):
        """
        初始化函数。
        参数:
            input_dim (int): 输入时间序列的特征维度。
            d_model (int): 模型的特征维度。必须能被 nhead 整除。
            nhead (int): 多头注意力机制中的头数。
            num_encoder_layers (int): Transformer编码器的层数。
            dim_feedforward (int): 编码器中前馈网络的维度。
            dropout (float): Dropout 的比例。
            output_dim (int): 输出预测的特征维度。
            pred_len (int): 预测序列的长度。
            seq_len (int): 输入序列的最大长度，用于位置编码。
        """
        super(VanillaTransformer, self).__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        # 1. 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=seq_len)

        # 3. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用 batch_first=True 以简化维度处理
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. 输出层 (Head)
        self.output_dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(d_model, output_dim)

        self.init_weights()

    def init_weights(self):
        # 初始化权重
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成一个上三角矩阵的掩码，用于防止注意力机制关注未来的位置。
        """
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        参数:
            src (Tensor): 输入序列，形状为 (batch_size, seq_len, input_dim)。
        返回:
            Tensor: 预测输出，形状为 (batch_size, pred_len, output_dim)。
        """
        # 1. 嵌入输入并按比例缩放
        # src shape: (batch_size, seq_len, input_dim)
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        # embedded_src shape: (batch_size, seq_len, d_model)

        # 2. 添加位置编码
        pos_encoded_src = self.positional_encoding(embedded_src)
        # pos_encoded_src shape: (batch_size, seq_len, d_model)

        # 3. 生成因果掩码
        mask = self._generate_square_subsequent_mask(src.size(1), src.device)

        # 4. 通过 Transformer 编码器
        encoder_output = self.transformer_encoder(pos_encoded_src, mask)
        # 6. 通过输出层得到最终预测
        prediction = self.output_layer(self.output_dropout(encoder_output))
        # prediction shape: (batch_size, output_dim * pred_len)
        return prediction


if __name__ == '__main__':
    # --- 模型超参数 ---
    input_dim = 10  # 输入特征维度
    d_model = 128  # 模型内部的特征维度 (必须能被 nhead 整除)
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 4  # Transformer编码器的层数
    dim_feedforward = 512  # 前馈网络的隐藏层维度
    output_dim = 1  # 预测目标维度
    pred_len = 1  # 预测未来12个时间步
    seq_len = 60  # 输入序列长度

    # --- 实例化模型 ---
    model = VanillaTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        output_dim=output_dim,
        seq_len=seq_len
    )

    print("模型结构:")
    print(model)

    # --- 创建模拟输入数据 ---
    batch_size = 32

    # 随机生成一批数据 (batch_size, seq_len, input_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    # --- 模型前向传播 ---
    print("\n--- 测试前向传播 ---")
    print(f"输入张量形状: {dummy_input.shape}")

    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        prediction = model(dummy_input)

    print(f"输出张量形状: {prediction.shape}")
    print(f"预期输出形状: ({batch_size}, {output_dim})")

    # 验证输出形状是否正确
    assert prediction.shape == (batch_size, seq_len, output_dim)

    print("\n模型结构和维度检查通过！")
