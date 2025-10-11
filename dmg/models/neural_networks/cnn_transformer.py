import torch
import torch.nn as nn
import math


class CausalConvEmbedding(nn.Module):
    """
    自定义的一维因果卷积嵌入层。
    该层包含两个带有残差连接的因果卷积子层，最后连接一个线性层。
    目的是在送入注意力层之前，提取时间维度上的局部关系特征。
    """

    def __init__(self, d_model, kernel_size):
        """
        初始化函数。
        参数:
            d_model (int): 模型的特征维度（即嵌入维度）。
            kernel_size (int): 卷积核的大小。
        """
        super(CausalConvEmbedding, self).__init__()
        # 定义因果卷积的填充量，确保卷积核只能看到“过去”的数据。
        # (kernel_size - 1) 的填充加在序列的左侧。
        self.causal_padding = kernel_size - 1

        # 第一个卷积子层
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            dilation=1
        )
        # 第二个卷积子层
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=1,
            dilation=1
        )
        # 激活函数
        self.relu = nn.ReLU()

        # 最后的线性层
        self.linear = nn.Linear(d_model, d_model)

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 使用 Xavier 初始化来初始化卷积层和线性层的权重
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        前向传播函数。
        参数:
            x (Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)。
        返回:
            Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        # Conv1d 需要的输入形状是 (batch_size, channels, length)，
        # 所以需要对输入 x 进行维度重排。
        x_permuted = x.permute(0, 2, 1)

        # --- 第一个卷积子层 ---
        # 对输入进行左侧填充以实现因果卷积
        x_padded = nn.functional.pad(x_permuted, (self.causal_padding, 0))
        # 经过卷积层和激活函数
        conv1_out = self.relu(self.conv1(x_padded))

        # --- 第二个卷积子层 (用于残差连接) ---
        # 再次进行左侧填充和卷积
        conv1_out_padded = nn.functional.pad(conv1_out, (self.causal_padding, 0))
        conv2_out = self.relu(self.conv2(conv1_out_padded))

        # --- 残差连接 ---
        # 将第一个卷积层的输出与第二个卷积层的输出相加
        residual_out = conv1_out + conv2_out

        # 将维度重排回 (batch_size, seq_len, d_model)
        residual_out_permuted = residual_out.permute(0, 2, 1)

        # --- 最后的线性层 ---
        final_out = self.linear(residual_out_permuted)

        return final_out


class PositionalEncoding(nn.Module):
    """
    标准的 Transformer 位置编码。
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x (Tensor): 输入张量，形状为 (seq_len, batch_size, d_model)。
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    用于时间序列预测的 Transformer 模型，集成了因果卷积嵌入层。
    """

    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward,
                 conv_kernel_size=3, dropout=0.1, output_dim=1):
        """
        初始化函数。
        参数:
            input_dim (int): 输入时间序列的特征维度。
            d_model (int): 模型的特征维度。
            nhead (int): 多头注意力机制中的头数。
            num_encoder_layers (int): Transformer编码器的层数。
            dim_feedforward (int): 编码器中前馈网络的维度。
            conv_kernel_size (int): 因果卷积层的核大小。
            dropout (float): Dropout 的比例。
            output_dim (int): 输出预测的特征维度。
        """
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model

        # 1. 输入嵌入层：将输入特征映射到 d_model 维度
        self.input_embedding = nn.Linear(input_dim, d_model)

        # 2. 自定义的因果卷积嵌入层
        self.conv_embedding = CausalConvEmbedding(d_model, conv_kernel_size)

        # 3. 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 4. Transformer 编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 设置为 True 以匹配 (batch, seq, feature) 输入格式
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )

        # 5. 输出层：将 Transformer 的输出映射到预测维度
        self.output_layer = nn.Linear(d_model, output_dim)

        self.init_weights()

    def init_weights(self):
        # 初始化输入嵌入层和输出层的权重
        nn.init.xavier_uniform_(self.input_embedding.weight)
        nn.init.zeros_(self.input_embedding.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _generate_square_subsequent_mask(self, sz):
        """
        生成一个上三角矩阵的掩码，用于防止注意力机制关注未来的位置。
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        前向传播函数。
        参数:
            src (Tensor): 输入序列，形状为 (batch_size, seq_len, input_dim)。
        返回:
            Tensor: 预测输出，形状为 (batch_size, seq_len, output_dim)。
        """
        # 1. 生成注意力掩码，确保模型是自回归的（只能看到过去）
        device = src.device
        src_mask = self._generate_square_subsequent_mask(src.size(1)).to(device)

        # 2. 将输入通过嵌入层 (batch_size, seq_len, d_model)
        input_embed = self.input_embedding(src) * math.sqrt(self.d_model)

        # 3. 将嵌入结果通过因果卷积层
        # (batch_size, seq_len, d_model)
        conv_embed = self.conv_embedding(input_embed)

        # 4. 将输入嵌入、卷积嵌入和位置嵌入相加
        # 注意：原文描述是 "added to the input embeddings just as in Eq. (14)"
        # 这里我们将卷积输出加到原始嵌入上，然后再加位置编码
        combined_embed = input_embed + conv_embed

        # PyTorch TransformerEncoderLayer 的输入需要是 (seq_len, batch, d_model)
        # 如果 batch_first=False。但我们已设为 True，所以输入是 (batch, seq, feature)
        # PositionalEncoding 模块默认处理 (seq, batch, feature)，我们需要调整一下
        # 为了简单起见，我们直接在 (batch, seq, feature) 上添加位置编码
        # 通常位置编码的实现也需要适配 batch_first
        # 在这个实现中，PositionalEncoding 是 (seq, 1, d_model)，可以通过广播机制添加

        # 适配 PositionalEncoding 的维度
        pos_encoded_embed = combined_embed.permute(1, 0, 2)
        pos_encoded_embed = self.positional_encoding(pos_encoded_embed)
        pos_encoded_embed = pos_encoded_embed.permute(1, 0, 2)

        # 5. 通过 Transformer 编码器
        encoder_output = self.transformer_encoder(pos_encoded_embed, src_mask)

        # 6. 通过输出层得到最终预测
        # (batch_size, seq_len, output_dim)
        output = self.output_layer(encoder_output[:, -1, :])

        return output


if __name__ == '__main__':
    # --- 模型超参数 ---
    input_dim = 10  # 输入特征维度 (例如，10个传感器读数)
    d_model = 128  # 模型内部的特征维度
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 4  # Transformer编码器的层数
    dim_feedforward = 512  # 前馈网络的隐藏层维度
    conv_kernel_size = 5  # 因果卷积的核大小
    output_dim = 1  # 预测目标维度 (例如，预测未来1个值)

    # --- 实例化模型 ---
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        conv_kernel_size=conv_kernel_size,
        output_dim=output_dim
    )

    print("模型结构:")
    print(model)

    # --- 创建模拟输入数据 ---
    batch_size = 32
    seq_len = 60  # 时间序列长度

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
    print(f"预期输出形状: ({batch_size}, {seq_len}, {output_dim})")

    # 验证输出形状是否正确
    assert prediction.shape == (batch_size, seq_len, output_dim)

    print("\n模型结构和维度检查通过！")
