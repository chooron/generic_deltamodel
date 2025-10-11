import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping


# --- 1. 准备虚拟数据 ---
def create_dummy_csv(filename="timeseries_data.csv"):
    """创建一个包含季节性和趋势的虚拟时间序列CSV文件"""
    if os.path.exists(filename):
        print(f"'{filename}' a'lready exists. Skipping creation.")
        return

    print(f"Creating dummy CSV file: '{filename}'")
    time = np.arange(0, 1000, 1)
    # 输入变量：一个有季节性的信号和一个随机噪声
    input_1 = np.sin(2 * np.pi * time / 365.25) + np.cos(2 * np.pi * time / 30.5)
    input_2 = np.random.randn(len(time)) * 0.2

    # 输出变量：输入的延迟和非线性变换
    output = 0.8 * np.roll(input_1, 10) + input_2 ** 2 + 0.1 * np.random.randn(len(time))

    df = pd.DataFrame({
        'input_1': input_1,
        'input_2': input_2,
        'output': output
    })
    df.to_csv(filename, index=False)


# --- 2. 数据集定义 ---
class TimeSeriesDataset(Dataset):
    """为滑动窗口方法创建数据集"""

    def __init__(self, data, input_cols, target_col, seq_len):
        self.data = data
        self.input_cols = input_cols
        self.target_col = target_col
        self.seq_len = seq_len

        self.X_data = torch.tensor(data[input_cols].values, dtype=torch.float32)
        self.y_data = torch.tensor(data[target_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 注意：t时刻的预测值，可以使用t时刻的输入变量
        # 输入序列：包含输入变量[t-seq_len+1, ..., t]
        x_sequence = self.X_data[idx:idx + self.seq_len]

        # 目标值：t时刻的输出变量
        y_target = self.y_data[idx + self.seq_len - 1]

        return x_sequence, y_target


# --- 3. LightningDataModule ---
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, input_cols, target_col, seq_len=30, batch_size=32, pred_len=90):
        super().__init__()
        self.csv_path = csv_path
        self.input_cols = input_cols
        self.target_col = target_col
        self.all_cols = input_cols + [target_col]
        self.seq_len = seq_len
        self.batch_size = batch_size
        # pred_len 仅用于自回归预测时，定义预热期后的预测长度
        self.pred_len = pred_len
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)

        # 划分数据集
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        self.train_df = df[:train_size]
        self.val_df = df[train_size:train_size + val_size]
        self.test_df = df[train_size + val_size:]

        # 标准化
        self.scaler.fit(self.train_df[self.all_cols])
        self.train_df[self.all_cols] = self.scaler.transform(self.train_df[self.all_cols])
        self.val_df[self.all_cols] = self.scaler.transform(self.val_df[self.all_cols])
        self.test_df[self.all_cols] = self.scaler.transform(self.test_df[self.all_cols])

        if stage == 'fit' or stage is None:
            self.train_dataset = TimeSeriesDataset(self.train_df, self.input_cols, self.target_col, self.seq_len)
            self.val_dataset = TimeSeriesDataset(self.val_df, self.input_cols, self.target_col, self.seq_len)
        if stage == 'test' or stage is None:
            self.test_dataset = TimeSeriesDataset(self.test_df, self.input_cols, self.target_col, self.seq_len)
        if stage == 'predict':
            # 预测时，我们可能需要整个数据集或测试集
            self.predict_dataset = TimeSeriesDataset(self.test_df, self.input_cols, self.target_col, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        # 预测时通常不用batch和shuffle
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False, num_workers=4)


# --- 4. LightningModule ---
class LSTMPredictor(pl.LightningModule):
    def __init__(self, n_features, hidden_size, n_layers, dropout, learning_rate,
                 model_type='sliding_window', warmup_len=365, pred_len=90):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, n_features)
        lstm_out, hidden = self.lstm(x, hidden)
        # 我们只关心最后一个时间点的输出
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.criterion(y_hat.squeeze(), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.model_type == 'sliding_window':
            x, y = batch
            y_hat, _ = self(x)
            loss = self.criterion(y_hat.squeeze(), y)
            self.log('val_loss', loss, prog_bar=True)

        elif self.hparams.model_type == 'autoregressive':
            # 在验证阶段模拟真实的预测场景
            # 注意：这里的batch是从DataLoader来的，我们取第一个样本来演示
            # 实际应用中可能需要更复杂的验证逻辑
            x_full_seq, _ = batch
            x_full_seq = x_full_seq[0]  # (seq_len, n_features)

            # 1. Warm-up
            warmup_input = x_full_seq.unsqueeze(0)  # (1, seq_len, n_features)
            _, hidden = self.lstm(warmup_input)

            # 2. Autoregressive prediction
            # 此处逻辑比较复杂，因为需要模拟输入
            # 简单起见，我们这里只计算一个loss作为演示
            # 实际的验证应该在整个验证集上进行多步预测并计算总loss
            # 这里我们仍用单步loss代替
            y_hat, _ = self.linear(hidden[0].squeeze(0))

            # 获取真实y值（dataloader的y是t时刻的，这里为了匹配，我们简单地用最后一个x的y）
            # 这部分逻辑在实际中需要和数据模块对齐
            # 为了简单，我们还是用滑动窗口的loss来代替
            x, y = batch
            y_hat_val, _ = self(x)
            loss = self.criterion(y_hat_val.squeeze(), y)
            self.log('val_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.hparams.model_type == 'sliding_window':
            # 对于滑动窗口，每次预测都是独立的
            x, _ = batch
            y_hat, _ = self(x)
            return y_hat.squeeze()

        elif self.hparams.model_type == 'autoregressive':
            # 1. Warm-up Phase
            # batch 是一个 (x, y) 元组, x shape: (1, seq_len, n_features)
            warmup_data, _ = batch

            # 使用预热数据初始化隐藏状态
            _, hidden = self.lstm(warmup_data)

            predictions = []
            current_input_features = warmup_data[:, -1, :]  # (1, n_features)

            # 2. Autoregressive Prediction Phase
            for _ in range(self.hparams.pred_len):
                # 将输入reshape为 (1, 1, n_features)
                current_input_seq = current_input_features.unsqueeze(1)

                lstm_out, hidden = self.lstm(current_input_seq, hidden)
                prediction = self.linear(lstm_out[:, -1, :])

                predictions.append(prediction.squeeze().item())

                # 更新下一次的输入
                # 关键：t时刻的预测值y_hat(t) 将作为t+1时刻输入的一部分
                # 同时，t+1时刻的其他观测输入变量也需要提供
                # 在这个虚拟例子中，我们假设未来的输入变量是未知的，所以我们用最后一个观测值或预测值来填充
                # 在真实水文场景中，这里应该填入气象预报值！

                # 获取最新的观测输入（除了目标列）
                # 这里我们用最后一个真实观测的输入变量作为未来所有步的输入
                # 这是一个简化！
                next_input_vars = current_input_features[:, :-1]

                # 组合成新的输入
                current_input_features = torch.cat([next_input_vars, prediction], dim=1)

            return torch.tensor(predictions)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# --- 5. 主脚本 ---
if __name__ == '__main__':
    # --- 通用设置 ---
    CSV_FILE = "timeseries_data.csv"
    INPUT_COLS = ['input_1', 'input_2']
    # 注意: 在自回归模式中, output也是输入的一部分
    INPUT_COLS_AUTOREGRESSIVE = ['input_1', 'input_2', 'output']
    TARGET_COL = 'output'
    SEQ_LEN = 30
    PRED_LEN = 90

    # 创建数据
    create_dummy_csv(CSV_FILE)

    # --- 方法一: 滑动窗口模型 ---
    print("\n" + "=" * 50)
    print("METHOD 1: SLIDING WINDOW")
    print("=" * 50)

    # 数据模块
    dm_sliding = TimeSeriesDataModule(
        csv_path=CSV_FILE,
        input_cols=INPUT_COLS,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        batch_size=64
    )

    # 模型
    model_sliding = LSTMPredictor(
        n_features=len(INPUT_COLS),
        hidden_size=50,
        n_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        model_type='sliding_window'
    )

    # 训练
    trainer_sliding = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        callbacks=[RichProgressBar(), EarlyStopping('val_loss', patience=3)]
    )
    trainer_sliding.fit(model_sliding, datamodule=dm_sliding)

    # 预测
    print("\n--- Predicting with Sliding Window Model ---")
    predictions_sliding = trainer_sliding.predict(model_sliding, datamodule=dm_sliding)
    # predictions_sliding会是一个列表，每个元素是大小为1的tensor
    print(f"Generated {len(predictions_sliding)} one-step predictions.")
    print("First 5 predictions:", [f"{p.item():.4f}" for p in predictions_sliding[:5]])

    # --- 方法二: 自回归模型 ---
    print("\n" + "=" * 50)
    print("METHOD 2: WARM-UP + AUTOREGRESSIVE")
    print("=" * 50)

    # 数据模块 (注意输入列包含了目标列本身)
    dm_autoregressive = TimeSeriesDataModule(
        csv_path=CSV_FILE,
        input_cols=INPUT_COLS_AUTOREGRESSIVE,  # output作为输入之一
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,  # 训练时的seq_len
        batch_size=64,
        pred_len=PRED_LEN
    )

    # 模型
    model_autoregressive = LSTMPredictor(
        n_features=len(INPUT_COLS_AUTOREGRESSIVE),
        hidden_size=50,
        n_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        model_type='autoregressive',
        warmup_len=SEQ_LEN,  # 预热长度和训练seq_len保持一致
        pred_len=PRED_LEN
    )

    # 训练 (训练逻辑和方法一相同，都是Teacher Forcing)
    trainer_autoregressive = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        callbacks=[RichProgressBar(), EarlyStopping('val_loss', patience=3)]
    )
    trainer_autoregressive.fit(model_autoregressive, datamodule=dm_autoregressive)

    # 预测
    print("\n--- Predicting with Autoregressive Model ---")
    # 预测时，它会为dataloader中的每个样本生成一个长序列的预测
    predictions_autoregressive = trainer_autoregressive.predict(model_autoregressive, datamodule=dm_autoregressive)
    # predictions_autoregressive会是一个列表，每个元素是一个长度为 PRED_LEN 的tensor
    print(f"Generated {len(predictions_autoregressive)} sequences of {PRED_LEN}-step predictions.")
    print("First prediction sequence (first 10 values):")
    print([f"{val:.4f}" for val in predictions_autoregressive[0][:10].tolist()])