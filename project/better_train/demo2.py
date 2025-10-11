import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping


# --- 1. 准备虚拟数据 (与之前相同) ---
def create_dummy_csv(filename="timeseries_data_long.csv"):
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Skipping creation.")
        return
    print(f"Creating dummy CSV file: '{filename}'")
    time = np.arange(0, 5000, 1)  # 生成更长的数据以适应分块
    input_1 = np.sin(2 * np.pi * time / 365.25) + np.cos(2 * np.pi * time / 30.5)
    input_2 = np.random.randn(len(time)) * 0.2
    output = 0.8 * np.roll(input_1, 10) + input_2 ** 2 + 0.1 * np.random.randn(len(time))
    df = pd.DataFrame({'input_1': input_1, 'input_2': input_2, 'output': output})
    df.to_csv(filename, index=False)


# --- 2. 全新的数据集定义 (核心改动) ---
class WarmupForecastDataset(Dataset):
    """
    根据预热期+预测期创建不重叠的数据块。
    每个样本包含完整的(预热+预测)输入 和 仅(预测期)的目标。
    """

    def __init__(self, data, input_cols, target_col, warmup_len, forecast_len):
        self.data = data
        self.input_cols = input_cols
        self.target_col = target_col
        self.warmup_len = warmup_len
        self.forecast_len = forecast_len
        self.chunk_len = warmup_len + forecast_len

        self.X_data = torch.tensor(data[input_cols].values, dtype=torch.float32)
        self.y_data = torch.tensor(data[target_col].values, dtype=torch.float32)

    def __len__(self):
        # 计算可以切分出多少个完整的、不重叠的块
        return (len(self.data) - self.chunk_len) // self.chunk_len

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_len
        end_idx = start_idx + self.chunk_len

        # 输入X是整个数据块
        x_chunk = self.X_data[start_idx:end_idx]

        # 目标Y只是数据块中的预测期部分
        y_forecast = self.y_data[start_idx + self.warmup_len: end_idx]

        # y_forecast需要reshape成(forecast_len, 1)以匹配模型输出
        return x_chunk, y_forecast.unsqueeze(-1)


# --- 3. LightningDataModule (适配新的Dataset) ---
class WarmupForecastDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, input_cols, target_col, warmup_len, forecast_len, batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.input_cols = input_cols
        self.target_col = target_col
        self.warmup_len = warmup_len
        self.forecast_len = forecast_len
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_path)
        all_cols = self.input_cols + ([self.target_col] if self.target_col not in self.input_cols else [])

        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        self.train_df = df[:train_size]
        self.val_df = df[train_size:train_size + val_size]
        self.test_df = df[train_size + val_size:]

        self.scaler.fit(self.train_df[all_cols])
        self.train_df[all_cols] = self.scaler.transform(self.train_df[all_cols])
        self.val_df[all_cols] = self.scaler.transform(self.val_df[all_cols])
        self.test_df[all_cols] = self.scaler.transform(self.test_df[all_cols])

        if stage == 'fit' or stage is None:
            self.train_dataset = WarmupForecastDataset(self.train_df, self.input_cols, self.target_col, self.warmup_len,
                                                       self.forecast_len)
            self.val_dataset = WarmupForecastDataset(self.val_df, self.input_cols, self.target_col, self.warmup_len,
                                                     self.forecast_len)
        if stage == 'predict':
            # For prediction, we might just take one chunk from the test set
            self.predict_dataset = WarmupForecastDataset(self.test_df, self.input_cols, self.target_col,
                                                         self.warmup_len, self.forecast_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False, num_workers=4)


# --- 4. LightningModule (核心改动) ---
class LSTMForecastModel(pl.LightningModule):
    def __init__(self, n_features, hidden_size, n_layers, dropout, learning_rate,
                 warmup_len, forecast_len):
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

    def forward(self, x):
        """
        x shape: (batch, warmup_len + forecast_len, n_features)
        """
        # LSTM处理整个序列
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, warmup_len + forecast_len, hidden_size)

        # 关键：只选择预测期的输出进行线性变换
        forecast_period_out = lstm_out[:, self.hparams.warmup_len:, :]
        # forecast_period_out shape: (batch, forecast_len, hidden_size)

        # 应用线性层
        prediction = self.linear(forecast_period_out)
        # prediction shape: (batch, forecast_len, 1)
        return prediction

    def _common_step(self, batch, batch_idx):
        x, y = batch
        # x: (batch, chunk_len, features), y: (batch, forecast_len, 1)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 在预测时，我们仍然可以采用你最初的灵感：
        # 使用预热期数据，然后进行自回归生成
        # 这可以用来检验模型在真实预测场景下的泛化能力
        x_chunk, _ = batch  # x_chunk: (1, chunk_len, features)

        warmup_data = x_chunk[:, :self.hparams.warmup_len, :]

        # 1. Warm-up Phase
        _, hidden = self.lstm(warmup_data)

        # 2. Autoregressive Prediction Phase
        predictions = []
        # 获取预热期最后一个时间点的输入，作为自回归的起点
        current_input_features = warmup_data[:, -1, :]

        for _ in range(self.hparams.forecast_len):
            current_input_seq = current_input_features.unsqueeze(1)

            lstm_out, hidden = self.lstm(current_input_seq, hidden)
            prediction = self.linear(lstm_out[:, -1, :])

            predictions.append(prediction.squeeze().item())

            # 构造下一步的输入 (与之前的逻辑相同)
            # 在水文场景中，这里应该用未来的气象预报值
            next_input_vars = current_input_features[:, :-1]  # 假设最后一列是目标变量
            current_input_features = torch.cat([next_input_vars, prediction], dim=1)

        return torch.tensor(predictions)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# --- 5. 主脚本 ---
if __name__ == '__main__':
    pl.seed_everything(42)
    # --- 参数设置 ---
    CSV_FILE = "timeseries_data_long.csv"
    # 注意：这里的输入列也包含了目标列，因为模型在训练时需要看到整个序列的输入特征
    INPUT_COLS = ['input_1', 'input_2', 'output']
    TARGET_COL = 'output'

    WARMUP_LEN = 365
    FORECAST_LEN = 90

    # 创建数据
    create_dummy_csv(CSV_FILE)

    print("\n" + "=" * 50)
    print("METHOD: WARM-UP + FORECAST CHUNKS")
    print("=" * 50)

    # 数据模块
    data_module = WarmupForecastDataModule(
        csv_path=CSV_FILE,
        input_cols=INPUT_COLS,
        target_col=TARGET_COL,
        warmup_len=WARMUP_LEN,
        forecast_len=FORECAST_LEN,
        batch_size=16  # 序列很长，batch size可能需要小一点
    )

    # 模型
    model = LSTMForecastModel(
        n_features=len(INPUT_COLS),
        hidden_size=64,
        n_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        warmup_len=WARMUP_LEN,
        forecast_len=FORECAST_LEN
    )

    # 训练
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='auto',
        callbacks=[RichProgressBar(), EarlyStopping('val_loss', patience=3)]
    )
    trainer.fit(model, datamodule=data_module)

    # 预测
    print("\n--- Predicting with the trained model (using autoregressive generation) ---")
    predictions = trainer.predict(model, datamodule=data_module)

    print(f"Generated {len(predictions)} sequences of {FORECAST_LEN}-step predictions.")
    if predictions:
        print("First prediction sequence (first 10 values):")
        print([f"{val:.4f}" for val in predictions[0][:10].tolist()])