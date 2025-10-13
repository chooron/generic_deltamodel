import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping

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
        """
        Args:
            csv_path (str): CSV文件路径.
            input_cols (list): 输入特征的列名列表.
            target_col (str): 目标变量的列名.
            warmup_len (int): 用于预热的历史时间步长.
            forecast_len (int): 预测未来的时间步长.
            batch_size (int): 批处理大小.
        """
        super().__init__()
        # 保存所有初始化参数
        self.save_hyperparameters()
        self.scaler = StandardScaler()
        # self.train_df, self.val_df, self.test_df 将在 setup 中定义
        self.train_df = None
        self.val_df = None
        self.test_df = None

    def setup(self, stage=None):
        # 1. 加载并预处理数据
        df = pd.read_csv(self.hparams.csv_path)
        # 将 'date' 列转换为 datetime 对象，并设为索引
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 2. 按指定的日期范围划分数据
        self.train_df = df.loc['1980-10-01':'1995-09-30'].copy()
        self.val_df = df.loc['1995-10-01':'2000-09-30'].copy()
        self.test_df = df.loc['2000-10-01':'2010-09-30'].copy()

        # 使用 .copy() 避免 SettingWithCopyWarning

        # 3. 数据标准化
        # 准备所有需要标准化的列
        all_cols = self.hparams.input_cols + (
            [self.hparams.target_col] if self.hparams.target_col not in self.hparams.input_cols else [])

        # 仅在训练集上拟合 StandardScaler
        self.scaler.fit(self.train_df[all_cols])

        # 对所有数据集应用标准化
        self.train_df[all_cols] = self.scaler.transform(self.train_df[all_cols])
        self.val_df[all_cols] = self.scaler.transform(self.val_df[all_cols])
        self.test_df[all_cols] = self.scaler.transform(self.test_df[all_cols])

        # 4. 根据 stage 创建对应的 Dataset
        if stage in ('fit', None):
            self.train_dataset = WarmupForecastDataset(self.train_df, self.hparams.input_cols, self.hparams.target_col,
                                                       self.hparams.warmup_len,
                                                       self.hparams.forecast_len)
            self.val_dataset = WarmupForecastDataset(self.val_df, self.hparams.input_cols, self.hparams.target_col,
                                                     self.hparams.warmup_len,
                                                     self.hparams.forecast_len)
        if stage in ('test', None):
            self.test_dataset = WarmupForecastDataset(self.test_df, self.hparams.input_cols, self.hparams.target_col,
                                                      self.hparams.warmup_len,
                                                      self.hparams.forecast_len)
        if stage == 'predict':
            # 预测时通常使用测试集
            self.predict_dataset = WarmupForecastDataset(self.test_df, self.hparams.input_cols, self.hparams.target_col,
                                                         self.hparams.warmup_len, self.hparams.forecast_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4,
                          persistent_workers=True)

    def predict_dataloader(self):
        # 预测时 batch_size 通常为 1
        return DataLoader(self.predict_dataset, batch_size=1, shuffle=False, num_workers=4, persistent_workers=True)


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
    CSV_FILE = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_data\01013500.csv"
    # 注意：这里的输入列也包含了目标列，因为模型在训练时需要看到整个序列的输入特征
    INPUT_COLS = ['prcp(mm/day)','tmean(C)','dayl(day)','srad(W/m2)','vp(Pa)']
    TARGET_COL = 'flow(mm)'

    WARMUP_LEN = 365
    FORECAST_LEN = 365

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
        max_epochs=100,
        accelerator='cpu',
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
