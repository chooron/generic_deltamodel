import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from torchmetrics import R2Score, MeanSquaredError  # 导入R2Score
import sys

sys.path.append(r"E:\PaperCode\dpl-project\generic_deltamodel")
from dmg.models.neural_networks.cnn_transformer import TimeSeriesTransformer
# from dmg.models.neural_networks.simple_transformer import SimpleTransformer
import numpy as np
import os

# 确保结果可复现
pl.seed_everything(42)


class CamelsTimeSeriesDataset(Dataset):
    """
    用于CAMELS时间序列数据的PyTorch数据集。

    此版本为 "即时预测" (Nowcasting) 设置：
    使用包含当天在内的过去`lag`天forcing数据，来预测当天的target。
    即：Input: forcing[t-lag+1, ..., t], Target: target[t]
    """

    def __init__(self, forcing, target, lag=365):
        super().__init__()
        # 在这个模式下，horizon的概念不再适用，因为我们总是在预测“现在”
        self.forcing = torch.from_numpy(forcing).float()
        self.target = torch.from_numpy(target).float()
        self.lag = lag

        # 重新计算样本总数
        # 只要能凑齐一个长度为lag的输入窗口，就能进行一次预测
        self.num_samples = self.forcing.shape[0] - self.lag + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. 确定输入序列 x 的范围
        # 输入x从idx开始，长度为lag
        x_start = idx
        x_end = idx + self.lag

        # 2. 确定目标 y 的位置
        # 目标y的日期与输入x序列的最后一天完全相同
        y_idx = x_end - 1

        # 3. 切片获取数据
        x = self.forcing[x_start:x_end, :]
        y = self.target[y_idx, :]

        return x, y


class CamelsDataModule(pl.LightningDataModule):
    """
    封装了数据集加载、划分和DataLoader创建的LightningDataModule。
    """

    def __init__(self, data_path, basin_idx=0, lag=365, batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.basin_idx = basin_idx
        self.lag = lag
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # 1. 加载原始数据
        with open(self.data_path, "rb") as f:
            forcing, target, attribute = pickle.load(f)

        # 选择一个流域的数据
        select_forcing = forcing[self.basin_idx, :, :]
        select_target = target[self.basin_idx, :, :]
        select_attribute = attribute[self.basin_idx, :]
        select_basin_area = select_attribute[11]
        select_target = (10 ** 3) * select_target * 0.0283168 * 3600 * 24 / (select_basin_area * (10 ** 6))

        # 2. 创建完整的Dataset实例
        full_dataset = CamelsTimeSeriesDataset(
            forcing=select_forcing,
            target=select_target,
            lag=self.lag,
        )

        # 3. 划分数据集为 6:2:2
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))

        train_end = int(dataset_size * 0.5)
        val_end = train_end + int(dataset_size * 0.3)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


class R2Loss(nn.Module):
    """
    自定义损失函数，计算 -R^2。
    优化器最小化这个损失，就相当于最大化 R^2。
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        # 总平方和 (SST)
        ss_tot = torch.sum((y - torch.mean(y)) ** 2)
        # 残差平方和 (SSR)
        ss_res = torch.sum((y - y_hat) ** 2)
        # R^2 计算
        r2 = 1 - ss_res / ss_tot
        # 返回 -R^2 作为损失
        return -r2


class LitRNN(pl.LightningModule):
    """
    用于训练RNN模型的LightningModule。
    使用 R2Loss 进行优化，并使用 R2Score 进行监控。
    """

    def __init__(self, input_size, d_model=64, nhead=8,
                 num_encoder_layers=2, dim_feedforward=256,
                 conv_kernel_size=5, output_dim=1,seq_len=365,
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = TimeSeriesTransformer(
            input_dim=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            output_dim=output_dim,
            # seq_len=seq_len
        )
        # 使用自定义的 R2Loss 作为损失函数
        self.loss_fn = MeanSquaredError()

        # 使用 torchmetrics 计算 R2 分数作为监控指标
        self.r2_metric = R2Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        # 计算 R2 指标 (注意 y_hat 和 y 需要展平)
        r2 = self.r2_metric(y_hat.squeeze(), y.squeeze())
        return loss, r2

    def training_step(self, batch, batch_idx):
        loss, r2 = self._shared_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, r2 = self._shared_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, r2 = self._shared_step(batch)
        # 在 test_step 中，我们通常只关心指标，而不是损失
        # 我们将 R2 记录为 test_r2
        self.log('test_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return r2  # 返回指标值

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        return optimizer


if __name__ == '__main__':
    # --- 超参数设置 ---
    DATA_PATH = "E:\pycode\generic_deltamodel\data\camels_dataset"
    BASIN_INDEX = 0
    LAG = 365
    BATCH_SIZE = 256
    # LSTM 0.7
    LEARNING_RATE =5e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    PATIENCE = 50

    with open(DATA_PATH, "rb") as f:
        forcing_data, _, _ = pickle.load(f)
    INPUT_SIZE = forcing_data.shape[2]

    dm = CamelsDataModule(
        data_path=DATA_PATH,
        basin_idx=BASIN_INDEX,
        lag=LAG,
        batch_size=BATCH_SIZE
    )

    # --- 2. 初始化模型 ---
    lit_model = LitRNN(
        input_size=INPUT_SIZE,
        d_model=64,  # 模型内部的特征维度
        nhead=4,  # 多头注意力的头数
        num_encoder_layers=2,  # Transformer编码器的层数
        dim_feedforward=128,  # 前馈网络的隐藏层维度
        conv_kernel_size=3,  # 因果卷积的核大小
        output_dim=1,  # 预测目标维度 (例如，预测未来1个值)
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --- 3. 设置回调函数 ---
    early_stop_callback = EarlyStopping(
        monitor='val_r2',  # 监控验证集 R2 分数
        patience=PATIENCE,
        verbose=True,
        mode='max'  # R2 是越大越好，所以模式是 'max'
    )
    RNN_TYPE = "HOPE"
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'HOPE-best-{{epoch:02d}}-{{val_r2:.4f}}',
        save_top_k=1,
        monitor='val_r2',  # 监控验证集 R2 分数
        mode='max'  # 模式是 'max'
    )

    # --- 4. 初始化训练器并开始训练 ---
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop_callback, model_checkpoint],
        # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        accelerator='gpu',
        devices=1,
        logger=pl.loggers.TensorBoardLogger("logs/", name=f"{RNN_TYPE}_model_R2")
    )

    print(f"--- Starting training for {RNN_TYPE} model with R2 optimization ---")
    trainer.fit(model=lit_model, datamodule=dm)

    # --- 5. 使用最优模型评估三个数据集的拟合精度 ---
    print("\n" + "=" * 50)
    print("--- Training finished. Evaluating the best model on all datasets. ---")
    best_model_path = model_checkpoint.best_model_path

    if not best_model_path:
        print("No best model was saved. Evaluating with the last model state.")
        # 如果没有模型被保存（比如训练过早停止），则使用最后的模型
        best_model = lit_model
    else:
        print(f"Loading best model from: {best_model_path}")
        best_model = LitRNN.load_from_checkpoint(best_model_path)

    # 准备数据加载器
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # 在 Trainer 中运行评估
    # verbose=False 可以让输出更简洁
    train_results = trainer.test(model=best_model, dataloaders=train_loader, verbose=False)
    val_results = trainer.test(model=best_model, dataloaders=val_loader, verbose=False)
    test_results = trainer.test(model=best_model, dataloaders=test_loader, verbose=False)

    # 提取并打印 R2 分数
    train_r2 = train_results[0]['test_r2']
    val_r2 = val_results[0]['test_r2']
    test_r2 = test_results[0]['test_r2']

    print("\n--- Final Model Fitting Accuracy (R²) ---")
    print(f"Training Set R²:   {train_r2:.4f}")
    print(f"Validation Set R²: {val_r2:.4f}")
    print(f"Test Set R²:       {test_r2:.4f}")
    print("=" * 50)
