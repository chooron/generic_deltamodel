# 训练多步预测模型
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from torchmetrics import R2Score, MeanSquaredError
import os

# Ensure results are reproducible
pl.seed_everything(42)


class CamelsTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for CAMELS time series data.

    ### MODIFICATION ###
    This version is for "Multi-step Forecasting":
    Use past `lag` days of forcing data PLUS the forcing data for the next `horizon` days
    to predict the target for the next `horizon` days.
    - Input (X): forcing[t-lag, ..., t, ..., t+horizon-1] (Length: lag + horizon)
    - Target (Y): target[t, ..., t+horizon-1] (Length: horizon)
    """

    def __init__(self, forcing, target, lag=365, horizon=7):
        super().__init__()
        self.forcing = torch.from_numpy(forcing).float()
        self.target = torch.from_numpy(target).float()
        self.lag = lag
        self.horizon = horizon  ### MODIFICATION ###: Added horizon parameter

        # The total length of a single sample's sequence (input + output)
        self.total_seq_len = self.lag + self.horizon

        # Recalculate the total number of samples
        # A sample is valid if we can form a full input/output sequence
        self.num_samples = self.forcing.shape[0] - self.total_seq_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Define the full sequence range for this sample
        seq_start = idx
        seq_end = idx + self.total_seq_len

        # 2. Define the input (x) range
        # Input uses forcing data from the start of the sequence
        x_end = idx + self.lag + self.horizon
        x = self.forcing[seq_start:x_end, :]

        # 3. Define the target (y) range
        # Target starts after the lag period and has a length of `horizon`
        y_start = idx + self.lag
        y_end = y_start + self.horizon
        y = self.target[y_start:y_end, :]

        # Squeeze the last dimension of y if it's 1
        return x, y.squeeze(-1)


class CamelsDataModule(pl.LightningDataModule):
    """
    Encapsulates dataset loading, splitting, and DataLoader creation.
    """

    ### MODIFICATION ###: Added horizon parameter
    def __init__(self, data_path, basin_idx=0, lag=365, horizon=7, batch_size=64):
        super().__init__()
        self.data_path = data_path
        self.basin_idx = basin_idx
        self.lag = lag
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        with open(self.data_path, "rb") as f:
            forcing, target, attribute = pickle.load(f)

        select_forcing = forcing[self.basin_idx, :, :]
        select_target = target[self.basin_idx, :, :]
        select_attribute = attribute[self.basin_idx, :]
        select_basin_area = select_attribute[11]
        select_target = (10 ** 3) * select_target * 0.0283168 * 3600 * 24 / (select_basin_area * (10 ** 6))

        full_dataset = CamelsTimeSeriesDataset(
            forcing=select_forcing,
            target=select_target,
            lag=self.lag,
            horizon=self.horizon  ### MODIFICATION ###: Pass horizon to dataset
        )

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


class RNNModel(nn.Module):
    """
    ### MODIFICATION ###
    A generic RNN model that supports bidirectional LSTMs/GRUs.
    The output layer maps the final hidden state to a sequence of `output_size` (horizon).
    """

    def __init__(self, input_size, output_size=7, hidden_size=64, rnn_type='LSTM', num_layers=1, dropout=0.0,
                 bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        rnn_class = getattr(nn, rnn_type)
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # The input dimension for the fully connected layer depends on bidirectionality
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # For LSTMs, hidden is a tuple (h_n, c_n)
        # For GRUs/RNNs, hidden is just h_n
        out, hidden = self.rnn(x)

        if self.rnn_type == 'LSTM':
            hidden = hidden[0]  # We only need the hidden state h_n, not the cell state c_n

        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        if self.bidirectional:
            # Concatenate the final hidden states of the forward and backward paths
            # The last layer's forward hidden state is at index -2
            # The last layer's backward hidden state is at index -1
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Use the final hidden state of the last layer
            last_hidden = hidden[-1, :, :]

        # Pass the concatenated hidden state through the fully connected layer
        prediction = self.fc(last_hidden)
        return prediction


class LitRNN(pl.LightningModule):
    """
    LightningModule for training the RNN model.
    """

    ### MODIFICATION ###: Added horizon and bidirectional params
    def __init__(self, input_size, horizon=7, hidden_size=64, dropout=0.2, rnn_type='LSTM', bidirectional=True,
                 learning_rate=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = RNNModel(
            input_size=input_size,
            output_size=self.hparams.horizon,
            hidden_size=self.hparams.hidden_size,
            rnn_type=self.hparams.rnn_type,
            dropout=self.hparams.dropout,
            bidirectional=self.hparams.bidirectional
        )
        self.loss_fn = MeanSquaredError()
        # multioutput='uniform_average' is the default and suitable here
        self.r2_metric = R2Score(multioutput='uniform_average')

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        r2 = self.r2_metric(y_hat, y)  # No squeeze needed for multi-output
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
        self.log('test_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        return r2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate,
                                     weight_decay=self.hparams.weight_decay)
        return optimizer


if __name__ == '__main__':
    # --- Hyperparameter Settings ---
    DATA_PATH = "E:\PaperCode\dpl-project\generic_deltamodel\data\camels_data\camels_dataset"  # IMPORTANT: Update this path
    BASIN_INDEX = 0
    LAG = 365
    HORIZON = 7  ### MODIFICATION ###
    BATCH_SIZE = 256
    HIDDEN_SIZE = 64
    DROPOUT = 0.2
    RNN_TYPE = 'GRU'  # 'LSTM' or 'GRU'
    BIDIRECTIONAL = True  ### MODIFICATION ###
    LEARNING_RATE = 5e-4  # Might need tuning
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    PATIENCE = 20

    # --- 1. Initialize Data Module ---
    # Check if the data file exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}. Please update the DATA_PATH variable.")

    with open(DATA_PATH, "rb") as f:
        forcing_data, _, _ = pickle.load(f)
    INPUT_SIZE = forcing_data.shape[2]

    dm = CamelsDataModule(
        data_path=DATA_PATH,
        basin_idx=BASIN_INDEX,
        lag=LAG,
        horizon=HORIZON,  ### MODIFICATION ###
        batch_size=BATCH_SIZE
    )

    # --- 2. Initialize Model ---
    lit_model = LitRNN(
        input_size=INPUT_SIZE,
        horizon=HORIZON,  ### MODIFICATION ###
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        rnn_type=RNN_TYPE,
        bidirectional=BIDIRECTIONAL,  ### MODIFICATION ###
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --- 3. Set up Callbacks ---
    early_stop_callback = EarlyStopping(
        monitor='val_r2',
        patience=PATIENCE,
        verbose=True,
        mode='max'
    )

    model_prefix = f"Bi{RNN_TYPE}" if BIDIRECTIONAL else RNN_TYPE
    model_checkpoint = ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'{model_prefix}-best-{{epoch:02d}}-{{val_r2:.4f}}',
        save_top_k=1,
        monitor='val_r2',
        mode='max'
    )

    # --- 4. Initialize Trainer and Start Training ---
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop_callback, model_checkpoint],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=pl.loggers.TensorBoardLogger("logs/", name=f"{model_prefix}_model_R2_horizon{HORIZON}")
    )

    print(f"--- Starting training for {model_prefix} model with R2 optimization (Horizon: {HORIZON}) ---")
    trainer.fit(model=lit_model, datamodule=dm)

    # --- 5. Evaluate the best model on all datasets ---
    print("\n" + "=" * 50)
    print("--- Training finished. Evaluating the best model on all datasets. ---")
    best_model_path = model_checkpoint.best_model_path

    if not best_model_path or not os.path.exists(best_model_path):
        print("No best model was saved. Evaluating with the last model state.")
        best_model = lit_model
    else:
        print(f"Loading best model from: {best_model_path}")
        best_model = LitRNN.load_from_checkpoint(best_model_path)

    best_model.eval()  # Set model to evaluation mode

    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    train_results = trainer.test(model=best_model, dataloaders=train_loader, verbose=False)
    val_results = trainer.test(model=best_model, dataloaders=val_loader, verbose=False)
    test_results = trainer.test(model=best_model, dataloaders=test_loader, verbose=False)

    train_r2 = train_results[0]['test_r2']
    val_r2 = val_results[0]['test_r2']
    test_r2 = test_results[0]['test_r2']

    print("\n--- Final Model Fitting Accuracy (R²) ---")
    print(f"Training Set R²:   {train_r2:.4f}")
    print(f"Validation Set R²: {val_r2:.4f}")
    print(f"Test Set R²:       {test_r2:.4f}")
    print("=" * 50)