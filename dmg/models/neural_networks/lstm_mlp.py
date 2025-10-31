from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmg.models.neural_networks.layers.ann import AnnModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.
    
    Supports GPU and CPU forwarding.
    
    Parameters
    ----------
    nx1
        Number of LSTM input features.
    ny1
        Number of LSTM output features.
    hiddeninv1
        LSTM hidden size.
    nx2
        Number of MLP input features.
    ny2
        Number of MLP output features.
    hiddeninv2
        MLP hidden size.
    dr1
        Dropout rate for LSTM. Default is 0.5.
    dr2
        Dropout rate for MLP. Default is 0.5.
    device
        Device to run the model on. Default is 'cpu'.
    """

    def __init__(
            self,
            *,
            nx1: int,
            ny1: int,
            hiddeninv1: int,
            nx2: int,
            ny2: int,
            hiddeninv2: int,
            dr1: Optional[float] = 0.5,
            dr2: Optional[float] = 0.5,
            device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'

        # GPU-only HydroDL LSTM.
        self.lstminv = nn.Sequential(
            nn.Linear(nx1, hiddeninv1),
            nn.ReLU(),
            nn.LSTM(hiddeninv1, hiddeninv1, dropout=dr1, batch_first=False),
        )
        self.fc = nn.Linear(hiddeninv1, ny1)
        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2,
        )
    
    @classmethod
    def build_by_config(cls, config, device):
        return cls(
            nx1=config['nx'],
            ny1=config['ny1'],
            hiddeninv1=config["lstm_hidden_size"],
            nx2=config['nx2'],
            ny2=config['ny2'],
            hiddeninv2=config["mlp_hidden_size"],
            dr1=config["lstm_dropout"],
            dr2=config["mlp_dropout"],
            device=device,
        )

    def forward(
        self,
        data_dict:dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = data_dict['xc_nn_norm']
        z2 = data_dict['c_nn_norm']
        lstm_out, _ = self.lstminv(z1)  # dim: timesteps, gages, params
        fc_out = self.fc(lstm_out)
        ann_out = self.ann(z2)
        return F.sigmoid(fc_out), F.sigmoid(ann_out)
