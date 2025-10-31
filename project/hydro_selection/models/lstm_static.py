from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class LstmStaticModel(torch.nn.Module):
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
        nx2: int,
        ny: int,
        hiddeninv1: int,
        hiddeninv2: int,
        dr1: float = 0.5,
        dr2: float = 0.5,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = "LstmMlpModel"
        self.lstminv = nn.Sequential(
            nn.Linear(nx1, hiddeninv1),
            nn.ReLU(),
            nn.LSTM(hiddeninv1, hiddeninv1, dropout=dr1, batch_first=False),
        ).to((torch.device(device)))
        self.fc = nn.Sequential(
            nn.Linear(hiddeninv1 + nx2, hiddeninv2),
            nn.Tanh(),
            nn.Linear(hiddeninv2, ny),
        ).to(torch.device(device))

    @classmethod
    def build_by_config(cls, config: dict, device: Optional[str] = "cpu"):
        return cls(
            nx1=config["nx1"],
            nx2=config["nx2"],
            ny=config["ny"],
            hiddeninv1=config["lstm_hidden_size"],
            hiddeninv2=config["fc_hidden_size"],
            dr1=config["dr1"],
            device=device,
        )

    def forward(
        self,
        data_dict:dict,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        z1
            The LSTM input tensor.
        z2
            The MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        z1 = data_dict['x_nn_norm']
        z2 = data_dict['c_nn_norm']
        lstm_out, _ = self.lstminv(z1)  # dim: timesteps, gages, params
        fc_input = torch.concat([lstm_out[-1, :, :], z2], dim=-1)
        fc_out = self.fc(fc_input)
        return F.sigmoid(fc_out)
