from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_tcn import TCN

from dmg.models.neural_networks.ann import AnnModel


class TcnMlpModel(torch.nn.Module):
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
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = "LstmMlpModel"

        self.tcninv = TCN(
            num_inputs=nx1,
            num_channels=[hiddeninv1] * 8,
            kernel_size=4,
            dilations=None,
            dilation_reset=None,
            dropout=0.1,
            causal=True,
            use_norm="weight_norm",
            activation="relu",
            kernel_initializer="xavier_uniform",
            use_skip_connections=True,
            input_shape="NLC",
            output_projection=None,
            output_activation=None,
        )
        self.tncdrop = nn.Dropout(dr1)
        self.fc = nn.Linear(hiddeninv1, ny1)
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        tcn_out = self.tcninv(z1.permute(1, 0, 2))  # dim: timesteps, gages, params
        fc_out = self.fc(self.tncdrop(tcn_out).permute(1, 0, 2))
        ann_out = self.ann(z2)
        return F.sigmoid(fc_out), F.sigmoid(ann_out)
