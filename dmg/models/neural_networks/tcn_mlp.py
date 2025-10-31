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

        self.tcninv = nn.Sequential(
            nn.Linear(nx1, hiddeninv1),
            TCN(
                num_inputs=hiddeninv1,
                num_channels=[hiddeninv1] * 8,
                kernel_size=4,
                dilations=None,
                dilation_reset=None,
                dropout=0.1,
                causal=True,
                use_norm="weight_norm",
                activation="leaky_relu",
                kernel_initializer="xavier_uniform",
                use_skip_connections=True,
                input_shape="NLC",
                output_projection=None,
                output_activation=None,
            ),
        )
        self.ln = nn.LayerNorm(hiddeninv1)
        self.tncdrop = nn.Dropout(dr1)
        self.a = 0.2
        self.fc = nn.Linear(hiddeninv1, ny1)
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )
    
    @classmethod
    def build_by_config(cls, config, device):
        return cls(
            nx1=config['nx1'],
            ny1=config['ny1'],
            hiddeninv1=config["tcn_hidden_size"],
            nx2=config['nx2'],
            ny2=config['ny2'],
            hiddeninv2=config["mlp_hidden_size"],
            dr1=config["tcn_dropout"],
            dr2=config["mlp_dropout"],
            device=device,
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
        tcn_out = self.tcninv(
            z1.permute(1, 0, 2)
        )  # dim: timesteps, gages, params
        tcn_out = self.ln(tcn_out)
        fc_out = self.fc(self.tncdrop(tcn_out).permute(1, 0, 2))
        ann_out = self.ann(z2)
        return F.sigmoid(fc_out*self.a), F.sigmoid(ann_out)
