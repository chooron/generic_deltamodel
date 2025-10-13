from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmg.models.neural_networks.ann import AnnModel
from dmg.models.neural_networks.vanilla_transformer import VanillaTransformer


class VanillaTransformerMlpModel(torch.nn.Module):
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
            nhead=4,
            num_encoder_layers=2,
            transformer_dim_fc=256,
            device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'VanillaTransformerMlpModel'

        self.transfomerinv = nn.Sequential(
            VanillaTransformer(
                nx1, hiddeninv1, nhead,
                num_encoder_layers, transformer_dim_fc,
                dropout=dr1, output_dim=ny1, seq_len=730),
        )
        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2,
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
        z1_permute = torch.permute(z1, (1, 0, 2))
        transfomer_out = self.transfomerinv(z1_permute).permute(1, 0, 2)  # dim: timesteps, gages, params
        ann_out = self.ann(z2)
        return F.sigmoid(transfomer_out), F.sigmoid(ann_out)
