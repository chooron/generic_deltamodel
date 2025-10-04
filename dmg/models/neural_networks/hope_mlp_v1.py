from typing import Optional

import torch
import torch.nn.functional as F

from dmg.models.neural_networks.ann import AnnModel
from dmg.models.neural_networks.hope import Hope

class HopeMlpV1(torch.nn.Module):

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
        self.name = 'HopeMlpModel'
        self.hope_layer = Hope(
            input_size=nx1, output_size=ny1, hidden_size=hiddeninv1,
            dropout=dr1, n_layers=4,
        )
        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2,
        )

    @classmethod
    def build_by_config(cls, config):
        return cls(
            nx1=config['nn_model']['nx1'],
            ny1=config['nn_model']['ny1'],
            hiddeninv1=config['nn_model']['hope_hidden_size'],
            nx2=config['nn_model']['nx2'],
            ny2=config['nn_model']['ny2'],
            hiddeninv2=config['nn_model']['mlp_hidden_size'],
            dr1=config['nn_model']['hope_dropout'],
            dr2=config['nn_model']['mlp_dropout'],
            device=config['nn_model']['device'],
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
        hope_out = self.hope_layer(z1)  # dim: timesteps, gages, params
        ann_out = self.ann(z2)
        hope_out = F.sigmoid(hope_out)
        ann_out = F.sigmoid(ann_out)
        # print(hope_out.shape, ann_out.shape)
        return hope_out, ann_out
