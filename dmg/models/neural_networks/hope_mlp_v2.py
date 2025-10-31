from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmg.models.neural_networks.layers.ann import AnnModel
from dmg.models.neural_networks.layers.hope import Hope

class GaussianSmoother(nn.Module):
    def __init__(self, channels, kernel_size=9, sigma=2.0):
        super().__init__()
        half = kernel_size // 2
        x = torch.arange(-half, half + 1).float()
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        k = k.view(1, 1, kernel_size).repeat(channels, 1, 1)
        self.register_buffer("kernel", k)

    def forward(self, x):  # x: (B, C, L)
        pad = self.kernel.size(-1) // 2
        return F.conv1d(x, self.kernel, padding=pad, groups=x.size(1))

class HopeMlpV2(torch.nn.Module):
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
        self.name = "HopeMlpModel"
        config = {
            "lr_min": 0.001,
            "lr": 0.01,
            "lr_dt": 0.0,
            "min_dt": 0.001,
            "max_dt": 0.01,  # change to 0.01
            "wd": 0.01,  # change to 0.01
            "d_state": 64,
            "cfr": 1.0,
            "cfi": 1.0,
            "use_gated": False,
            "out_activation": 'glu',
        }
        self.hope_layer = Hope(
            input_size=nx1,
            output_size=ny1,
            hidden_size=hiddeninv1,
            dropout=dr1,
            cfg=config,
            prenorm=True,
            n_layers=2,
        )
        self.smoother = GaussianSmoother(channels=output_size, kernel_size=15, sigma=2.0)
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    @classmethod
    def build_by_config(cls, config):
        return cls(
            nx1=config["nn_model"]["nx1"],
            ny1=config["nn_model"]["ny1"],
            hiddeninv1=config["nn_model"]["hope_hidden_size"],
            nx2=config["nn_model"]["nx2"],
            ny2=config["nn_model"]["ny2"],
            hiddeninv2=config["nn_model"]["mlp_hidden_size"],
            dr1=config["nn_model"]["hope_dropout"],
            dr2=config["nn_model"]["mlp_dropout"],
            device=config["nn_model"]["device"],
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
        hope_out = self.hope_layer(
            torch.permute(z1, (1, 0, 2))
        )  # dim: timesteps, gages, params
        ann_out = self.ann(z2)
        hope_out = F.sigmoid(hope_out)
        hope_out = self.smoother(hope_out.transpose(1, 2)).transpose(1, 2)
        ann_out = F.sigmoid(ann_out)
        # print(hope_out.shape, ann_out.shape)
        return torch.permute(hope_out, (1, 0, 2)), ann_out
