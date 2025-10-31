from typing import Optional

import torch
import torch.nn.functional as F

from dmg.models.neural_networks.layers.ann import AnnModel
from dmg.models.neural_networks.layers.hope import Hope


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
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = "HopeMlpModel"
        # cfg = {"lr_min": 0.001, "lr": 0.01, "lr_dt": 0.0, "min_dt": 0.001,
        #        "max_dt": 1, "wd": 0.0, "d_state": 64, "cfr": 1.0, "cfi": 1.0}
        self.hope_layer = Hope(
            input_size=nx1,
            output_size=ny1,
            hidden_size=hiddeninv1,
            dropout=dr1,
            n_layers=4,
            prenorm=False,
            cfg={
                "lr_min": 0.001,
                "lr": 0.01,
                "lr_dt": 0.0,
                "min_dt": 0.001,
                "max_dt": 1,
                "wd": 0.0,
                "d_state": 64,
                "cfr": 1.0,
                "cfi": 1.0,
                "use_gated": False,
                "out_activation": "glu",
            },
        )
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    @classmethod
    def build_by_config(cls, config: dict, device: Optional[str] = "cpu"):
        return cls(
            nx1=config["nx"],
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = data_dict['xc_nn_norm']
        z2 = data_dict['c_nn_norm']
        hope_out = self.hope_layer(
            torch.permute(z1, (1, 0, 2))
        )  # dim: timesteps, gages, params
        ann_out = self.ann(z2)
        hope_out = F.sigmoid(hope_out.permute(1, 0, 2))
        ann_out = F.sigmoid(ann_out)
        # print(hope_out.shape, ann_out.shape)
        return hope_out, ann_out
