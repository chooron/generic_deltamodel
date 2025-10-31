from typing import Any

import torch
import torch.nn as nn


class MlpModel(nn.Module):
    """Multi-layer perceptron (MLP) model.
    
    Parameters
    ----------
    config
        Configuration dictionary with model settings.
    nx
        Number of input features.
    ny
        Number of output features.
    """
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__()
        self.name = 'MlpModel'
        self.config = config

        self.L1 = nn.Linear(
            nx,  self.config['hidden_size'],
        )
        self.L2 = nn.Linear(
            self.config['hidden_size'], self.config['hidden_size'],
        )
        self.L3 = nn.Linear(self.config['hidden_size'], ny)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x
            Input tensor.
        
        Returns
        -------
        out
            Output tensor.
        """
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        return out
