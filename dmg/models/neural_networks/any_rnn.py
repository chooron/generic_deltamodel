# a new rnn (dynamic estimate) with a common rnn (constant estimate)
from typing import Optional
import torch
import torchrecurrent

from dmg.models.neural_networks import AnnModel


class AnyRnnMlp(torch.nn.Module):
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
            rnn_cell: Optional[str] = 'FastRNN',
            device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        self.rnn_cell = getattr(torchrecurrent, rnn_cell)
        self.any_rnn = self.rnn_cell(
            input_size=nx1, hidden_size=hiddeninv1, dropout=dr1,
        ).to(torch.device(device))
        self.fc = torch.nn.Linear(hiddeninv1, ny1).to(torch.device(device))
        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2,
        ).to(torch.device(device))

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
        rnn_out, (hn, cn) = self.any_rnn(z1)
        fc_out = self.fc(rnn_out)
        ann_out = self.ann(z2)
        return fc_out, ann_out
