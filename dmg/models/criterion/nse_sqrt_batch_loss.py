from typing import Any, Optional, Union

import numpy as np
import torch

from dmg.models.criterion.base import BaseCriterion


class NseSqrtBatchLoss(BaseCriterion):
    """Square-root normalized squared error (NSE) loss function.

    Same as Fredrick 2019, batch NSE loss.
    Adapted from Yalan Song.

    Uses the first variable of the target array as the target variable.
    
    The sNSE is calculated as:
        p: predicted value,
        t: target value,
        sNSE = 1 - sum((p - t)^2) / sum(t - mean(t))

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - y_obs: Tensor of target observation data. (Required)

        - eps: Stability term to prevent division by zero. Default is 0.1.

        - beta: Stability term to prevent division by zero. Default is 1e-6.
    """
    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Union[torch.Tensor, float],
    ) -> None:
        super().__init__(config, device)
        self.name = 'Batch Sqrt NSE Loss'
        self.config = config
        self.device = device

        try:
            y_obs = kwargs['y_obs']
            self.std = np.nanstd(y_obs[:, :, 0].cpu().detach().numpy(), axis=0)
        except KeyError as e:
            raise KeyError("'y_obs' is not provided in kwargs") from e

        self.eps = kwargs.get('eps', config.get('eps', 0.1))
        self.beta = kwargs.get('beta', config.get('beta', 1e-6))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments.

            - sample_ids: indices of samples included in batch. (Required)
        
        Returns
        -------
        torch.Tensor
            The loss value.
        """
        prediction, target = self._format(y_pred, y_obs)

        try:
            sample_ids = kwargs['sample_ids'].astype(int)
        except KeyError as e:
            raise KeyError("'sample_ids' is not provided in kwargs") from e


        if len(target) > 0:
            # Prepare grid-based standard deviations for normalization.
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[sample_ids].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )

            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]

            sq_res = torch.sqrt((p_sub - t_sub)**2 + self.beta)
            norm_res = sq_res / (std_sub + self.eps)
            loss = torch.mean(norm_res)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
