from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class RmseLoss(BaseCriterion):
    """Root mean squared error (RMSE) loss function.

    The RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((p - t)^2))

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - alpha: Weighting factor for the log-sqrt RMSE. Default is 0.25.

        - beta: Stability term to prevent division by zero. Default is 1e-6.
    """
    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: float,
    ) -> None:
        super().__init__(config, device)
        self.name = 'RMSE Loss'
        self.config = config
        self.device = device

        self.alpha = kwargs.get('alpha', config.get('alpha', 0.25))
        self.beta = kwargs.get('beta', config.get('beta', 1e-6))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss.
        
        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments for interface compatibility, not used.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        prediction, target = self._format(y_pred, y_obs)

        if len(target) > 0:
            # Mask where observations are valid (not NaN).
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            loss = torch.sqrt(((p_sub - t_sub) ** 2).mean())
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
