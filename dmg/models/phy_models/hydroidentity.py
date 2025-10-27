from typing import Any, Optional, Union

import torch


class HydroIndentity(torch.nn.Module):
    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "Hydro Indentity Model"
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ["prcp", "tmean", "pet"]
        self.nmul = 1
        self.device = device
        self.parameter_bounds = {}
        self.phy_param_names = self.parameter_bounds.keys()
        self.learnable_param_count = 1
        self.learnable_param_count1 = 1
        self.learnable_param_count2 = 0

        if not device:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get("warm_up", self.warm_up)
            self.warm_up_states = config.get(
                "warm_up_states", self.warm_up_states
            )
            self.dy_drop = config.get("dy_drop", self.dy_drop)
            self.dynamic_params = config["dynamic_params"].get(
                self.__class__.__name__, self.dynamic_params
            )
            self.variables = config.get("variables", self.variables)
            self.routing = config.get("routing", self.routing)
            self.comprout = config.get("comprout", self.comprout)
            self.nearzero = config.get("nearzero", self.nearzero)
            self.nmul = config.get("nmul", self.nmul)

    def forward(
        self,
        _: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for HBV.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            这时候输入的parameter其实就是nn的预测结果，我们就直接传出即可

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Initialization
        if self.warm_up_states:
            pass
        else:
            # No state warm up - run the full model for warm_up days.
            self.pred_cutoff = self.warm_up

        return {"streamflow": parameters}