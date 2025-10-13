import os  # <<< NEW: Import os for environment variables
from typing import Optional, Union, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler


class PubSampler(BaseSampler):
    """
    Hydrological data sampler for Predictions in Ungauged Basins (PUB) studies.

    This sampler determines the train/validation split based on a specified
    test group file, with paths configured via environment variables.
    """

    def __init__(
            self,
            config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['delta_model']['phy_model']['warm_up']
        self.rho = config['delta_model']['rho']
        self._setup_basin_indices()

    def load_data(self):
        """Custom implementation for loading data."""
        print("Loading data...")

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        print("Preprocessing data...")

    # <<< CHANGED: The entire method is rewritten for the PUB setup >>>
    def _setup_basin_indices(self) -> None:
        """
        Sets up training and validation indices for a PUB cross-validation fold.

        It reads the test basin group from a file specified by a config ID
        and defines the training set as all other available basins.
        Paths are read from environment variables.
        """
        # 1. Get file paths from environment variables
        groups_dir = os.getenv("BASIN_GROUPS_DIR")
        all_basins_file = os.getenv("GAGE_INFO")

        if not groups_dir or not all_basins_file:
            raise EnvironmentError(
                "Please set the 'BASIN_GROUPS_DIR' and 'GAGE_INFO' environment variables."
            )

        # 2. Get the test group ID from the config
        try:
            test_group_id = self.config['test']['test_group_id']
        except KeyError:
            raise KeyError("Config must contain ['test']['test_group_id']")

        # 3. Load all basin IDs to create a complete ID-to-index mapping
        all_basin_ids = np.load(all_basins_file)
        id_to_index_map = {basin_id: i for i, basin_id in enumerate(all_basin_ids)}

        # 4. Load the basin IDs for the designated test group
        test_group_file = os.path.join(groups_dir, f"group_{test_group_id}.npy")
        if not os.path.exists(test_group_file):
            raise FileNotFoundError(f"Test group file not found: {test_group_file}")

        test_basin_ids = np.load(test_group_file)

        # 5. Define train and validation indices
        all_ids_set = set(all_basin_ids)
        test_ids_set = set(test_basin_ids)

        if not test_ids_set.issubset(all_ids_set):
            raise ValueError("Test basin IDs contain IDs not present in the main basin file.")

        train_ids_set = all_ids_set - test_ids_set

        # Convert basin IDs back to zero-based array indices
        self.val_indices = np.array(sorted([id_to_index_map[bid] for bid in test_ids_set]))
        self.train_indices = np.array(sorted([id_to_index_map[bid] for bid in train_ids_set]))

        print(f"PUB Setup for Fold {test_group_id}:")
        print(f"  - Test basins: {len(self.val_indices)} (from group_{test_group_id}.npy)")
        print(f"  - Train basins: {len(self.train_indices)} (all other basins)")

    # The rest of the methods remain unchanged as they correctly use
    # self.train_indices and self.val_indices established above.

    def get_training_sample(
            self,
            dataset: dict[str, NDArray[np.float32]],
            nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch from the training basins."""
        batch_size = self.config['train']['batch_size']
        n_train_basins = len(self.train_indices)

        local_indices, i_t = random_index(n_train_basins, nt, (batch_size, self.rho), warm_up=self.warm_up)
        global_indices = self.train_indices[local_indices]

        return {
            'x_phy': self.select_subset(dataset['x_phy'], global_indices, i_t),
            'c_phy': dataset['c_phy'] if len(dataset['c_phy']) == 0 else dataset['c_phy'][global_indices],
            'c_nn': dataset['c_nn'][global_indices],
            'c_nn_norm': dataset['c_nn_norm'][global_indices],
            'x_nn_norm': self.select_subset(dataset['x_nn_norm'], global_indices, i_t, has_grad=False),
            'xc_nn_norm': self.select_subset(dataset['xc_nn_norm'], global_indices, i_t, has_grad=False),
            'target': self.select_subset(dataset['target'], global_indices, i_t)[self.warm_up:, :],
            'batch_sample': global_indices,
        }

    def get_validation_sample(
            self,
            dataset: dict[str, NDArray[np.float32]],
            basin_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a complete data sample for one validation basin."""
        if basin_idx not in self.val_indices:
            raise ValueError(f"Basin index {basin_idx} is not in the validation set.")

        i_grid = np.array([basin_idx])

        validation_batch = {}
        for key, value in dataset.items():
            if value.ndim == 3:
                tensor = value[:, i_grid, :]
            elif value.ndim == 2:
                tensor = value[i_grid, :]
            else:
                continue
            validation_batch[key] = tensor
        return validation_batch

    def select_subset(
            self,
            x: torch.Tensor,
            i_grid: NDArray[np.int_],
            i_t: Optional[NDArray[np.int_]] = None,
            c: Optional[NDArray[np.float32]] = None,
            tuple_out: bool = False,
            has_grad: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, nx = len(i_grid), x.shape[-1]
        if i_t is not None:
            x_tensor = torch.zeros(
                [self.rho + self.warm_up, batch_size, nx],
                device=self.device, requires_grad=has_grad,
            )
            for k in range(batch_size):
                x_tensor[:, k:k + 1, :] = x[i_t[k] - self.warm_up:i_t[k] + self.rho, i_grid[k]:i_grid[k] + 1, :]
        else:
            x_tensor = x[:, i_grid, :].float().to(self.device) if x.ndim == 3 else x[i_grid, :].float().to(self.device)

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, self.rho + self.warm_up, 1)
            return (x_tensor, c_tensor) if tuple_out else torch.cat((x_tensor, c_tensor), dim=2)

        return x_tensor
