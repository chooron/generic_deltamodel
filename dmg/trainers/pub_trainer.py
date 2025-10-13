import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid
from dmg.core.data.samplers.pub_sampler import PubSampler
# Make sure your factory can import your new PubSampler
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class PubTrainer(BaseTrainer):
    """
    Generic trainer adapted for PUB (Predictions in Ungauged Basins) experiments.
    """

    def __init__(
            self, config: dict[str, Any],
            model: torch.nn.Module = None,
            train_dataset: Optional[dict] = None,
            eval_dataset: Optional[dict] = None,
            dataset: Optional[dict] = None,
            loss_func: Optional[torch.nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[torch.nn.Module] = None,
            verbose: Optional[bool] = False
    ) -> None:
        super().__init__(config, model)
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.is_in_train = False
        self.sampler = import_data_sampler(config['data_sampler'])(config)

        if 'train' in config['mode']:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            # Now, initialize the PubSampler with the n_basins argument.
            log.info("Initializing experiment")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config['delta_model']['nn_model']['lr_scheduler']:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            self.start_epoch = self.config['train']['start_epoch'] + 1
            if self.start_epoch > 1:
                self.load_states()

    # ... (init_optimizer, init_scheduler, load_states methods remain unchanged) ...
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer."""
        name = self.config['train']['optimizer']
        learning_rate = self.config['delta_model']['nn_model']['learning_rate']
        optimizer_dict = {
            # 'SGD': torch.optim.SGD,
            # 'Adam': torch.optim.Adam,
            # 'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            # 'RMSprop': torch.optim.RMSprop,
        }
        cls = optimizer_dict.get(name)
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized.")
        self.optimizer = cls(self.model.get_parameters(), lr=learning_rate)
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler."""
        name = self.config['delta_model']['nn_model']['lr_scheduler']
        scheduler_dict = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }
        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized.")
        self.scheduler = cls(self.optimizer, **self.config['delta_model']['nn_model']['lr_scheduler_params'])
        return self.scheduler

    def load_states(self) -> None:
        """Load model, optimizer, and scheduler states from a checkpoint."""
        path = self.config['model_path']
        checkpoint_file = os.path.join(path, f"train_state_epoch_{self.start_epoch - 1}.pt")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint for epoch {self.start_epoch - 1}.")
        checkpoint = torch.load(checkpoint_file)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.load_model(epoch=self.start_epoch - 1)
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        torch.set_rng_state(checkpoint['random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'], self.config
        )
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_one_epoch(epoch, n_samples, n_minibatch, n_timesteps)

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        """Train model for one epoch."""
        start_time = time.perf_counter()
        prog_str = f"Epoch {epoch}/{self.epochs}"
        self.current_epoch = epoch
        self.total_loss = 0.0

        for mb in tqdm.tqdm(range(1, n_minibatch + 1), desc=prog_str, leave=False, dynamic_ncols=True):
            self.current_batch = mb

            # <<< MODIFIED: Updated the call to match the PubSampler's signature. >>>
            # The sampler internally knows the number of training basins, so we only need to pass the number of timesteps.
            dataset_sample = self.sampler.get_training_sample(dataset=self.train_dataset, nt=n_timesteps)

            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.total_loss += loss.item()

            if self.verbose:
                tqdm.tqdm.write(f"Epoch {epoch}, batch {mb} | loss: {loss.item()}")

        if self.use_scheduler:
            self.scheduler.step()

        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

        if epoch % self.config['train']['save_epoch'] == 0:
            self.model.save_model(epoch)
            save_train_state(
                self.config, epoch=epoch, optimizer=self.optimizer,
                scheduler=self.scheduler, clear_prior=True
            )

    def evaluate(self) -> None:
        """
        Run model evaluation for a PUB experiment.

        This method iterates through each basin in the validation set one by one,
        generates a prediction for its entire time series, and collects the results.
        """
        self.is_in_train = False

        # <<< MODIFIED: Complete rewrite of the evaluation logic for PUB. >>>

        # Get the list of basin indices that are held out for validation
        validation_indices = self.sampler.val_indices
        log.info(f"Validating Model on {len(validation_indices)} ungauged basins.")

        # This list will store the output dictionary for each validation basin
        all_basin_predictions = []
        observations = self.eval_dataset['target']

        # Loop through each validation basin individually
        for basin_idx in tqdm.tqdm(validation_indices, desc='Evaluating Basins', leave=False, dynamic_ncols=True):
            # 1. Get the complete data for one validation basin
            dataset_sample = self.sampler.get_validation_sample(
                self.eval_dataset,
                basin_idx,
            )

            # 2. Run the model forward for this single basin
            prediction_dict = self.model(dataset_sample, eval=True)

            # 3. Detach tensors and move to CPU to save memory
            model_name = self.config['delta_model']['phy_model']['model'][0]
            cleaned_prediction = {
                key: tensor.cpu().detach() for key, tensor in prediction_dict[model_name].items()
            }
            all_basin_predictions.append(cleaned_prediction)

        # The `all_basin_predictions` variable is now equivalent to the old `batch_predictions`
        # and can be passed to the metrics calculation function.
        self.predictions = self._batch_data(all_basin_predictions)

        # Save predictions and calculate metrics
        log.info("Saving model outputs + Calculating metrics")
        save_outputs(self.config, all_basin_predictions,
                     observations)  # Pass original targets for saving if needed

        # Calculate metrics using the collected predictions
        self.calc_metrics(all_basin_predictions, observations)

    # ... (inference, _batch_data, _forward_loop methods remain) ...
    # NOTE: The _forward_loop is no longer used by evaluate(). If you use inference(),
    # it may need a similar modification to work correctly for PUB.
    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['simulation']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(self, batch_list: list[dict[str, torch.Tensor]], target_key: str = None) -> Any:
        """Merge batch data into a single dictionary."""
        data = {}
        try:
            if target_key:
                # For PUB, each item in batch_list has a 'batch' dim of 1. Cat will merge them.
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()
            for key in batch_list[0].keys():
                # Assuming spatial dimension is always 1 for this kind of data
                dim = 1 if len(batch_list[0][key].shape) == 3 else 0
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(self, data: dict, batch_start: NDArray, batch_end: NDArray) -> list:
        """Forward loop for batched evaluation/inference. No longer used by evaluate()."""
        ...

    def calc_metrics(self, all_basin_predictions: list[dict[str, torch.Tensor]], observations: torch.Tensor) -> None:
        """Calculate and save model performance metrics for the validation basins."""
        target_name = self.config['train']['target'][0]
        predictions = self._batch_data(all_basin_predictions, target_name)

        # <<< MODIFIED: Select only the observations for the validation basins. >>>
        # This is crucial for a correct comparison.
        validation_indices = self.sampler.val_indices
        observations = self.eval_dataset['target'][:, validation_indices, :]

        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
        target = target[self.config['delta_model']['phy_model']['warm_up']:, :]

        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )
        metrics.dump_metrics(self.config['out_path'])

    # ... (_log_epoch_stats remains unchanged) ...
    def _log_epoch_stats(self, epoch: int, loss_dict: dict, n_minibatch: int, start_time: float) -> None:
        """Log statistics after each epoch."""
        avg_loss_dict = {key: value / n_minibatch for key, value in loss_dict.items()}
        loss_str = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_alloc = int(torch.cuda.memory_reserved(device=self.config['device']) / 1e6)
        log.info(f"Loss after epoch {epoch}: {loss_str}\n~ Runtime {elapsed:.2f} s, {mem_alloc} Mb reserved GPU memory")
