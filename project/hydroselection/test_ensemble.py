import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import TensorDataset, DataLoader
from torchensemble import SoftGradientBoostingRegressor  # voting is a classic ensemble strategy


class ExpHydro(nn.Module):
    """
    ExpHydro model implementation as a PyTorch module.

    This class implements a simplified version of the ExpHydro hydrological model.
    The model parameters (f, Smax, Qmax, Df, Tmax, Tmin) are defined as
    torch.nn.Parameter, allowing them to be optimized within a PyTorch-based
    framework.

    The model simulates the water balance in a catchment, including processes
    like snow accumulation and melt, soil moisture, and runoff generation.
    """

    def __init__(self, f=1.0, Smax=100.0, Qmax=10.0, Df=2.0, Tmax=0.0, Tmin=-2.0):
        """
        Initializes the ExpHydro model with its parameters.

        The parameters are registered as torch.nn.Parameter to enable optimization.

        Args:
            f (float): A parameter controlling the shape of the runoff response.
            Smax (float): The maximum soil moisture storage.
            Qmax (float): The maximum quick flow.
            Df (float): The degree-day factor for snowmelt.
            Tmax (float): The temperature threshold for snowmelt.
            Tmin (float): The temperature threshold for snow/rain separation.
        """
        super(ExpHydro, self).__init__()
        # Register parameters for optimization
        self.f = nn.Parameter(torch.tensor(f, dtype=torch.float32))
        self.Smax = nn.Parameter(torch.tensor(Smax, dtype=torch.float32))
        self.Qmax = nn.Parameter(torch.tensor(Qmax, dtype=torch.float32))
        self.Df = nn.Parameter(torch.tensor(Df, dtype=torch.float32))
        self.Tmax = nn.Parameter(torch.tensor(Tmax, dtype=torch.float32))
        self.Tmin = nn.Parameter(torch.tensor(Tmin, dtype=torch.float32))

    def descale_parameters(self):
        f, Smax, Qmax, Df, Tmax, Tmin = (
            F.sigmoid(self.f) * 0.1,
            F.sigmoid(self.Smax) * 1400.0 + 100.0,
            F.sigmoid(self.Qmax) * 50.0 + 10.0,
            F.sigmoid(self.Df) * 5.0 + 0.01,
            F.sigmoid(self.Tmax) * 3.0,
            F.sigmoid(self.Tmin) * -3.0,
        )
        return f, Smax, Qmax, Df, Tmax, Tmin

    @staticmethod
    def step_fct(x):
        """A smoothed step function using tanh."""
        return (torch.tanh(5.0 * x) + 1.0) * 0.5

    @staticmethod
    def Ps(P, T, Tmin):
        """Calculates solid precipitation (snow)."""
        return ExpHydro.step_fct(Tmin - T) * P

    @staticmethod
    def Pr(P, T, Tmin):
        """Calculates liquid precipitation (rain)."""
        return ExpHydro.step_fct(T - Tmin) * P

    @staticmethod
    def M(S0, T, Tmax, Df):
        """Calculates snowmelt."""
        return ExpHydro.step_fct(T - Tmax) * ExpHydro.step_fct(S0) * torch.min(S0, Df * (T - Tmax))

    @staticmethod
    def ET(S1, Pet, Smax):
        """Calculates actual evapotranspiration."""
        return ExpHydro.step_fct(S1) * ExpHydro.step_fct(S1 - Smax) * Pet + \
            ExpHydro.step_fct(S1) * ExpHydro.step_fct(Smax - S1) * Pet * (S1 / Smax)

    @staticmethod
    def Qb(S1, Smax, Qmax, f):
        """Calculates baseflow."""
        return ExpHydro.step_fct(S1) * ExpHydro.step_fct(S1 - Smax) * Qmax + ExpHydro.step_fct(S1) * ExpHydro.step_fct(
            Smax - S1) * Qmax * torch.exp(-f * (Smax - S1))

    @staticmethod
    def Qs(S1, Smax):
        """Calculates surface flow (quick flow)."""
        return ExpHydro.step_fct(S1) * ExpHydro.step_fct(S1 - Smax) * (S1 - Smax)

    def forward(self, x):
        """
        Runs the model forward in time.

        This method implements a custom loop to simulate the model's behavior
        over a series of time steps. It updates the model's internal states
        (snow pack and soil moisture) and calculates the output discharge.

        Args:
            x (torch.Tensor): A tensor of input data. For training with a
                DataLoader, this will typically have a batch dimension, e.g.,
                (batch_size, time_steps, num_features).

        Returns:
            torch.Tensor: A time series of the calculated total discharge (Q_out),
                with a batch dimension, e.g., (batch_size, time_steps).
        """
        # If the input is batched, loop over the batch dimension.
        # This model's state logic is for a single sequence.
        if x.dim() == 3:
            outputs = [self.forward(x_i) for x_i in x]
            return torch.cat(outputs, dim=0)

        # From here, x is a single sequence: (time_steps, num_features)
        precp_series, temp_series, pet_series = x[:, 0], x[:, 1], x[:, 2]
        S1, S2 = torch.tensor(0.0), torch.tensor(0.0)
        Q_out_list = []
        f, Smax, Qmax, Df, Tmax, Tmin = self.descale_parameters()

        # Loop through the time series
        for i in range(len(precp_series)):
            precp = precp_series[i]
            temp = temp_series[i]
            pet = pet_series[i]

            # Calculate output flow from the current state (before update)
            Q_out = ExpHydro.Qb(S2, Smax, Qmax, f) + ExpHydro.Qs(S2, Smax)

            # Calculate state changes based on fluxes (logic from exp_hydro_single_step)
            dS1 = ExpHydro.Ps(precp, temp, Tmin) - ExpHydro.M(S1, temp, Tmax, Df)
            dS2 = ExpHydro.Pr(precp, temp, Tmin) + ExpHydro.M(S1, temp, Tmax, Df) - ExpHydro.ET(S2, pet, Smax) - Q_out

            # Update states for the next time step (Euler integration)
            S1 = S1 + dS1
            S2 = S2 + dS2

            # Store the calculated discharge for this time step
            Q_out_list.append(Q_out)

        # Combine the list of outputs into a single tensor
        output = torch.stack(Q_out_list)

        # Add a batch dimension for consistency, so output is (1, time_steps)
        return output.unsqueeze(0)


data_path = r"E:\PaperCode\generic_deltamodel\data\camels_data\camels_dataset"
with open(data_path, 'rb') as f:
    forcings, target, attributes = pickle.load(f)

# Convert numpy arrays to torch tensors
test_basin_data = torch.from_numpy(forcings[1, :, :]).float()
test_basin_target = torch.from_numpy(target[1, :, 0]).float()

# The model processes a single time series at a time. To use a DataLoader,
# we define a "sample" as the entire time series for one basin.
# We add a batch dimension of 1 to our data.
# Input shape: (batch_size=1, time_steps, num_features)
# Target shape: (batch_size=1, time_steps)
test_basin_data = test_basin_data.unsqueeze(0)
test_basin_target = test_basin_target.unsqueeze(0)

# # Create a dataset and dataloader.
# # As requested, the entire dataset is treated as a single item in one batch.
# dataset = TensorDataset(test_basin_data, test_basin_target)
# train_loader = DataLoader(dataset, batch_size=1)
#
# # Set up the ensemble model
# ensemble_model = SoftGradientBoostingRegressor(
#     estimator=ExpHydro,
#     n_estimators=10,
#     cuda=False
# )
# ensemble_model.set_optimizer(
#     "Adam",  # type of parameter optimizer
#     lr=1e-3,  # learning rate of parameter optimizer
#     weight_decay=1e-4,  # weight decay of parameter optimizer
# )
# # Train the ensemble model
# # The fit method requires a DataLoader
# ensemble_model.fit(
#     train_loader=train_loader,  # training data
#     epochs=100
# )  # the number of training epochs