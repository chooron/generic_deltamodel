import numpy as np
import sys
sys.path.append(r'E:\PaperCode\generic_deltamodel')

from dmg import ModelHandler
from dmg.core.data import txt_to_array
from dmg.core.post import plot_hydrograph
from dmg.core.utils import Dates, import_data_loader, print_config, set_randomseed
from hydrodl2.models.hbv import hbv
from example import load_config

#------------------------------------------#
# Define model settings here.
CONFIG_PATH = '../example/conf/config_dhbv.yaml'
#------------------------------------------#


# 1. Load configuration dictionary of model parameters and options.
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
print_config(config)

# Set random seed for reproducibility.
set_randomseed(config['random_seed'])

# 2. Initialize the differentiable HBV 1.1p model (LSTM + HBV 1.1p).
model = ModelHandler(config, verbose=True)

# 3. Load and initialize a dataset dictionary of NN and HBV model inputs.
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=False, overwrite=False)

# 4. Forward the model to get the predictions.
train_dataset = data_loader.train_dataset
output = model(
    data_loader.train_dataset,
    eval=True,
)

print("-------------\n")
print(f"Streamflow predictions for {output['Hbv']['streamflow'].shape[0]} days and "
      f"{output['Hbv']['streamflow'].shape[1]} basins ~ \nShowing the first 5 days for "
        f"first basin: \n {output['Hbv']['streamflow'][:5,:1].cpu().detach().numpy().squeeze()}")


# #------------------------------------------#
# # Define model settings here.
# CONFIG_PATH = r'../example/conf/config_dhmets.yaml'
# #------------------------------------------#

# #------------------------------------------#
# # Choose a basin by USGS gage ID to plot.
# GAGE_ID = 1022500
# TARGET = 'streamflow'

# # Resample to 3-day prediction. Options: 'D', 'W', 'M', 'Y'.
# RESAMPLE = '3D'

# # Set the paths to the gage ID lists...
# GAGE_ID_PATH = config['observations']['gage_info']  #./gage_id.npy
# GAGE_ID_531_PATH = config['observations']['subset_path']  #./531sub_id.txt
# #------------------------------------------#


# # 1. Get the streamflow predictions and daily timesteps of the prediction window.
# print(f"HBV states and fluxes: {list(output['Hbv_1_1p'].keys())} \n")

# pred = output['Hbv_1_1p'][TARGET]
# timesteps = Dates(config['simulation'], config['delta_model']['rho']).batch_daily_time_range

# # Remove warm-up period to match model output (see Note above.)
# timesteps = timesteps[config['delta_model']['phy_model']['warm_up']:]


# # 2. Load the gage ID lists and get the basin index.
# gage_ids = np.load(GAGE_ID_PATH, allow_pickle=True)
# gage_ids_531 = txt_to_array(GAGE_ID_531_PATH)

# print(f"First 20 available gage IDs: \n {gage_ids[:20]} \n")
# print(f"First 20 available gage IDs (531 subset): \n {gage_ids_531[:20]} \n")

# if config['observations']['name'] == 'camels_671':
#     if GAGE_ID in gage_ids:
#         basin_idx = list(gage_ids).index(GAGE_ID)
#     else:
#         raise ValueError(f"Basin with gage ID {GAGE_ID} not found in the CAMELS 671 dataset.")

# elif config['observations']['name'] == 'camels_531':
#     if GAGE_ID in gage_ids_531:
#         basin_idx = list(gage_ids_531).index(GAGE_ID)
#     else:
#         raise ValueError(f"Basin with gage ID {GAGE_ID} not found in the CAMELS 531 dataset.")
# else:
#     raise ValueError(f"Observation data supported: 'camels_671' or 'camels_531'. Got: {config['observations']}")


# # 3. Get the data for the chosen basin and plot.
# streamflow_pred_basin = pred[:, basin_idx].squeeze()

# plot_hydrograph(
#     timesteps,
#     streamflow_pred_basin,
#     resample=RESAMPLE,
#     title=f"Hydrograph for Gage ID {GAGE_ID}",
#     ylabel='Streamflow (mm/day)',
# )