import numpy as np
import sys
sys.path.append(r'E:\PaperCode\generic_deltamodel')

from dmg import ModelHandler
from dmg.core.data import txt_to_array
from dmg.core.post import plot_hydrograph
from dmg.core.utils import Dates, import_data_loader, print_config, set_randomseed, import_trainer
from hydrodl2.models.hbv import hbv
from example import load_config

#------------------------------------------#
# Define model settings here.
CONFIG_PATH = '../example/conf/config_dhbv.yaml'
#------------------------------------------#


# 1. Load configuration dictionary of model parameters and options.
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
config['train']['start_epoch'] = 40
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

# model evaluation
config['mode'] = 'test'
set_randomseed(config['random_seed'])

model = ModelHandler(config, verbose=True)

data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)

trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    eval_dataset=data_loader.eval_dataset,
    verbose=True,
)

print('Evaluating model...')
trainer.evaluate()
print(f'Metrics and predictions saved to \n{config['out_path']}')