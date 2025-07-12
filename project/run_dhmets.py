import numpy as np
import sys
sys.path.append(r'E:\PaperCode\generic_deltamodel')

from dmg import ModelHandler
from dmg.core.utils import import_data_loader, import_trainer, print_config, set_randomseed
from example import load_config

#------------------------------------------#
# Define model settings here.
CONFIG_PATH = '../example/conf/config_dhmets.yaml'
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
print(f"Streamflow predictions for {output['hmets']['streamflow'].shape[0]} days and " \
      f"{output['hmets']['streamflow'].shape[1]} basins ~ \nShowing the first 5 days for " \
        f"first basin: \n {output['hmets']['streamflow'][:5,:1].cpu().detach().numpy().squeeze()}")

# 1. Load configuration dictionary of model parameters and options.
config = load_config(CONFIG_PATH)
config['mode'] = 'train'
print_config(config)

# Set random seed for reproducibility.
set_randomseed(config['random_seed'])

# 2. Initialize the differentiable HBV 1.1p model (LSTM + HBV 1.1p) with model handler.
model = ModelHandler(config, verbose=True)

# 3. Load and initialize a dataset dictionary of NN and HBV model inputs.
data_loader_cls = import_data_loader(config['data_loader'])
data_loader = data_loader_cls(config, test_split=True, overwrite=False)


# 4. Initialize trainer to handle model training.
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    train_dataset=data_loader.train_dataset,
    verbose=True
)

# 5. Start model training.
trainer.train()
print(f'Training complete. Model saved to \n{config['model_path']}')