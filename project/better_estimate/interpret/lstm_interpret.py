import os
import sys

import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg import ModelHandler  # noqa
from dmg.core.data.loaders import HydroLoader  # noqa
from dmg.core.utils import import_trainer  # noqa
from project.better_estimate import load_config  # noqa

font_family = 'Times New Roman'
plt.rcParams.update({
    'font.family': font_family,
    'font.serif': [font_family],
    'mathtext.fontset': 'custom',
    'mathtext.rm': font_family,
    'mathtext.it': font_family,
    'mathtext.bf': font_family,
    'axes.unicode_minus': False,
})

config = load_config(r'conf/config_dhbv_lstm.yaml')
config['mode'] = 'test'
config['test']['test_epoch'] = 100
model = ModelHandler(config, verbose=True)
loader = HydroLoader(config, test_split=True, overwrite=False)
loader.load_dataset()
eval_dataset = loader.eval_dataset
trainer_cls = import_trainer(config['trainer'])
trainer = trainer_cls(
    config,
    model,
    eval_dataset=eval_dataset,
    verbose=True,
)
lstm_model = trainer.model.model_dict['Hbv_2'].nn_model
lstm_layer = lstm_model.lstminv
# print("LSTM model structure:", lstm_model)
# lstm_model.eval()
# test_data = (eval_dataset['xc_nn_norm'][0:730, :, :], eval_dataset['c_nn_norm'][:, :])
# print("Test data shapes:", test_data[0].shape, test_data[1].shape)
# est_output = lstm_model(*test_data)

ig = IntegratedGradients(lstm_layer)
attributions, delta = ig.attribute(eval_dataset['xc_nn_norm'][0:730, :, :], return_convergence_delta=True)