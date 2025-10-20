import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.data import load_json
from dmg.core.post.plot_statbox import plot_boxplots
from project.better_estimate import load_config

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

lstm_config = load_config(r'conf/config_dhbv_lstm.yaml')
hopev1_config = load_config(r'conf/config_dhbv_hopev1.yaml')
lstm_criteria_path = os.path.join(lstm_config['out_path'], 'metrics.json')
hopev1_criteria_path = os.path.join(hopev1_config['out_path'], 'metrics.json')
lstm_criteria_df = pd.DataFrame(load_json(lstm_criteria_path))
hopev1_criteria_df = pd.DataFrame(load_json(hopev1_criteria_path))

fig, ax = plt.subplots(figsize=(10, 4))
group_labels = [f"sub-basin-{i}" for i in range(10)]
plot_boxplots([lstm_criteria_df] * 10, [hopev1_criteria_df] * 10, ax=ax,
              group_labels=group_labels, column_name='nse', ylim=(0, 1),
              ylabel='NSE')
plt.show()
