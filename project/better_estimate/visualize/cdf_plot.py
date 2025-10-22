import os
import sys
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.getenv("PROJ_PATH"))
from dmg.core.data import load_json
from dmg.core.post import plot_cdf
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
hope_config = load_config(r'conf/config_dhbv_hopev1.yaml')
gru_config = load_config(r'conf/config_dhbv_gru.yaml')
transformer_config = load_config(r'conf/config_dhbv_transformer.yaml')

METRIC = 'kge'
metrics = []
# 1. Load the evaluation metrics.
for config in [lstm_config, gru_config, transformer_config, hope_config]:
    metrics.append(load_json(os.path.join(config['out_path'], 'metrics.json')))
model_labels = ['LSTM', "GRU", "Transformer", "S4D"]  # todo 先就用这个名字，后面写的时候再改
plot_config = {"xbounds": (0.0, 1),
               "ybounds": (0, 1),
               "show_arrow": True,
               "fontsize": 18,
               "legend_fontsize": 16,
               "ticksize": 18,
               "linewidth": 1.8,
               "show_count_label":True,
               "count_threshold":0.6,
               "axis_width": 2.0}
# 2. Plot the CDF for NSE.
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5),
                         constrained_layout=True)
plot_cdf(
    ax=axes[0],
    metrics=metrics,
    metric_names=["nse"],
    model_labels=model_labels,
    xlabel="nse".upper(),
    **plot_config
)
plot_cdf(
    ax=axes[1],
    metrics=metrics,
    metric_names=["kge"],
    model_labels=model_labels,
    xlabel="kge".upper(),
    **plot_config
)
fig.tight_layout()
fig.savefig(os.path.join(os.getenv("PROJ_PATH"), "project/better_estimate/visualize/figures/cdf_plot.png"))
