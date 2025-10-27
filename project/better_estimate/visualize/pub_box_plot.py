import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.data import load_json
from dmg.core.post.plot_statbox import plot_boxplots
from project.better_estimate import load_config

font_family = "Times New Roman"
plt.rcParams.update(
    {
        "font.family": font_family,
        "font.serif": [font_family],
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.it": font_family,
        "mathtext.bf": font_family,
        "axes.unicode_minus": False,
    }
)

lstm_config = load_config(r"conf/config_dhbv_pub_lstm.yaml")
hopev1_config = load_config(r"conf/config_dhbv_pub_hopev1.yaml")
lstm_criteria_path = os.path.join(lstm_config["out_path"], "metrics.json")
hopev1_criteria_path = os.path.join(hopev1_config["out_path"], "metrics.json")
lstm_df_list = []
hopev1_df_list = []
# basin_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
basin_ids = [11, 12, 13, 14, 15, 16, 17]
for i in basin_ids:
# for i in [11, 12, 13, 14, 15, 16, 17]:
    new_lstm_criteria_path = re.sub(r"pub_basin_\d+", f"pub_basin_{i}", lstm_criteria_path)
    new_hopev1_criteria_path = re.sub(r"pub_basin_\d+", f"pub_basin_{i}", hopev1_criteria_path)
    lstm_criteria_df = pd.DataFrame(load_json(new_lstm_criteria_path))
    hopev1_criteria_df = pd.DataFrame(load_json(new_hopev1_criteria_path))
    lstm_df_list.append(lstm_criteria_df)
    hopev1_df_list.append(hopev1_criteria_df)

fig, ax = plt.subplots(figsize=(9, 4))
group_labels = [f"PUB-{i+1}" for i in range(len(basin_ids))]
criterion = "kge"
plot_boxplots(
    lstm_df_list,
    hopev1_df_list,
    ax=ax,
    group_labels=group_labels,
    column_name=criterion,
    ylim=(0, 1),
    ylabel=criterion.upper(),
    legend_labels=["LSTM", "HOPEv1"],
)
fig.savefig(os.path.join(os.getenv("PROJ_PATH"), f"project/better_estimate/visualize/figures/pur_boxplot_{criterion}.png"), dpi=300)