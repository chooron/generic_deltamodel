import json
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import sys
from dotenv import load_dotenv

from dmg.core.data.loaders import HydroLoader
from dmg.core.post.plot_line import plot_time_series

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import cartopy.crs as ccrs

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

from dmg.core.data import txt_to_array, load_json
from dmg.core.post import geoplot_single_metric
from project.better_estimate import load_config

# ------------------------------------------#
# Choose the metric to plot. (See available metrics printed above, or in the metrics_agg.json file).
METRIC = 'fhv_abs'


# Set the paths to the gage id lists and shapefiles...

def fetch_data(config):
    GAGE_ID_PATH = os.getenv("GAGE_INFO")  # ./gage_id.npy
    GAGE_ID_531_PATH = os.getenv("SUBSET_PATH")  # ./531sub_id.txt
    SHAPEFILE_PATH = os.path.join(os.getenv("PROJ_PATH"), "data\camels_loc\camels_671_loc.shp")
    # ------------------------------------------#

    # 1. Load gage ids + basin shapefile with geocoordinates (lat, long) for every gage.
    gage_ids = np.load(GAGE_ID_PATH, allow_pickle=True)
    gage_ids_531 = txt_to_array(GAGE_ID_531_PATH)
    coords = gpd.read_file(SHAPEFILE_PATH)

    # 2. Format geocoords for 531- and 671-basin CAMELS sets.
    coords_531 = coords[coords['gage_id'].isin(list(gage_ids_531))].copy()

    coords['gage_id'] = pd.Categorical(coords['gage_id'], categories=list(gage_ids), ordered=True)
    coords_531['gage_id'] = pd.Categorical(coords_531['gage_id'], categories=list(gage_ids_531), ordered=True)

    coords = coords.sort_values('gage_id')  # Sort to match order of metrics.
    basin_coords_531 = coords_531.sort_values('gage_id')

    # 3. Load the evaluation metrics.
    metrics_path = os.path.join(config['out_path'], 'metrics.json')
    metrics = load_json(metrics_path)

    # 4. Add the evaluation metrics to the basin shapefile.
    if config['observations']['name'] == 'camels_671':
        coords[METRIC] = metrics[METRIC]
        full_data = coords
    elif config['observations']['name'] == 'camels_531':
        coords_531[METRIC] = metrics[METRIC]
        full_data = coords_531
    else:
        raise ValueError(f"Observation data supported: 'camels_671' or 'camels_531'. Got: {config['observations']}")
    return full_data


lstm_config = load_config(r'conf/config_dhbv_lstm.yaml')
lstm_config['mode'] = 'test'
hopev1_config = load_config(r'conf/config_dhbv_hopev1.yaml')
hopev1_config['mode'] = 'test'

loader = HydroLoader(lstm_config, test_split=True, overwrite=False)
loader.load_dataset()

lstm_data = fetch_data(config=lstm_config)
hopev1_data = fetch_data(config=hopev1_config)
diff_data = hopev1_data[METRIC].values - lstm_data[METRIC].values
diff_df = hopev1_data.copy()
diff_df[METRIC] = diff_data
valid_ids = hopev1_data.loc[hopev1_data[METRIC] < 20.0, 'gage_id']
filtered = diff_df[diff_df['gage_id'].isin(valid_ids)]
gauge_ids = diff_df['gage_id'].values.tolist()
sorted_result = filtered.sort_values(by=METRIC, ascending=False)
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 9),
#                          subplot_kw={'projection': ccrs.Mercator()},
#                          constrained_layout=True)
# # 5. Plot the evaluation results spatially.
# geoplot_single_metric(lstm_data, METRIC, ax=axes[0], cmap='plasma')
# geoplot_single_metric(hopev1_data, METRIC, ax=axes[1], cmap='plasma')
# base_cmap = cm.get_cmap('RdBu')
# colors = base_cmap(np.linspace(0.05, 0.95, 256))  # 去掉最白部分
# custom_cmap = mcolors.LinearSegmentedColormap.from_list("RdBu_no_white", colors)
# geoplot_single_metric(diff_df, METRIC, ax=axes[2], vmin=-0.3, vmax=0.3, cmap=custom_cmap, marker='^')
# fig.savefig(os.path.join(os.getenv("PROJ_PATH"), "project/better_estimate/visualize/figures/spatial_plot.png"), dpi=300)


def plot_combine_line(ax, settings):
    print("=============")
    print(f"basin id: {sorted_result['gage_id'].values[settings['idx']]}")
    print(lstm_data.loc[lstm_data["gage_id"] == sorted_result['gage_id'].values[settings['idx']], METRIC])
    print(hopev1_data.loc[hopev1_data["gage_id"] == sorted_result['gage_id'].values[settings['idx']], METRIC])
    select_idx = gauge_ids.index(sorted_result['gage_id'].values[settings['idx']])
    target = loader.eval_dataset['target'].cpu().numpy()[:, select_idx, :]
    lstm_streamflow_pred = np.load(os.path.join(lstm_config['out_path'], "streamflow.npy"))[:, select_idx, :]
    hopev1_streamflow_pred = np.load(os.path.join(hopev1_config['out_path'], "streamflow.npy"))[:, select_idx, :]

    dates = pd.date_range(start=datetime(1995, 10, 1) + timedelta(365),
                          periods=len(lstm_streamflow_pred), freq='D')
    plot_data = {'target': target[365:365 + len(lstm_streamflow_pred)],
                 'LSTM': lstm_streamflow_pred,
                 'S4D': hopev1_streamflow_pred}

    plot_time_series(
        dates,
        plot_data,
        ax=ax,
        colors={'LSTM': '#0077BB', 'S4D': '#CC3311'},
        time_range=settings['time_range'],
        ylabel_fontsize=18,
        ylabel=settings['ylabel'],
        tick_fontsize=16,
        legend_fontsize=18,
    )


fig2, axes2 = plt.subplots(nrows=1, ncols=4,
                           figsize=(20, 5),
                           constrained_layout=True)
# 0: ('1999-01-01', '1999-12-31'), 1: ('2004-01-01', '2004-12-31'), 2: ('2003-01-01', '2003-12-31')
plot_combine_line(axes2[0], {'idx': 0, 'time_range': ('2007-03-01', '2007-12-01'), 'ylabel':"Streamflow (mm/day)"})
plot_combine_line(axes2[1], {'idx': 1, 'time_range': ('2002-01-01', '2002-12-01'), 'ylabel':None})
plot_combine_line(axes2[2], {'idx': -3, 'time_range': ('2008-01-01', '2008-12-01'), 'ylabel':None})
plot_combine_line(axes2[3], {'idx': -2, 'time_range': ('2002-01-01', '2003-01-01'), 'ylabel':None})
fig2.savefig(os.path.join(os.getenv("PROJ_PATH"), "project/better_estimate/visualize/figures/hydrograph_plot.png"), dpi=300)

# 11180960