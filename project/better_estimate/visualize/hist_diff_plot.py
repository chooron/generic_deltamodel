import json
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import os
import sys
from dotenv import load_dotenv

from dmg.core.data.loaders import HydroLoader
from dmg.core.post.plot_diff_hist import plot_distribution_curve

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
lstm_config = load_config(r'conf/config_dhbv_lstm.yaml')
lstm_config['mode'] = 'test'
hopev1_config = load_config(r'conf/config_dhbv_hopev1.yaml')
hopev1_config['mode'] = 'test'

loader = HydroLoader(lstm_config, test_split=True, overwrite=False)
loader.load_dataset()


# Set the paths to the gage id lists and shapefiles...

def fetch_data(config, metric):
    GAGE_ID_PATH = os.getenv("GAGE_INFO")  # ./gage_id.npy
    GAGE_ID_531_PATH = os.getenv("SUBSET_PATH")  # ./531sub_id.txt
    SHAPEFILE_PATH = r'E:\PaperCode\dpl-project\generic_deltamodel\data\camels_data\camels_loc\camels_671_loc.shp'
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
        coords[metric] = metrics[metric]
        full_data = coords
    elif config['observations']['name'] == 'camels_531':
        coords_531[metric] = metrics[metric]
        full_data = coords_531
    else:
        raise ValueError(f"Observation data supported: 'camels_671' or 'camels_531'. Got: {config['observations']}")
    return full_data


def load_diff_data(metric):
    lstm_data = fetch_data(config=lstm_config, metric=metric)
    hopev1_data = fetch_data(config=hopev1_config, metric=metric)
    diff_data = hopev1_data[metric].values - lstm_data[metric].values
    diff_df = hopev1_data.copy()
    diff_df[metric] = diff_data
    return diff_df


nse_diff = load_diff_data('nse')
nse_diff_values = nse_diff['nse'].values
nse_diff_values = np.where(nse_diff_values > 0.2, 0.2, nse_diff_values)
nse_diff_values = np.where(nse_diff_values < -0.2, -0.2, nse_diff_values)
kge_diff = load_diff_data('kge')
kge_diff_values = kge_diff['kge'].values
kge_diff_values = np.where(kge_diff_values > 0.2, 0.2, kge_diff_values)
kge_diff_values = np.where(kge_diff_values < -0.2, -0.2, kge_diff_values)

fhv_abs_diff = load_diff_data('fhv_abs')
fhv_abs_diff_values = fhv_abs_diff['fhv_abs'].values
fhv_abs_diff_values = np.where(fhv_abs_diff_values > 20, 20, fhv_abs_diff_values)
fhv_abs_diff_values = np.where(fhv_abs_diff_values < -20, 20, fhv_abs_diff_values)

kge_diff_values = np.where(kge_diff_values > 0.2, 0.2, kge_diff_values)
kge_diff_values = np.where(kge_diff_values < -0.2, -0.2, kge_diff_values)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

plot_distribution_curve(
    ax=ax1,
    data=nse_diff_values,
    data_name='NSE Difference',
    axislabel_fontsize=18,
    legend_fontsize=18,
    text_fontsize=18,
    bins=20
)
plot_distribution_curve(
    ax=ax2,
    data=kge_diff_values,
    data_name='KGE Difference',
    axislabel_fontsize=18,
    legend_fontsize=18,
    text_fontsize=18,
    bins=20
)
plot_distribution_curve(
    ax=ax3,
    data=fhv_abs_diff_values,
    data_name='|FHV| Difference',
    axislabel_fontsize=18,
    legend_fontsize=18,
    text_fontsize=18,
    line1=-5,
    line2=5,
    bins=20
)
fig.savefig(os.path.join(os.getenv("PROJ_PATH"), "project/better_estimate/visualize/figures/diff_hist_plot.png"), dpi=300)
