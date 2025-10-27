import os
import re
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getenv("PROJ_PATH"))

from dmg.core.post.plot_baseflowindex_scatter import \
    plot_baseflow_scatter  # noqa
from project.better_estimate import load_config  # noqa

lstm_config = load_config(r"conf/config_dhbv_pub_lstm.yaml")
hopev1_config = load_config(r"conf/config_dhbv_pub_hopev1.yaml")
lstm_out_path = lstm_config["out_path"]
hopev1_out_path = hopev1_config["out_path"]
basin_group_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# basin_group_ids = [11, 12, 13, 14, 15, 16, 17]
all_basin_ids_list = []
lstm_bfi_list = []
hopev1_bfi_list = []

for i in basin_group_ids:
    print(f"Processing group {i}...")
    new_lstm_criteria_path = re.sub(r"pub_basin_\d+", f"pub_basin_{i}", lstm_out_path)
    new_hopev1_criteria_path = re.sub(r"pub_basin_\d+", f"pub_basin_{i}", hopev1_out_path)

    sub_group_ids = np.load(
        os.path.join(os.getenv("DATA_PATH"), "basin_groups", f"group_{i}.npy"), allow_pickle=True
    )
    all_basin_ids_list.append(sub_group_ids)

    # --- 修正 1: 使用 new_..._path 来加载数据 ---
    lstm_gwflow = np.load(os.path.join(new_lstm_criteria_path, "gwflow.npy"), allow_pickle=True)
    hopev1_gwflow = np.load(os.path.join(new_hopev1_criteria_path, "gwflow.npy"), allow_pickle=True)
    lstm_flow = np.load(os.path.join(new_lstm_criteria_path, "streamflow.npy"), allow_pickle=True)
    hopev1_flow = np.load(
        os.path.join(new_hopev1_criteria_path, "streamflow.npy"), allow_pickle=True
    )

    # 假设这里的 [:, 0] 是正确的（例如，原始数据是 (time, basins, 1)）
    lstm_baseflow_index = (lstm_gwflow.sum(axis=0) / lstm_flow.sum(axis=0))[:, 0]
    hopev1_baseflow_index = (hopev1_gwflow.sum(axis=0) / hopev1_flow.sum(axis=0))[:, 0]

    lstm_bfi_list.append(lstm_baseflow_index)
    hopev1_bfi_list.append(hopev1_baseflow_index)

print("Data loading complete.")

# --- 修正 2: 将列表合并 (concatenate) 为单个 NumPy 数组 ---
all_basin_ids = np.concatenate(all_basin_ids_list)
lstm_bfi = np.concatenate(lstm_bfi_list)
hopev1_bfi = np.concatenate(hopev1_bfi_list)

# --- 修正 3: 正确索引 Pandas DataFrame 以保留顺序和重复值 ---
gage_ids = np.load(os.path.join(os.getenv("DATA_PATH"), "gage_id.npy"), allow_pickle=True)
hydroinfos = pd.read_csv(os.path.join(os.getenv("DATA_PATH"), "camels_hydro.txt"), sep=";")

# 关键步骤：将 'gauge_id' 设为索引，以便进行高效和有序的查找
hydroinfos_indexed = hydroinfos.set_index("gauge_id")

# 使用 .loc[] 按 all_basin_ids 中的顺序（包括重复项）提取行
# .loc[] 会完全按照您提供的 ID 列表的顺序返回数据
selected_hydroinfos = hydroinfos_indexed.loc[all_basin_ids]

# 将 'baseflow_index' 列提取为 NumPy 数组
real_baseflow_index = selected_hydroinfos.baseflow_index.values
print("Real baseflow index sample:", real_baseflow_index.shape)
print("Real baseflow index sample:", lstm_bfi.shape)
print("Real baseflow index sample:", hopev1_bfi.shape)

plot_baseflow_scatter(
    real_baseflow_index,
    lstm_bfi,
    hopev1_bfi,
)
