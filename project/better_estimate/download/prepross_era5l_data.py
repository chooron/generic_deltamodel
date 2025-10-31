import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

# --- 1. 定义文件和参数 ---

# 包含已生成 CSV 文件的目录
BASE_DATA_PATH = os.getenv("DATA_PATH") 

# !! 新文件：包含流域 ID 和对应土壤深度的 CSV 文件
# 您需要自己准备这个文件 (从 SHP 导出或从 CAMELS 属性表获取)
SOIL_DEPTH_CSV = os.path.join(BASE_DATA_PATH, "camels_soil.txt")

# 新 CSV 中流域 ID 和土壤深度的列名
WATERSHED_ID_COLUMN = "gauge_id"
SOIL_DEPTH_COLUMN = "soil_depth_statsgo" # 假设单位是米 (m)

# 年份范围 (必须与你生成的 CSV 文件名匹配)
YEAR_RANGE_STR = "1995-2010"

# ERA5-Land 土壤层边界定义 (单位: mm)
# (Top, Bottom)
LAYER_BOUNDS_MM = {
    1: (0, 70),
    2: (70, 280),
    3: (280, 1000),
    4: (1000, 2890)
}

# --- 2. 加载流域属性 (土壤深度) ---
print(f"正在从 {SOIL_DEPTH_CSV} 加载土壤深度...")
df_depth = pd.read_csv(SOIL_DEPTH_CSV, sep=';')

# 检查所需列
if WATERSHED_ID_COLUMN not in df_depth.columns or SOIL_DEPTH_COLUMN not in df_depth.columns:
    print(f"!! 错误: {SOIL_DEPTH_CSV} 必须包含 '{WATERSHED_ID_COLUMN}' 和 '{SOIL_DEPTH_COLUMN}' 列。")
    exit()

# 创建一个从 gage_id (字符串) 到土壤深度(mm)的映射
try:
    # 确保 gage_id 是字符串以便于映射
    df_depth[WATERSHED_ID_COLUMN] = df_depth[WATERSHED_ID_COLUMN].astype(str)
    # 转换土壤深度为数值，填充 NaN 为 0，并转换为 mm
    df_depth['soil_depth_mm'] = pd.to_numeric(df_depth[SOIL_DEPTH_COLUMN], errors='coerce').fillna(0) * 1000
    
    soil_depth_map = df_depth.set_index(WATERSHED_ID_COLUMN)['soil_depth_mm']
    print(f"成功加载并转换了 {len(soil_depth_map)} 个流域的土壤深度。")
    print(soil_depth_map.head())
except Exception as e:
    print(f"!! 错误: 处理土壤深度数据时出错: {e}")
    exit()


# --- 3. 处理蒸发 (e) 和雪深 (sd) ---
# e (蒸发): m -> mm (取绝对值)
# sd (雪深): m (水当量) -> mm (水当量)

for var_name in ['e', 'sd']:
    input_csv = os.path.join(BASE_DATA_PATH, 'era5l_data', f"output_{var_name}_{YEAR_RANGE_STR}_monthly.csv")
    output_csv = os.path.join(BASE_DATA_PATH, 'era5l_data', f"output_{var_name}_mm_{YEAR_RANGE_STR}_monthly.csv")
    
    if not os.path.exists(input_csv):
        print(f"\n!! 警告: 未找到文件 {input_csv}, 跳过 {var_name}")
        continue

    print(f"\n正在处理: {var_name}")
    df = pd.read_csv(input_csv, parse_dates=['time'])
    
    # 提取时间列和数据列
    time_col = df['time']
    data_cols = df.columns.drop('time')
    df_data = df[data_cols].copy()
    
    if var_name == 'e':
        print("    转换: m -> mm, 并取绝对值 (负号表示损失)")
        # 乘以 1000 并取绝对值。这就是您要的“蒸发量总值”的处理。
        df_data = df_data.abs() * 1000
    elif var_name == 'sd':
        print("    转换: m -> mm")
        # 乘以 1000
        df_data = df_data * 1000
    
    # 将时间列插回
    df_data.insert(0, 'time', time_col)
    df_data.to_csv(output_csv, index=False)
    print(f"    成功保存: {output_csv}")


# --- 4. 计算总土壤水 (swvl1, swvl2, swvl3, swvl4) ---
print("\n--- 开始处理总土壤水计算 ---")
try:
    print("正在加载所有 swvl (体积比) 数据...")
    # 将 time 设为索引以便于对齐, 确保 gage_id 列为字符串
    def load_swvl_csv(var_name):
        file_path = os.path.join(BASE_DATA_PATH, 'era5l_data', f"output_{var_name}_{YEAR_RANGE_STR}_monthly.csv")
        df = pd.read_csv(file_path, parse_dates=['time']).set_index('time')
        df.columns = df.columns.astype(str) # 确保列名(gage_id)是字符串
        return df

    df_swvl1 = load_swvl_csv("swvl1")
    df_swvl2 = load_swvl_csv("swvl2")
    df_swvl3 = load_swvl_csv("swvl3")
    df_swvl4 = load_swvl_csv("swvl4")
    print("所有 swvl 数据加载完毕。")
except FileNotFoundError as e:
    print(f"!! 错误: 缺少一个或多个 swvl CSV 文件。无法计算总土壤水。{e}")
    exit()

# 获取流域ID列 (假设所有文件的列都一致)
watershed_cols = df_swvl1.columns
# 创建一个空的 DataFrame 来存储结果 (单位: mm)
df_total_soil_water_mm = pd.DataFrame(index=df_swvl1.index, columns=watershed_cols, dtype=float)

print(f"正在为 {len(watershed_cols)} 个流域计算加权土壤水...")

# 遍历每一个流域 (每一列)
for gage_id in watershed_cols:
    
    if gage_id not in soil_depth_map:
        # 检查是否在土壤深度地图中
        print(f"    警告: 在 {SOIL_DEPTH_CSV} 中未找到 {gage_id} 的土壤深度, 该列将填充 NaN。")
        df_total_soil_water_mm[gage_id] = np.nan
        continue

    # 1. 获取该流域的最大土壤深度 (mm)
    max_depth_mm = soil_depth_map[gage_id]

    # 2. 根据最大深度计算每层的有效厚度 (mm)
    L1_thick = np.clip(max_depth_mm, LAYER_BOUNDS_MM[1][0], LAYER_BOUNDS_MM[1][1]) - LAYER_BOUNDS_MM[1][0]
    L2_thick = np.clip(max_depth_mm, LAYER_BOUNDS_MM[2][0], LAYER_BOUNDS_MM[2][1]) - LAYER_BOUNDS_MM[2][0]
    L3_thick = np.clip(max_depth_mm, LAYER_BOUNDS_MM[3][0], LAYER_BOUNDS_MM[3][1]) - LAYER_BOUNDS_MM[3][0]
    L4_thick = np.clip(max_depth_mm, LAYER_BOUNDS_MM[4][0], LAYER_BOUNDS_MM[4][1]) - LAYER_BOUNDS_MM[4][0]
    
    # 3. 计算总土壤水 (mm)
    total_sw_mm_series = (
        (df_swvl1[gage_id] * L1_thick) +
        (df_swvl2[gage_id] * L2_thick) +
        (df_swvl3[gage_id] * L3_thick) +
        (df_swvl4[gage_id] * L4_thick)
    )
    
    df_total_soil_water_mm[gage_id] = total_sw_mm_series

# --- 5. 保存总土壤水结果 ---
output_sw_csv = os.path.join(BASE_DATA_PATH, 'era5l_data',  f"output_total_soil_water_mm_{YEAR_RANGE_STR}_monthly.csv")
# 恢复 time 列
df_total_soil_water_mm.reset_index().to_csv(output_sw_csv, index=False, float_format='%.6f')

print(f"\n*** 成功计算并保存总土壤水: {output_sw_csv} ***")
print("--- 所有后处理已完成 ---")