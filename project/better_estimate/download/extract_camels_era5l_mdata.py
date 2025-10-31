import xarray as xr
import geopandas as gpd
import pandas as pd
import rioxarray  # 自动添加 .rio 扩展
import zipfile
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. 定义核心参数 ---
BASE_PATH = r"E:\ERA5LAND_CAMELS_MONTHLY"
SHP_FILE = os.path.join(
    os.getenv("DATA_PATH"), "camels_loc", "camels_671_loc.shp"
)
GRIB_FILE_IN_ZIP = "data.grib"
EXTRACTED_GRIB_FILE = "temp_extracted.grib"
WATERSHED_ID_COLUMN = "gage_id"
TARGET_CRS = "EPSG:4326"  # 假设所有 GRIB 文件的 CRS 都是 WGS84

# --- 2. 定义循环参数 ---
# 定义要处理的年份范围 (包含 2000 和 2010)
YEARS_TO_PROCESS = range(1995, 2000) 
# 定义要提取的变量列表
VARIABLES_TO_PROCESS = ["swvl1", "swvl2", "swvl3", "swvl4", "sd", "e"]


def process_single_file(year, variable_name, base_path, gdf_projected):
    """
    处理单个年份和单个变量的 GRIB 文件, 提取流域均值。
    
    Args:
        year (int): 要处理的年份。
        variable_name (str): GRIB 变量的短名称 (e.g., 'swvl1')。
        base_path (str): GRIB 压缩文件的根目录。
        gdf_projected (gpd.GeoDataFrame): 预先投影到 GRIB CRS 的 GeoDataFrame。

    Returns:
        pd.DataFrame: 包含该年份时间序列的 DataFrame (time, gage_id_1, ...), 
                      如果文件不存在或出错则返回 None。
    """
    
    # --- 3. 解压缩 GRIB 文件 ---
    zip_file_path = os.path.join(base_path, f"era5land_monthly_{year}.zip")
    
    if not os.path.exists(zip_file_path):
        print(f"    !! 警告: 文件未找到, 跳过: {zip_file_path}")
        return None

    print(f"    正在从 {zip_file_path} 中解压 {GRIB_FILE_IN_ZIP}...")
    try:
        with zipfile.ZipFile(zip_file_path, "r") as z:
            z.extract(GRIB_FILE_IN_ZIP)
            # 重命名为统一的临时文件名,如果已存在则覆盖
            if os.path.exists(EXTRACTED_GRIB_FILE):
                os.remove(EXTRACTED_GRIB_FILE)
            os.rename(GRIB_FILE_IN_ZIP, EXTRACTED_GRIB_FILE)
    except Exception as e:
        print(f"    !! 错误: 解压文件 {zip_file_path} 失败: {e}")
        return None

    # --- 4. 加载数据 (只加载特定变量) ---
    print(f"    正在加载 GRIB: {EXTRACTED_GRIB_FILE} (变量: {variable_name})...")
    try:
        ds = xr.open_dataset(
            EXTRACTED_GRIB_FILE,
            engine="cfgrib",
            backend_kwargs={'filter_by_keys': {'shortName': variable_name}}
        )
        
        # 选取数据变量
        data_array = ds[variable_name]
        print(f"    已加载 {variable_name} (维度: {data_array.dims})")

    except Exception as e:
        print(f"    !! 错误: 加载 GRIB 或变量 {variable_name} 失败: {e}")
        if os.path.exists(EXTRACTED_GRIB_FILE):
            os.remove(EXTRACTED_GRIB_FILE)
        return None

    # --- 5. 统一坐标参考系 (CRS) ---
    if data_array.rio.crs is None:
        data_array = data_array.rio.set_spatial_dims(
            x_dim="longitude", y_dim="latitude"
        )
        data_array = data_array.rio.set_crs(TARGET_CRS)

    # --- 6. 循环提取并计算每个时间步的空间平均值 ---
    results_dict = {}  # 用字典存储结果: {gage_id: [time_series_values]}
    
    for index, row in gdf_projected.iterrows():
        watershed_id = row[WATERSHED_ID_COLUMN]
        geometry = [row.geometry]  # 裁剪需要一个几何列表

        # 核心步骤: 裁剪
        clipped_data = data_array.rio.clip(geometry, all_touched=True, drop=True)
        
        # 计算每个时间步的空间平均值
        spatial_dims = [dim for dim in clipped_data.dims if dim in ['latitude', 'longitude', 'lat', 'lon', 'y', 'x']]
        mean_timeseries = clipped_data.mean(dim=spatial_dims, skipna=True)
        
        results_dict[watershed_id] = mean_timeseries.values.tolist()

    # --- 7. 构建宽格式 DataFrame (时间 × 流域ID) ---
    time_values = data_array.coords['time'].values if 'time' in data_array.coords else data_array.coords['valid_time'].values
    df_results = pd.DataFrame(results_dict)
    df_results.insert(0, 'time', time_values)

    # --- 9. 清理临时文件 ---
    ds.close() # 确保文件已关闭
    os.remove(EXTRACTED_GRIB_FILE)
    print(f"    已删除临时文件: {EXTRACTED_GRIB_FILE}")

    return df_results


# ===============================================
# --- 主执行脚本 ---
# ===============================================

print("--- 脚本开始执行 ---")

# --- A. 仅加载和投影一次 Shapefile ---
print(f"正在加载 Shapefile: {SHP_FILE}...")
gdf_original = gpd.read_file(SHP_FILE)
print(f"Shapefile CRS: {gdf_original.crs}. 正在预转换为: {TARGET_CRS}")

# 将 Shapefile 转换为 GRIB 的 CRS (仅执行一次)
gdf_projected = gdf_original.to_crs(TARGET_CRS)
print(f"Shapefile 转换完成. {len(gdf_projected)} 个流域待处理.")
print(gdf_projected.head())
print("\n" + "="*30 + "\n")


# --- B. 遍历所有变量和所有年份 ---
year_range_str = f"{min(YEARS_TO_PROCESS)}-{max(YEARS_TO_PROCESS)}"

# 外循环: 变量
for var_name in VARIABLES_TO_PROCESS:
    print(f"--- 开始处理变量: {var_name} (年份: {year_range_str}) ---")
    
    all_yearly_dfs = []  # 用于存储每年的 DataFrame

    # 内循环: 年份
    for year in YEARS_TO_PROCESS:
        print(f"\n  -- 正在处理: {var_name} - {year} --")
        
        df_year = process_single_file(
            year=year,
            variable_name=var_name,
            base_path=BASE_PATH,
            gdf_projected=gdf_projected
        )
        
        if df_year is not None and not df_year.empty:
            all_yearly_dfs.append(df_year)
        else:
            print(f"    警告: 未能处理 {var_name} - {year} 的数据或数据为空.")
    
    # --- C. 合并并保存当前变量的所有年份数据 ---
    if not all_yearly_dfs:
        print(f"\n!! 变量 {var_name} 未找到任何有效数据, 跳过保存.")
        print("\n" + "="*30 + "\n")
        continue

    print(f"\n正在合并 {var_name} 的 {len(all_yearly_dfs)} 年数据...")
    
    # 纵向合并所有 DataFrame
    combined_df = pd.concat(all_yearly_dfs, ignore_index=True)
    
    # 确保时间戳是日期时间对象并排序
    combined_df['time'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.sort_values(by='time').reset_index(drop=True)

    # --- D. 保存到 CSV ---
    CSV_OUTPUT_FILE = f"output_{var_name}_{year_range_str}_monthly.csv"
    combined_df.to_csv(CSV_OUTPUT_FILE, index=False)
    
    print(f"\n*** 成功保存: {CSV_OUTPUT_FILE} ***")
    print(f"    合并后表格维度: {combined_df.shape}")
    print("\n" + "="*30 + "\n")

print("--- 所有处理已完成 ---")