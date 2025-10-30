import os
import ee
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
import logging
import time

# === 设置日志 ===
logging.basicConfig(
    filename='era5l_download.log',
    filemode='w',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# === 加载环境变量 ===
load_dotenv()

# === 初始化 GEE ===
try:
    ee.Initialize(project="banded-charmer-465108-k7")
    logging.info("✅ 成功初始化 Google Earth Engine")
except Exception as e:
    logging.error("❌ 初始化失败: %s", e)
    raise SystemExit(e)

# === 1. 读取流域 shapefile ===
shapefile = os.path.join(os.getenv("DATA_PATH"), 'camels_loc', 'camels_671_loc.shp')
gdf = gpd.read_file(shapefile)
logging.info("已加载 shapefile，共 %d 个流域", len(gdf))

if 'basin_id' not in gdf.columns:
    gdf['basin_id'] = range(1, len(gdf) + 1)

# === 2. 加载 ERA5-Land 数据 ===
dataset = (
    ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate('2000-01-01', '2001-12-31')
    .select('total_evaporation_sum')
)
logging.info("已加载 ERA5-Land 数据集：2000–2001 年，共 %d 张影像", dataset.size().getInfo())

# === 3. 日期序列 ===
dates = pd.date_range('2000-01-01', '2001-12-31', freq='D')

# === 4. 定义提取函数 ===
def extract_timeseries(geometry, basin_id):
    results = []
    for date in tqdm(dates, desc=f'Basin {basin_id}', leave=False, ncols=80):
        start = ee.Date(str(date.date()))
        end = start.advance(1, 'day')  # 下一天
        try:
            img = dataset.filterDate(start, end).first()
            if img is None:
                continue
            mean = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10000,
                maxPixels=1e13
            ).get('total_evaporation_sum')
            val = mean.getInfo()
        except Exception as e:
            logging.warning(f"⚠️ Basin {basin_id} | {date.date()} 提取失败: {e}")
            val = None
        results.append({'basin_id': basin_id, 'date': date.strftime('%Y-%m-%d'), 'evap_m': val})
    return results
# === 5. 主循环 ===
all_records = []
start_time = time.time()

for i, row in tqdm(list(gdf.iterrows()), total=len(gdf), desc="总体进度", ncols=100):
    basin_id = row['basin_id']
    geom = ee.Geometry(row.geometry.__geo_interface__)
    logging.info(f"开始处理流域 {basin_id} ({i+1}/{len(gdf)})")
    basin_data = extract_timeseries(geom, basin_id)
    all_records.extend(basin_data)
    logging.info(f"流域 {basin_id} 提取完成，共 {len(basin_data)} 条记录")

# === 6. 输出 ===
df = pd.DataFrame(all_records)
df['evap_mm'] = pd.to_numeric(df['evap_m'], errors='coerce') * 1000
df.to_csv('ERA5L_evaporation_CAMELS_2000_2001.csv', index=False)

elapsed = (time.time() - start_time) / 60
logging.info("✅ 全部完成，输出文件：ERA5L_evaporation_CAMELS_2000_2001.csv")
logging.info("总耗时：%.2f 分钟", elapsed)
