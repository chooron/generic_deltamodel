import cdsapi
import os

# === 配置输出路径 ===
out_dir = r"E:\ERA5LAND_CAMELS_MONTHLY"  # 修改为你的目标路径
os.makedirs(out_dir, exist_ok=True)

# === 数据集名称 ===
dataset = "reanalysis-era5-land-monthly-means"

# === 公共请求参数 ===
base_request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "snow_depth_water_equivalent",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        "total_evaporation"
    ],
    "month": [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ],
    "data_format": "grib",
    "download_format": "zip",
    "time": ["00:00"],
    "area": [49, -125, 25, -66],  # 北, 西, 南, 东
}

# === 初始化客户端 ===
client = cdsapi.Client()

# === 遍历 2000–2010 年 ===
for year in range(1995, 2000):
    request = base_request.copy()
    request["year"] = [str(year)]

    target_file = os.path.join(out_dir, f"era5land_monthly_{year}.zip")
    print(f"▶ 正在下载 {year} 年数据 -> {target_file}")

    client.retrieve(dataset, request).download(target_file)
    print(f"✅ 完成 {year} 年数据下载\n")

print("全部下载完成。")
