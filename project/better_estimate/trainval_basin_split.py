import numpy as np
import os

# --- 1. 配置参数 ---

# 源文件路径
SOURCE_FILE_PATH = r"E:\PaperCode\dpl-project\generic_deltamodel\data\camels_data\gage_id.npy"

# 输出文件夹路径 (将在源文件同目录下创建)
OUTPUT_DIRECTORY = os.path.join(os.path.dirname(SOURCE_FILE_PATH), "basin_groups")

# 要划分的分组数量
NUM_GROUPS = 10

# 随机数种子，用于保证每次划分结果一致，便于复现实验
RANDOM_SEED = 42


# --- 2. 主逻辑 ---

def create_basin_groups():
    """
    加载流域ID，随机打乱并分成N组，然后保存到文件。
    """
    # 检查源文件是否存在
    if not os.path.exists(SOURCE_FILE_PATH):
        print(f"错误: 源文件未找到，请检查路径: {SOURCE_FILE_PATH}")
        return

    # 加载流域ID
    try:
        basin_ids = np.load(SOURCE_FILE_PATH)
        print(f"成功加载 {len(basin_ids)} 个流域ID。")
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return

    # 设置随机数种子以保证结果可复现
    np.random.seed(RANDOM_SEED)
    print(f"使用随机数种子: {RANDOM_SEED}")

    # 创建一个打乱顺序的ID副本
    shuffled_ids = basin_ids.copy()
    np.random.shuffle(shuffled_ids)
    print("已将流域ID随机打乱。")

    # 将打乱后的ID分割成N组
    # np.array_split 适用于总数无法被N整除的情况
    basin_groups = np.array_split(shuffled_ids, NUM_GROUPS)
    print(f"已将ID分割成 {NUM_GROUPS} 组。")

    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"结果将保存在目录: {OUTPUT_DIRECTORY}")

    # 循环保存每一组到单独的 .npy 文件
    for i, group in enumerate(basin_groups):
        output_filename = f"group_{i}.npy"
        output_filepath = os.path.join(OUTPUT_DIRECTORY, output_filename)
        np.save(output_filepath, group)
        print(f"  -> 已保存第 {i} 组 ({len(group)} 个流域) 到 {output_filename}")

    print("\n--- 操作完成 ---")
    print("所有流域分组均已成功创建并保存。")


if __name__ == "__main__":
    create_basin_groups()