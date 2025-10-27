import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def plot_baseflow_scatter(
    real_baseflow, 
    lstm_baseflow, 
    hopev1_baseflow, 
    figsize=(8, 8), 
    fontsize_labels=14, 
    fontsize_ticks=12,
    fontsize_legend=12,
    axis_linewidth=1.5,
    output_path=None
):
    """
    绘制观测与模拟基流指数的科研风格散点图。

    参数:
    real_baseflow (array-like): 真实的基流指数 (x轴)。
    lstm_baseflow (array-like): LSTM 模型的基流指数 (y轴)。
    hopev1_baseflow (array-like): Hopev1 模型的基流指数 (y轴)。
    figsize (tuple): 图像尺寸。
    fontsize_labels (int): 坐标轴标签的字体大小。
    fontsize_ticks (int): 坐标轴刻度的字体大小。
    fontsize_legend (int): 图例的字体大小。
    axis_linewidth (float): 坐标轴边框和刻度线的宽度。
    output_path (str, optional): 图像保存路径。如果为 None，则只显示图像。
    """
    
    # --- 1. 数据准备和相关性计算 ---
    # 确保输入为 numpy 数组
    real_baseflow = np.asarray(real_baseflow)
    lstm_baseflow = np.asarray(lstm_baseflow)
    hopev1_baseflow = np.asarray(hopev1_baseflow)
    
    # 计算皮尔逊相关系数 (R)
    # R 值和 p-value
    corr_lstm, _ = pearsonr(lstm_baseflow, real_baseflow)
    corr_hopev1, _ = pearsonr(hopev1_baseflow, real_baseflow)
    
    # --- 2. 创建画布和坐标轴 ---
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- 3. 绘制散点图 ---
    # LSTM
    ax.scatter(
        real_baseflow, 
        lstm_baseflow, 
        alpha=0.6, 
        color='blue', 
        marker='o',  # 圆形
        label=f'LSTM ($R$ = {corr_lstm:.2f})'
    )
    
    # Hopev1
    ax.scatter(
        real_baseflow, 
        hopev1_baseflow, 
        alpha=0.6, 
        color='red', 
        marker='s',  # 方形
        label=f'Hopev1 ($R$ = {corr_hopev1:.2f})'
    )
    
    # --- 4. 绘制 1:1 对角线 ---
    # 确定所有数据的范围
    all_data = np.concatenate([real_baseflow, lstm_baseflow, hopev1_baseflow])
    min_val = np.min(all_data) * 0.95
    max_val = np.max(all_data) * 1.05
    
    # 绘制 y=x 线
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')
    
    # --- 5. 设置标签和图例 ---
    ax.set_xlabel('Observed Baseflow Index', fontsize=fontsize_labels)
    ax.set_ylabel('Simulated Baseflow Index', fontsize=fontsize_labels)
    ax.legend(fontsize=fontsize_legend, frameon=False) # frameon=False 移除图例边框
    
    # --- 6. 设置坐标轴范围 ---
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    # --- 7. 应用科研绘图风格 ---
    
    # 设置所有四个边框的线宽
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)
        
    # 设置刻度
    ax.tick_params(
        axis='both',          # 同时应用于 x 和 y 轴
        which='major',        # 主刻度
        labelsize=fontsize_ticks, # 刻度标签字体大小
        direction='in',       # 刻度线朝内
        width=axis_linewidth, # 刻度线宽度
        length=6,             # 刻度线长度
        top=True,             # 显示顶部刻度线
        right=True            # 显示右侧刻度线
    )
    
    # (可选) 设置次刻度
    ax.tick_params(
        axis='both', 
        which='minor', 
        direction='in', 
        width=axis_linewidth * 0.7, 
        length=3,
        top=True,
        right=True
    )
    
    # 确保布局紧凑
    plt.tight_layout()
    
    # --- 8. 保存或显示 ---
    if output_path:
        print(f"Saving figure to {output_path}...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()