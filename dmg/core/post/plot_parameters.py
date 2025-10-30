import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_parameters(data, titles=None, gray_color='0.7', median_color='blue', smooth_window=None):
    """
    绘制参数序列图，突出显示中位数成分，并可选择应用滑动平均平滑。
    
    参数:
    data (np.array): 输入数据，期望 shape 为 (T, K, N)
                     T: 序列长度
                     K: 参数/类型数量
                     N: 成分数量
    titles (list of str): K个子图的标题列表。
    gray_color (str): N个成分的默认颜色。
    median_color (str): 中位数成分的突出显示颜色。
    smooth_window (int, optional): 滑动平均的窗口大小。
                                   如果为 None 或 <= 1，则不进行平滑。
    """
    
    # 检查数据维度
    if data.ndim != 3:
        # 更新了错误消息以反映 (T, K, N)
        raise ValueError(f"Input data has {data.ndim} dimensions, but expected 3 (T, K, N).")
        
    # --- 关键修改：更改了维度的解包顺序 ---
    # 旧: T, N, K = data.shape
    T, K, N = data.shape
    
    # 如果没有提供标题，创建默认标题
    if titles is None:
        titles = [f'Parameter Type {k+1}' for k in range(K)]
    elif len(titles) != K:
        print(f"Warning: Provided {len(titles)} titles, but expected {K}. Using default titles.")
        titles = [f'Parameter Type {k+1}' for k in range(K)]

    # 创建 K 个垂直排列的子图
    fig, axes = plt.subplots(K, 1, figsize=(14, 5 * K), squeeze=False)
    
    # 遍历 K 个参数类型
    for k in range(K):
        ax = axes[k, 0] # 获取当前子图
        
        # --- 关键修改：更改了数据切片方式 ---
        # 提取当前参数的所有成分数据, 确保切片后 shape 仍为 (T, N)
        # 旧: param_data = data[:, :, k]
        param_data = data[:, k, :]
        
        # --- 平滑处理 (此部分逻辑不变) ---
        if smooth_window is not None and smooth_window > 1:
            df = pd.DataFrame(param_data)
            df_smooth = df.rolling(window=smooth_window, center=True, min_periods=1).mean()
            param_data = df_smooth.values
        # --- 结束平滑处理 ---

        # 1. 计算每个成分的“时间平均值”（此部分逻辑不变）
        temporal_averages = np.nanmean(param_data, axis=0) # shape (N,)
        
        # 2. 找到所有平均值的中位数（此部分逻辑不变）
        median_avg_value = np.median(temporal_averages)
        
        # 3. 找到平均值最接近中位数的那个成分的索引（此部分逻辑不变）
        median_component_index = np.argmin(np.abs(temporal_averages - median_avg_value))
        
        # 4. 提取中位数成分的序列数据（此部分逻辑不变）
        median_sequence = param_data[:, median_component_index]
        
        # 5. 绘制所有 N 个成分的灰色线条（此部分逻辑不变）
        ts = np.arange(T)
        for i in range(N):
            # ax.scatter(ts, param_data[:, i], color=gray_color, alpha=0.6, linewidth=1.5)
            ax.plot(param_data[:, i], color=gray_color, alpha=0.6, linewidth=1.5)
            
        # 6. 绘制中位数成分的彩色线条（此部分逻辑不变）
        ax.plot(median_sequence, color=median_color, linewidth=2.5, 
                  label=f'Median Component (Index {median_component_index})')
        
        # 7. 设置图表样式（此部分逻辑不变）
        ax.set_title(titles[k], fontsize=14)
        ax.set_xlabel('Time Step (Sequence Length)')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # 调整布局
    plt.tight_layout()
    
    return fig, axes

def plot_parameter_distributions(data, titles=None, bins=30, median_color='blue'):
    """
    绘制不同参数类型在时间维度上的分布直方图。

    参数:
    data (np.array): shape = (T, K, N)
                     T: 时间长度
                     K: 参数类型数量
                     N: 成分数量
    titles (list of str, optional): K个子图的标题。
    bins (int): 直方图分箱数。
    median_color (str): 中位数竖线的颜色。
    """
    
    # --- 数据检查 ---
    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (T, K, N), but got {data.shape}")
    
    T, K, N = data.shape
    
    # --- 处理标题 ---
    if titles is None:
        titles = [f'Parameter Type {k+1}' for k in range(K)]
    elif len(titles) != K:
        print(f"Warning: Provided {len(titles)} titles, expected {K}. Using default titles.")
        titles = [f'Parameter Type {k+1}' for k in range(K)]
    
    # --- 创建图像 ---
    fig, axes = plt.subplots(K, 1, figsize=(10, 3 * K), squeeze=False)
    
    for k in range(K):
        ax = axes[k, 0]
        
        # 提取当前参数类型所有时间和成分的值
        flat_data = data[:, k, :].flatten()
        flat_data = flat_data[~np.isnan(flat_data)]  # 去除 NaN
        
        # 绘制直方图
        ax.hist(flat_data, bins=bins, color='gray', alpha=0.7, edgecolor='black')
        
        # 绘制中位数竖线
        median_val = np.median(flat_data)
        ax.axvline(median_val, color=median_color, linestyle='--', linewidth=2,
                   label=f'Median = {median_val:.3f}')
        
        # 设置标题和样式
        ax.set_title(f'{titles[k]} - Value Distribution', fontsize=13)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    return fig, axes