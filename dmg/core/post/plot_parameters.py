import matplotlib.pyplot as plt
import numpy as np


def plot_parameters(data, titles=None, gray_color='0.7', median_color='blue'):
    """
    绘制参数序列图，突出显示中位数成分。
    
    参数:
    data (np.array): 输入数据，期望 shape 为 (T, N, K)
                       T: 序列长度 (e.g., 730)
                       N: 成分数量 (e.g., 16)
                       K: 参数/类型数量 (e.g., 3)
    titles (list of str): K个子图的标题列表。
    gray_color (str): 16个成分的默认颜色。
    median_color (str): 中位数成分的突出显示颜色。
    """
    
    # 检查数据维度
    if data.ndim != 3:
        raise ValueError(f"Input data has {data.ndim} dimensions, but expected 3 (T, N, K).")
        
    T, N, K = data.shape
    
    # 如果没有提供标题，创建默认标题
    if titles is None:
        titles = [f'Parameter Type {k+1}' for k in range(K)]
    elif len(titles) != K:
        print(f"Warning: Provided {len(titles)} titles, but expected {K}. Using default titles.")
        titles = [f'Parameter Type {k+1}' for k in range(K)]

    # 创建 K 个垂直排列的子图
    # squeeze=False 确保 'axes' 总是一个数组，即使 K=1
    fig, axes = plt.subplots(K, 1, figsize=(14, 5 * K), squeeze=False)
    
    # 遍历 K 个参数类型
    for k in range(K):
        ax = axes[k, 0] # 获取当前子图
        
        # 提取当前参数的所有成分数据, shape (T, N)
        param_data = data[:, :, k]
        
        # 1. 计算每个成分的“时间平均值”
        temporal_averages = np.mean(param_data, axis=0) # shape (N,)
        
        # 2. 找到所有平均值的中位数
        median_avg_value = np.median(temporal_averages)
        
        # 3. 找到平均值最接近中位数的那个成分的索引
        median_component_index = np.argmin(np.abs(temporal_averages - median_avg_value))
        
        # 4. 提取中位数成分的序列数据
        median_sequence = param_data[:, median_component_index]
        
        # 5. 绘制所有 N (16) 个成分的灰色线条
        for i in range(N):
            ax.plot(param_data[:, i], color=gray_color, alpha=0.6, linewidth=1.5)
            
        # 6. 绘制中位数成分的彩色线条（使其在顶层）
        ax.plot(median_sequence, color=median_color, linewidth=2.5, 
                label=f'Median Component (Index {median_component_index})')
        
        # 7. 设置图表样式
        ax.set_title(titles[k], fontsize=14)
        ax.set_xlabel('Time Step (Sequence Length)')
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    # 调整布局
    plt.tight_layout()
    
    # 您可以保存图像或直接显示
    # plt.savefig('your_plot_filename.png')
    # plt.show()
    
    return fig, axes