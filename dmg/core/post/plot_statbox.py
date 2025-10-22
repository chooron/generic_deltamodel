import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional


def plot_boxplots(
        list1: List[pd.DataFrame],
        list2: List[pd.DataFrame],
        column_name: str,
        group_labels: List[str],
        ax: plt.Axes,
        list1_color: str = '#4C72B0',  # A modern blue
        list2_color: str = '#DD8452',  # A complementary orange
        ylabel: str = 'Value',
        ylim: Optional[Tuple[float, float]] = None,
        legend_labels: List[str] = ['Condition A', 'Condition B'],
        label_fontsize: int = 14,
        tick_fontsize: int = 12,
        xtick_rotation: int = 0
):
    """
    出版级风格箱线图（矩形版）
    - 对称坐标轴（四边都有）
    - 右下角 legend，带边框与背景
    """
    # --- 1. 输入检查 ---
    if len(list1) != len(list2):
        raise ValueError("Input lists 'list1' and 'list2' must have the same length.")
    if len(list1) != len(group_labels):
        raise ValueError("The length of 'group_labels' must match the length of the DataFrame lists.")
    if not legend_labels or len(legend_labels) != 2:
        raise ValueError("'legend_labels' must be a list containing exactly two string labels.")

    # --- 2. 准备数据 ---
    data_to_plot = []
    positions = []
    colors = [list1_color, list2_color]
    group_spacing = 3

    for i, (df1, df2) in enumerate(zip(list1, list2)):
        if column_name not in df1.columns or column_name not in df2.columns:
            print(f"Warning: Column '{column_name}' not found in DataFrame pair at index {i}. Skipping.")
            continue

        data_to_plot.append(df1[column_name].dropna())
        data_to_plot.append(df2[column_name].dropna())

        base_position = i * group_spacing
        positions.extend([base_position - 0.45, base_position + 0.45])

    if not data_to_plot:
        print("Warning: No data available for plotting.")
        return

    # --- 3. 绘制箱线图 ---
    boxprops = dict(linewidth=1.5, edgecolor='black')
    medianprops = dict(linewidth=2.5, color='#c44e52')  # red median
    whiskerprops = dict(linewidth=1.5, linestyle='-')
    capprops = dict(linewidth=1.5)
    flierprops = dict(marker='o', markerfacecolor='gray', markersize=4,
                      linestyle='none', alpha=0.5)

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops
    )

    # --- 4. 颜色设置 ---
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i % 2])
        box.set_alpha(0.9)

    # --- 5. X轴标签与分隔线 ---
    tick_positions = [i * group_spacing for i in range(len(group_labels))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels, rotation=xtick_rotation, ha='center')

    for i in range(len(group_labels) - 1):
        separator_pos = (tick_positions[i] + tick_positions[i + 1]) / 2
        ax.axvline(separator_pos, color='grey', linestyle=':', linewidth=1)

    # --- 6. 轴标签与字体 ---
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize, direction='out')

    if ylim is not None:
        ax.set_ylim(ylim)

    # --- 7. Legend ---
    legend_patches = [
        mpatches.Patch(facecolor=list1_color, label=legend_labels[0], edgecolor='black'),
        mpatches.Patch(facecolor=list2_color, label=legend_labels[1], edgecolor='black')
    ]
    legend = ax.legend(
        handles=legend_patches,
        fontsize=tick_fontsize,
        loc='lower right',
        frameon=True,
        facecolor='white',
        edgecolor='gray'
    )
    legend.get_frame().set_alpha(0.9)

    # --- 8. 对称坐标轴（四边） ---
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color('black')

    ax.set_xlim(min(positions) - 1, max(positions) + 1)

    plt.tight_layout()
