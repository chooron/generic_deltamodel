import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


def plot_cdf(
        metrics: list[dict],
        metric_names: list[str],
        ax: plt.Axes,
        model_labels: list[str] = None,
        colors: list[str] = None,
        title: str = "",
        xlabel: str = "Metric Value",
        xbounds: tuple = None,
        ybounds: tuple = (0, 1.02),
        show_arrow: bool = False,
        show_median_label: bool = True,
        # --- NEW PARAMETERS ---
        show_count_label: bool = False,
        count_threshold: float = 0.6,
        # --- END NEW PARAMETERS ---
        fontsize: int = 14,
        ticksize: int = 12,
        legend_fontsize: int = 12,
        linewidth: float = 1.8,
        axis_width: float = 1.2,
):
    if not metrics or not metric_names:
        raise ValueError("错误：未提供指标数据或指标名称。")

    if model_labels is None:
        model_labels = [f"Model {i + 1}" for i in range(len(metrics))]
    elif len(model_labels) != len(metrics):
        raise ValueError("错误：模型标签数量必须与模型数量匹配。")

    if colors is None:
        colors = ['#0077BB', '#EE7733', '#009988', '#CC3311', '#33BBEE', '#EE3377', '#BBBBBB']
    color_cycle = cycle(colors)

    # --- NEW: List to store count strings ---
    count_info_lines = []

    for metric_name in metric_names:
        for i, model_data in enumerate(metrics):
            if metric_name in model_data and isinstance(model_data[metric_name], (list, np.ndarray)):
                values = np.array(model_data[metric_name])
                values = values[~np.isnan(values)]
                if values.size == 0:
                    print(f"警告：模型 {model_labels[i]} 的指标 '{metric_name}' 无有效数据。")
                    continue

                # --- NEW: Calculate count and store it ---
                count = np.sum(values > count_threshold)
                count_info_lines.append(f"{model_labels[i]}: {count}")
                # --- END NEW ---

                sorted_values = np.sort(values)
                cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                median_val = np.median(values)

                legend_label = f"{model_labels[i]} - {metric_name.upper()}$_{{50}}$: {median_val:.3f}"
                ax.plot(sorted_values, cdf, label=legend_label,
                        color=next(color_cycle), linewidth=linewidth)
            else:
                print(f"警告：模型 {model_labels[i]} 中未找到指标 '{metric_name}' 或格式无效。")

    ax.set_title(title, fontsize=fontsize, weight='bold')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel("CDF", fontsize=fontsize)

    if xbounds:
        ax.set_xlim(xbounds)
    ax.set_ylim(ybounds)

    ax.axhline(0.5, color='black', linewidth=axis_width, linestyle='--')
    if show_median_label:
        ax.text(ax.get_xlim()[1], 0.5, ' 0.5', va='center', ha='left', fontsize=ticksize, color='black')

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=legend_fontsize)

    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='#666666', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=ticksize, width=axis_width, )
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
        spine.set_color('black')

    # --- NEW: Code to add the annotation box ---
    if show_count_label:
        annotation_title = f"Count > {count_threshold}"
        annotation_text = f"$\\bf{{{annotation_title}}}$\n" + "\n".join(count_info_lines)
        ax.text(0.7, 0.05, annotation_text,
                transform=ax.transAxes,  # Use axis fraction coordinates (0,0 is bottom-left)
                fontsize=legend_fontsize - 2,
                verticalalignment='bottom',
                ha='left',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.7))
    # --- END NEW ---

    if show_arrow:
        ax.annotate("Better →", xy=(0.05, 0.1), xycoords='axes fraction',
                    fontsize=fontsize, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1, alpha=0.8))