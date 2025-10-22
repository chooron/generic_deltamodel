import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde

def plot_distribution_curve(ax, data, data_name='Value', title_fontsize=16, axislabel_fontsize=12, legend_fontsize=10, text_fontsize=12, bins=50, line1=-0.05, line2=0.05):
    """
    Generates and plots a histogram and KDE curve on a given matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to plot on.
        data (np.ndarray): The input numpy array.
        data_name (str): The name of the data for use in labels and titles.
        title_fontsize (int): The font size for the plot title.
        axislabel_fontsize (int): The font size for the axis labels.
        legend_fontsize (int): The font size for the legend.
        text_fontsize (int): The font size for the text annotations.
        bins (int): The number of bins for the histogram.
        line1 (float): The position of the first vertical line.
        line2 (float): The position of the second vertical line.
    """
    # Create the histogram plot on the provided axes
    ax.hist(data, bins=bins, density=True, color='#a9a9a9', alpha=0.7, label=f'Histogram')

    # Create a KDE (Kernel Density Estimate) for a smooth curve
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    ax.plot(x_range, kde(x_range), color='#2F4F4F', linewidth=2, label=f'KDE Curve')

    # Add vertical dashed lines
    ax.axvline(x=line1, color='firebrick', linestyle='--', linewidth=1.5, label=f'x = {line1}')
    ax.axvline(x=line2, color='darkgreen', linestyle='--', linewidth=1.5, label=f'x = {line2}')

    # Calculate the counts for each region
    count_less = np.sum(data < line1)
    count_greater = np.sum(data > line2)
    count_between = len(data) - count_less - count_greater

    # Add text annotations for the counts
    y_max = ax.get_ylim()[1]
    bbox_props = dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor='black', alpha=0.9)

    # Apply this style to your text calls
    ax.text(line1 * 2.5, y_max * 0.2, f'Count < {line1}: {count_less}',
            fontsize=text_fontsize, color='firebrick', ha='center', bbox=bbox_props)

    ax.text(line2 * 2.5, y_max * 0.2, f'Count > {line2}: {count_greater}',
            fontsize=text_fontsize, color='darkgreen', ha='center', bbox=bbox_props)

    ax.text((line1 + line2) / 2, y_max * 0.4, f'Count between: {count_between}',
            fontsize=text_fontsize, ha='center', bbox=bbox_props)
    # Add labels and title
    ax.set_xlabel(data_name, fontsize=axislabel_fontsize)
    ax.set_ylabel('Density', fontsize=axislabel_fontsize)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
