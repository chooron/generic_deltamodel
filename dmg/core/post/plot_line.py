import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.ticker import MaxNLocator  # <-- Import the new tool
from typing import Dict, Tuple, Optional


def plot_time_series(
        dates: pd.DatetimeIndex,
        data: Dict[str, np.ndarray],
        time_range: Optional[Tuple[str, str]] = None,
        target_name: str = 'target',
        colors: Optional[Dict[str, str]] = None,
        title: str = '',
        ylabel: str = 'Streamflow (mm/day)',
        title_fontsize: int = 16,
        ylabel_fontsize: int = 12,
        tick_fontsize: int = 11,
        legend_fontsize: int = 11,
        axis_linewidth: float = 1.2,
        ax: Optional[plt.Axes] = None
):
    """
    根据指定的时间范围，绘制高度可定制化的流量过程线图 (V6)。

    Parameters
    ----------
    dates : pd.DatetimeIndex
        x-axis datetime index.
    data : Dict[str, np.ndarray]
        Dictionary containing y-axis Numpy arrays.
    time_range : Optional[Tuple[str, str]], optional
        Tuple ('YYYY-MM-DD', 'YYYY-MM-DD') to select the time period for plotting.
        This also strictly sets the x-axis limits. If None, the full range is plotted.
    target_name : str, optional
        Key name for the target/observation data.
    colors : Optional[Dict[str, str]], optional
        Colors for non-target lines.
    title : str, optional
        Plot title.
    ylabel : str, optional
        Y-axis label.
    title_fontsize : int, optional
        Font size for the title.
    ylabel_fontsize : int, optional
        Font size for the y-axis label.
    tick_fontsize : int, optional
        Font size for axis tick labels.
    legend_fontsize : int, optional
        Font size for the legend.
    axis_linewidth : float, optional
        Linewidth for the plot's bounding box (spines).
    ax : Optional[plt.Axes], optional
        A pre-existing Axes object to plot on.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # --- 1. Validate Input Length ---
    y_len = len(next(iter(data.values())))
    if len(dates) != y_len:
        raise ValueError(f"Length mismatch: dates has {len(dates)} points, data has {y_len} points.")

    # --- 2. Select Data and Set Limits Based on time_range ---
    if time_range:
        try:
            start_date, end_date = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
            start_idx = dates.searchsorted(start_date, side='left')
            end_idx = dates.searchsorted(end_date, side='right')

            plot_dates = dates[start_idx:end_idx]
            plot_data = {name: arr[start_idx:end_idx] for name, arr in data.items()}
        except Exception as e:
            raise ValueError(f"Invalid `time_range`. Error: {e}")
    else:
        # If no range, use the full dataset
        start_date, end_date = dates[0], dates[-1]
        plot_dates = dates
        plot_data = data

    # --- 3. Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    else:
        fig = ax.figure

    if colors is None:
        colors = {}

    for name, y_values in plot_data.items():
        if name == target_name:
            ax.plot(plot_dates, y_values, color='0', linestyle='-', label=name, linewidth=1.0, zorder=0, alpha=1)
        else:
            ax.plot(plot_dates, y_values, color=colors.get(name), linestyle='dotted', label=name, linewidth=2.0, alpha=0.8)

    # --- 4. Formatting and Customization ---
    # Set strict plot limits
    ax.set_xlim(start_date, end_date)

    ax.set_title(title, fontsize=title_fontsize, weight='bold')
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_xlabel('')

    ax.legend(fontsize=legend_fontsize, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # --- MODIFIED X-AXIS HANDLING ---
    # 1. Set a locator to find at most 4 "nice" tick locations.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

    # 2. Set the format of the labels to 'YYYY-MM'. Labels will be horizontal by default.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # 3. The call to fig.autofmt_xdate() is REMOVED to prevent rotation.

    # Set axis spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(axis_linewidth)

    ax.grid(True, linestyle=':', alpha=0.7, color='gray')

    return fig, ax