import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def geoplot_single_metric(
        gdf: gpd.GeoDataFrame,
        metric_name: str,
        title: str = None,
        map_color: bool = False,
        draw_rivers: bool = False,
        dynamic_colorbar: bool = False,
        dpi: int = 100,
        marker='o',
        marker_size: int = 50,
        ax=None,
        cax_pos=[0.83, 0.07, 0.16, 0.03],
        cmap="RdBu",
        vmin=0,
        vmax=1,
):
    """Geographically map a single model performance metric using Cartopy.

    Parameters
    ----------
    gdf
        GeoDataFrame with 'lat', 'lon', and metric columns.
    metric_name
        The name of the metric column to plot.
    title
        The title of the plot.
    map_color
        Whether to use color for the map background.
    draw_rivers
        Whether to draw rivers on the map.
    dynamic_colorbar
        If True, colorbar limits are based on data. If False, vmin/vmax are used.
    dpi
        The resolution of the plot.
    marker_size
        The size of the markers.
    ax
        A pre-existing Matplotlib Axes with a Cartopy projection.
    cmap
        The colormap to use for the scatter points.
    vmin
        Minimum value for the colorbar.
    vmax
        Maximum value for the colorbar.

    Returns
    -------
    fig
        The Matplotlib Figure object.
    ax
        The Matplotlib Axes object.
    """
    # --- Input Validation ---
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        raise ValueError("GeoDataFrame must include 'lat' and 'lon' columns.")
    if metric_name not in gdf.columns:
        raise ValueError(f"GeoDataFrame does not contain the column '{metric_name}'.")

    # --- Setup Figure and Axes ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi, subplot_kw={'projection': ccrs.Mercator()})
    else:
        fig = ax.figure

    # --- Setup Map Extent ---
    min_lat, max_lat = gdf['lat'].min() - 2.5, gdf['lat'].max() + 2.5
    min_lon, max_lon = gdf['lon'].min() - 2.5, gdf['lon'].max() + 2.5
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # --- Add Map Features ---
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=1, linestyle=':', zorder=2)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, zorder=2)
    if map_color:
        ax.add_feature(cfeature.LAND, zorder=0)
        ax.add_feature(cfeature.OCEAN, zorder=0)
    else:
        ax.add_feature(cfeature.LAND, facecolor='white', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    if draw_rivers:
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.3, edgecolor='blue', zorder=1)

    # --- Plot the Metric Data ---
    scatter = ax.scatter(
        gdf['lon'], gdf['lat'],
        c=gdf[metric_name],
        s=marker_size,
        marker=marker,
        cmap=cmap,
        alpha=0.8,
        edgecolor='none',
        transform=ccrs.PlateCarree(),
        vmin=(None if dynamic_colorbar else vmin),
        vmax=(None if dynamic_colorbar else vmax),
        zorder=3
    )

    # --- MODIFIED: Add Inset Colorbar ---
    # Create an inset axes for the colorbar inside the main plot.
    # Position: [left, bottom, width, height] in axes coordinates (from 0 to 1).
    # To center a 0.3 width bar: left = (1.0 - 0.3) / 2 = 0.35
    cax = ax.inset_axes(cax_pos)

    # Create the colorbar using the new inset axes (cax).
    cbar = fig.colorbar(scatter, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(title, fontsize=16, labelpad=-45)  # 使用 labelpad 调整标题位置
    # --- End of Modification ---
    return fig, ax