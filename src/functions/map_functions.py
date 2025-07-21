import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from geopy.distance import geodesic
from scipy.stats import gaussian_kde
from functions.math_functions import interpolate
import datetime
import re
import os




def get_projection(projection):
    """
    Returns a cartopy projection object based on the string name.
    Supported: 'platecarree', 'mercator', 'transversemercator', 'lambert', 'albers'.
    """
    if isinstance(projection, str):
        proj = projection.lower()
        if proj == 'platecarree':
            return ccrs.PlateCarree()
        elif proj == 'mercator':
            return ccrs.Mercator()
        elif proj == 'transversemercator':
            return ccrs.TransverseMercator()
        elif proj == 'lambert':
            return ccrs.LambertConformal()
        elif proj == 'albers':
            return ccrs.AlbersEqualArea()
        else:
            raise ValueError(f"Unknown projection: {projection}")
    return projection

def plot_heatmap(points, score, method='rbf', title="Heatmap", 
                alpha=0.6, osm_zoom=14, projection='platecarree', grid_resolution=200, padding=0.01, save_path=None):
    """
    Plots a heatmap of given points and scores using the specified interpolation method.

    Args:
        points (array-like): Array of (lon, lat) points, shape (N, 2).
        score (array-like): Values at each point, shape (N,).
        method (str): Interpolation method ('rbf', 'griddata', 'kde').
        title (str): Plot title.
        alpha (float): Transparency of the heatmap.
        osm_zoom (int): OSM basemap zoom level.
        projection (str or cartopy.crs.Projection): Map projection to use.
        grid_resolution (int): Grid resolution for interpolation.
        padding (float): Padding for plot extent.
        save_path(str): Save heatmap to path

    Returns:
        None
    """
    points = np.array(points)
    score = np.array(score)
    
    # Validation
    if points.shape[0] != len(score):
        raise ValueError(f"Points ({points.shape[0]}) and score ({len(score)}) must have same length")
    
    lat_bounds = (points[:, 1].min(), points[:, 1].max())
    lon_bounds = (points[:, 0].min(), points[:, 0].max())

    if method in ['rbf', 'griddata']:
        lon_i_m, lat_i_m, grid = interpolate(points, score, lat_bounds, lon_bounds, 
                                            method=method, grid_resolution=grid_resolution)
    elif method == 'kde':
        lat_i = np.linspace(lat_bounds[0], lat_bounds[1], grid_resolution)
        lon_i = np.linspace(lon_bounds[0], lon_bounds[1], grid_resolution)
        lon_i_m, lat_i_m = np.meshgrid(lon_i, lat_i)
        
        xy = np.vstack([points[:,0], points[:,1]])
        weights = np.array(score)
        kde = gaussian_kde(xy, weights=weights, bw_method=0.05)
        grid = kde(np.vstack([lon_i_m.ravel(), lat_i_m.ravel()])).reshape(lat_i_m.shape)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate extent with padding
    lon_range = lon_bounds[1] - lon_bounds[0]
    lat_range = lat_bounds[1] - lat_bounds[0]
    
    extent = [lon_bounds[0] - padding*lon_range, lon_bounds[1] + padding*lon_range,
              lat_bounds[0] - padding*lat_range, lat_bounds[1] + padding*lat_range]
    
    # Create figure with selected projection
    proj_obj = get_projection(projection)
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': proj_obj})
    
    # Add OSM basemap (always in PlateCarree)
    osm = cimgt.OSM()
    ax.add_image(osm, osm_zoom)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title(title)

    # Plot heatmap (data always in PlateCarree)
    pcm = ax.pcolormesh(lon_i_m, lat_i_m, grid, cmap='coolwarm_r', shading='auto',
                        transform=ccrs.PlateCarree(), alpha=alpha)
    
    # Add colorbar
    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05, aspect=20)
    cbar.set_label('lower = better')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")  
    sanitized_title = re.sub(r'[^\w\s]', '', title).replace(' ', '_')
    output_prefix = f"{sanitized_title}_{timestamp}"

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heatmap to {save_path}")
    else:
        # Fallback filename if save_path is not provided
        default_save_path = f"../output/{output_prefix}_heatmap.png"
        os.makedirs(os.path.dirname(default_save_path), exist_ok=True)
        plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved heatmap to {default_save_path}")

    plt.tight_layout()
    plt.show()


def plot_points(points, title="Points on OSM", osm_zoom=14, projection='platecarree'):
    """
    Plots points on an OSM basemap.

    Args:
        points (array-like): Array of (lon, lat) points, shape (N, 2).
        title (str): Plot title.
        osm_zoom (int): OSM basemap zoom level.
        projection (str or cartopy.crs.Projection): Map projection to use.

    Returns:
        None
    """
    points = np.array(points)
    
    lat_bounds = (points[:, 1].min(), points[:, 1].max())
    lon_bounds = (points[:, 0].min(), points[:, 0].max())
    
    # Add padding
    lat_range = lat_bounds[1] - lat_bounds[0]
    lon_range = lon_bounds[1] - lon_bounds[0]
    padding = 0.01
    
    extent = [lon_bounds[0] - padding*lon_range, lon_bounds[1] + padding*lon_range,
              lat_bounds[0] - padding*lat_range, lat_bounds[1] + padding*lat_range]

    proj_obj = get_projection(projection)
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': proj_obj})
    
    osm = cimgt.OSM()
    ax.add_image(osm, osm_zoom)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, alpha=0.7, 
               transform=ccrs.PlateCarree(), edgecolors='black', linewidths=0.5,
               label=f'{len(points)} points')
    
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def generate_points(lat0, lat1, lon0, lon1, distance_per_point):
    """
    Generates a grid of points within the given latitude and longitude bounds.

    Args:
        lat0 (float): Minimum latitude.
        lat1 (float): Maximum latitude.
        lon0 (float): Minimum longitude.
        lon1 (float): Maximum longitude.
        distance_per_point (float): Distance between points in meters.

    Returns:
        np.ndarray: Array of (lon, lat) points.
    """
    lat_meters = geodesic((lat0, lon0), (lat1, lon0)).meters
    avg_lat = (lat0 + lat1) / 2
    lon_meters = geodesic((avg_lat, lon0), (avg_lat, lon1)).meters
    
    lat_points = int(lat_meters / distance_per_point) + 1
    lon_points = int(lon_meters / distance_per_point) + 1
    
    lat_axis = np.linspace(lat0, lat1, lat_points)
    lon_axis = np.linspace(lon0, lon1, lon_points)
    
    # Use both axes correctly
    lon_mesh, lat_mesh = np.meshgrid(lon_axis, lat_axis)
    pts = np.round(np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()]), 4)
    
    print(f"Generated grid: {lon_points} x {lat_points} = {len(pts)} points")
    return pts  # Return only points in (lon, lat) format

def summary(lat0, lat1, lon0, lon1, points, score):
    """
    Prints a summary of the area and score statistics.

    Args:
        lat0 (float): Minimum latitude.
        lat1 (float): Maximum latitude.
        lon0 (float): Minimum longitude.
        lon1 (float): Maximum longitude.
        points (array-like): Array of points.
        score (array-like): Array of scores.

    Returns:
        None
    """
    print(20*"=")
    print(f"SW corner: {lat0}, {lon0}")
    print(f"NE corner: {lat1}, {lon1}")
    print(f"Area: {np.abs(lat0-lat1):.3f} lat x {np.abs(lon0-lon1):.3f} lon")
    print(20*"=")
    print(f"Number of points: {len(points)}")
    print(f"Number of score values: {len(score)}")
    print(f"Average score: {np.mean(score):.4f}")
    print(20*"=")



