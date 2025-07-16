import numpy as np
from scipy.interpolate import RBFInterpolator, griddata

def interpolate(pts, vals, lat_bounds, lon_bounds, method='rbf', grid_resolution=200):
    """
    Interpolates values at given points using the specified method.

    Args:
        pts (array-like): Array of (lon, lat) points.
        vals (array-like): Values at each point.
        lat_bounds (tuple): (min_lat, max_lat).
        lon_bounds (tuple): (min_lon, max_lon).
        method (str): Interpolation method ('rbf' or 'griddata').
        grid_resolution (int): Grid resolution.

    Returns:
        tuple: (lon_i_m, lat_i_m, grid) - meshgrid and interpolated grid.
    """
    lat_i = np.linspace(lat_bounds[0], lat_bounds[1], grid_resolution)
    lon_i = np.linspace(lon_bounds[0], lon_bounds[1], grid_resolution)
    
    # Use both axes correctly
    lon_i_m, lat_i_m = np.meshgrid(lon_i, lat_i)
    
    vals = np.array(vals).ravel()
    
    if method == 'griddata':
        print("Using griddata method")
        grid = griddata(pts, vals, (lon_i_m, lat_i_m), method='cubic')
    elif method == 'rbf':
        print("Using RBF method")
        query = np.column_stack([lon_i_m.ravel(), lat_i_m.ravel()])
        rbf = RBFInterpolator(pts, vals, kernel='thin_plate_spline', smoothing=0.01)
        grid = rbf(query).reshape(lat_i_m.shape)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return lon_i_m, lat_i_m, grid