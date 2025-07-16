import numpy as np
from functions.map_functions import plot_points

# Example: generate random points in Warsaw area
# Latitude: 52.20 - 52.26, Longitude: 20.95 - 21.05
np.random.seed(42)
num_points = 30
lats = np.random.uniform(52.20, 52.26, num_points)
lons = np.random.uniform(20.95, 21.05, num_points)
points = np.column_stack([lons, lats])  # (lon, lat)

projections = [
    ('platecarree', 'PlateCarree (default)'),
    ('mercator', 'Mercator'),
    ('transversemercator', 'Transverse Mercator'),
    ('lambert', 'Lambert Conformal'),
    ('albers', 'Albers Equal Area')
]

for proj, proj_name in projections:
    print(f"\nGenerating points map with projection: {proj_name}")
    plot_points(points, f"Random Points in Warsaw - {proj_name}", projection=proj) 