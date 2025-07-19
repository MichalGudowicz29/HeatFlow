from functions.mcda_functions import *
from functions.math_functions import *
from functions.map_functions import *

criteria = [
    {}
]

criteria_types = []

poi_indices = []


lat0, lat1 = 53.401, 53.445
lon0, lon1 = 14.490, 14.567
distance_per_point = 500 # meters
weights_file_path = "data/ev_rancom.csv" # or other file path

def main():
    points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    names = [f"Point_{i}" for i in range(len(points))]
    points_lat_lon = [(p[1], p[0]) for p in points]
    names = [f"Point_{i}" for i in range(len(points))]

    preferences, ranking = analyze_locations(
        points=points_lat_lon,
        points_names=names,
        criteria=criteria,
        criteria_types=criteria_types,
        weights_file=weights_file_path,
        poi_indices=poi_indices,
        output_prefix="analysis",
        export_results=False,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0
    )

    plot_heatmap(points, preferences, method="griddata", 
                title="title", alpha=0.8, grid_resolution=200)
    
    # Plot points
    plot_points(points, "Grid Points")
    
    # Summary
    summary(lat0, lat1, lon0, lon1, points, preferences)

if __name__ == "__main__":
    main()