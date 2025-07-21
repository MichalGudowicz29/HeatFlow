import datetime
import os
import re
from functions.mcda_functions import load_and_filter_criteria, analyze_locations
from functions.map_functions import plot_heatmap, plot_points
from functions.generate_points import generate_points
from functions.rancom import make_rancom_weights
from functions.summary import summary

# Configuration
CRITERIA_PATH = "../data/criteria.json"
SELECTED_IDS = ["parking", "primary_roads", "existing_stations", "grocery", "restaurant"]
lat0, lat1 = 53.401, 53.445
lon0, lon1 = 14.490, 14.567
distance_per_point = 500  
weights_file_path = "../data/ev_rancom.csv"

def main():
    """
    Analyze EV charging station locations, generate plots, and save results with timestamps.
    """
    # Ensure output and cache directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("cache", exist_ok=True)


    # Load criteria from JSON
    criteria, criteria_types, poi_indices = load_and_filter_criteria(
        criteria_path=CRITERIA_PATH,
        selected_ids=SELECTED_IDS,
        merge=False,
        merge_id=[]
    )

    # Create weights file if it doesn't exist
    criteria_names = [c["name"] for c in criteria]
    if not os.path.exists(weights_file_path):
        make_rancom_weights(criteria_names, weights_file_path)

    # Generate points
    points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    points_lat_lon = [(p[1], p[0]) for p in points]
    names = [f"Point_{i}" for i in range(len(points))]

    output_prefix = "ev_analysis"
    # Analyze locations
    preferences, ranking = analyze_locations(
        points=points_lat_lon,
        points_names=names,
        criteria=criteria,
        criteria_types=criteria_types,
        weights_file=weights_file_path,
        poi_indices=poi_indices,
        output_prefix=output_prefix,
        export_results=True,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0,
    )
    preferences = preferences.tolist()

    # Plot heatmap with timestamped filename
    heatmap_title = "title"
    plot_heatmap(
        points,
        preferences,
        method="griddata",
        title=heatmap_title,
        alpha=0.8,
        grid_resolution=200,
    )

    # Plot points with timestamped filename
    points_title = "Grid Points"
    plot_points(
        points,
        points_title,
        save_path=points_filename
    )

    # Summary
    summary(lat0, lat1, lon0, lon1, points, preferences)

if __name__ == "__main__":
    main()
