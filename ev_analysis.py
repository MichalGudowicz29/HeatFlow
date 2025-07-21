from functions.map_functions import *
from functions.math_functions import *
from functions.mcda_functions import *
import numpy as np

# Configuration
# Coordinates
lat0, lat1 = 53.401, 53.445 
lon0, lon1 = 14.508, 14.587
# Distance between points
distance_per_point = 500

EV_CRITERIA = [
        {
            "name": "Proximity to parking areas",
            "id": "parking",
            "method": "distance", 
            "api_params": ("amenity", "parking"),
            "type": -1,  # mniejsza odległość = lepiej
        },
        {
            "name": "Proximity to main road",
            "id": "primary_roads", 
            "method": "distance",
            "api_params": ("highway", "primary"),
            "type": -1,
        },
        {
            "name": "Distance from existing stations",
            "id": "existing_stations",
            "method": "distance",
            "api_params": ("amenity", "charging_station"),
            "type": 1,  # większa odległość = lepiej (nie za blisko istniejących)
        },
        {
            "name": "Proximity to shopping",
            "id": "shopping",
            "method": "distance",
            "api_params": ("shop", "*"),
            "type": -1,  # mniejsza odległość = lepiej
        },
        {
            "name": "Restaurant",
            "id": "restaurant",
            "method": "distance", 
            "api_params": ("amenity", "restaurant"),
            "type": -1,  # mniejsza odległość = lepiej
        },
        {
            "name": "Density od residential",
            "id": "density_residential",
            "method": "count",
            "api_params": ("building", "residential"),
            "type": 1,  # więcej = lepiej
        }
    ]

CRITERIA_TYPES = np.array([-1, -1, 1, -1, -1, 1])  
POI_INDICES = None  # No merging



weights_file_path = "data/ev_charing.csv"
criteria_names = [
    "Odleglosc do parkingu",
    "Odleglosc do glownej drogi",
    "Odleglosc od istniejacych juz stacji",
    "Odleglosc od sklepow", 
    "Odleglosc od restauracji",
    "Gestosc zabudowy mieszkalnej"
]

print(f"Criterias: {len(criteria_names)}")
for i, name in enumerate(criteria_names):
    print(f"  {i}: {name}")

if not os.path.exists(weights_file_path):
    make_rancom_weights(criteria_names, weights_file_path)

def main():
    """
    Main function for EV station preference analysis.
    Generates grid points, analyzes locations, plots heatmap and points, prints summary.
    """
    points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    names = [f"Point_{i}" for i in range(len(points))]

    # IMPORTANT: Convert to (lat, lon) for analyze_locations
    points_lat_lon = [(p[1], p[0]) for p in points]  # Convert (lon,lat) -> (lat,lon)
    
    names = [f"Point_{i}" for i in range(len(points))]

    preferences, ranking = analyze_locations(
        points=points_lat_lon,
        points_names=names,
        criteria=EV_CRITERIA,
        criteria_types=CRITERIA_TYPES,
        weights_file=weights_file_path,
        poi_indices=POI_INDICES,  # Add this parameter!
        output_prefix="new_ev_analysis",
        export_results=False,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0
    )
    preferences = preferences.tolist()
    
    plot_heatmap(points, preferences, method="griddata", 
                title="EV Station Preference Heatmap", alpha=0.8, grid_resolution=200)
    
    # Plot points
    plot_points(points, "Grid Points")
    
    # Summary
    summary(lat0, lat1, lon0, lon1, points, preferences)

if __name__ == "__main__":
    main()
    
