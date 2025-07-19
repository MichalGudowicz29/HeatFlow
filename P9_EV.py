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
            "name": "Główne ciągi komunikacyjne - trunk",
            "id": "trunk_roads",
            "method": "distance", 
            "api_params": ("highway", "trunk"),
            "type": -1,  # mniejsza odległość = lepiej
        },
        {
            "name": "Główne ciągi komunikacyjne - primary",
            "id": "primary_roads", 
            "method": "distance",
            "api_params": ("highway", "primary"),
            "type": -1,
        },
        {
            "name": "Dostępność energii - stacje transformatorowe",
            "id": "power_stations",
            "method": "distance",
            "api_params": ("power", "substation"),
            "type": -1,
        },
        {
            "name": "Centra handlowe",
            "id": "shopping_centers",
            "method": "count",
            "api_params": ("shop", "mall"),
            "type": 1,  # więcej = lepiej
        },
        {
            "name": "Biura",
            "id": "offices",
            "method": "count", 
            "api_params": ("office", "yes"),
            "type": 1,
        },
        {
            "name": "Uczelnie",
            "id": "universities",
            "method": "count",
            "api_params": ("amenity", "university"),
            "type": 1,
        },
        {
            "name": "Hotele (miejsca publiczne)",
            "id": "hotels", 
            "method": "count",
            "api_params": ("tourism", "hotel"),
            "type": 1,
        },
        {
            "name": "Restauracje (miejsca publiczne)",
            "id": "restaurants",
            "method": "count", 
            "api_params": ("amenity", "restaurant"),
            "type": 1,
        },
        {
            "name": "Stacje benzynowe (infrastruktura)",
            "id": "fuel_stations",
            "method": "count",
            "api_params": ("amenity", "fuel"),
            "type": 1,
        },
        {
            "name": "Gęstość zabudowy mieszkaniowej",
            "id": "residential_density",
            "method": "count",
            "api_params": ("building", "residential"), 
            "type": 1,
        }
    ]

POI_INDICES = [3, 4, 5, 6, 7] # Centra handlowe, biura, uczelnie, hotele, restauracje
CRITERIA_TYPES = np.array([-1, -1, -1, 1, 1, 1])  # after merging POI
weights_file_path = "data/ev_rancom.csv"
merged_criteria_names = [
    "Główne ciągi komunikacyjne - trunk",
    "Główne ciągi komunikacyjne - primary",
    "Dostępność energii - stacje transformatorowe",
    "Miejsca publiczne (centra handlowe, biura, uczelnie, hotele, restauracje)", 
    "Stacje benzynowe (infrastruktura)",
    "Gęstość zabudowy mieszkaniowej"
]

if not os.path.exists(weights_file_path):
    make_rancom_weights(merged_criteria_names, weights_file_path)



def main():
    """
    Main function for bike station preference analysis.
    Generates grid points, analyzes locations, plots heatmap and points, prints summary.
    """
    points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    #points = BIKE_POINTS
    names = [f"Point_{i}" for i in range(len(points))]
    #names = BIKE_POINTS_NAMES

    # IMPORTANT: Convert to (lat, lon) for analyze_locations
    points_lat_lon = [(p[1], p[0]) for p in points]  # Convert (lon,lat) -> (lat,lon)
    
    names = [f"Point_{i}" for i in range(len(points))]

    preferences, ranking = analyze_locations(
        points=points_lat_lon,
        points_names=names,
        criteria=EV_CRITERIA,
        criteria_types=CRITERIA_TYPES,
        weights_file=weights_file_path,
        poi_indices=POI_INDICES,
        output_prefix="ev_analysis",
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