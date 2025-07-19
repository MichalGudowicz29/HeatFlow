from functions.map_functions import plot_heatmap, plot_points, generate_points, summary
from functions.mcda_functions import analyze_locations, make_rancom_weights
import numpy as np
import os

# Configuration
# Coordinates
lat0, lat1 = 53.401, 53.445 
lon0, lon1 = 14.508, 14.587
# Distance between points
distance_per_point = 500

BIKE_CRITERIA = [
    {
        "name": "Number of public transport stops",
        "id": "count_pub_trans",
        "method": "count",
        "api_params": ("public_transport", "platform"),
        "type": 1,
    },
    {
        "name": "Number of university areas",
        "id": "count_uni_area",
        "method": "count",
        "api_params": ("amenity", "university"),
        "type": 1,
    },
    {
        "name": "Bike paths",
        "id": "bi_path",
        "method": "distance",
        "api_params": ("highway", "cycleway"),
        "type": -1,
    },
    {
        "name": "Number of supermarkets",
        "id": "supermarket",
        "method": "count",
        "api_params": ("shop", "supermarket"),
        "type": 1,
    },
    {
        "name": "Number of parks nearby",
        "id": "recreation_park_count",
        "method": "count",
        "api_params": ("leisure", "park"),
        "type": 1,
    },
    {
        "name": "Number of offices/workplaces",
        "id": "office_count",
        "method": "count",
        "api_params": ("office", "yes"),
        "type": 1,
    },
    {
        "name": "Number of restaurants",
        "id": "restaurant_count",
        "method": "count",
        "api_params": ("amenity", "restaurant"),
        "type": 1,
    },
    {
        "name": "Number of fast food places",
        "id": "fast_food_count",
        "method": "count",
        "api_params": ("amenity", "fast_food"),
        "type": 1,
    },
    {
        "name": "Number of tourist attractions",
        "id": "tourism_attraction_count",
        "method": "count",
        "api_params": ("tourism", "attraction"),
        "type": 1,
    },
    {
        "name": "Residential building density",
        "id": "residential_building_count",
        "method": "count",
        "api_params": ("building", "residential"),
        "type": 1,
    },
]

# FIXED: Points in Szczecin - changed to (lon, lat) format
BIKE_POINTS = [
    (14.547968868949667, 53.43296129639522),  # Plac Grunwaldzki
    (14.555278766434393, 53.43209966711944),  # Pazim/Galaxy
    (14.49176316404754, 53.44771696811074),   # WI
    (14.485328899575466, 53.42733614288465),  # Ster
    (14.53133731478859, 53.42777787163157),   # Turzyn
    (14.499642240087212, 53.40365605376617),  # Cukrowa Uniwerek
    (14.567364906386535, 53.44247491754028),  # Fabryka Wody + Mieszkania
    (14.537395453240668, 53.427400712136276), # Kościuszki
    (14.559356706858086, 53.422574981749726), # Wyszyńskiego
    (14.550517853490351, 53.42451975173007),  # Brama Portowa
]

BIKE_POINTS_NAMES = [
    "Plac Grunwaldzki", "Pazim/Galaxy", "WI", "Ster", "Turzyn",
    "Cukrowa Uniwerek", "Fabryka Wody + Mieszkania", "Kościuszki",
    "Wyszyńskiego", "Brama Portowa"
]

POI_INDICES = [3, 5, 6, 7, 8]  # supermarkets, offices, restaurants, fast food, attractions
CRITERIA_TYPES = np.array([1, 1, -1, 1, 1, 1])  # after merging POI

weights_file_path = "data/as_rancom.csv"
merged_criteria_names = [
    "Number of public transport stops",
    "Number of university areas",
    "Bike paths",
    "POI",
    "Number of parks nearby",
    "Residential building density"
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
        criteria=BIKE_CRITERIA,
        criteria_types=CRITERIA_TYPES,
        weights_file="data/as_rancom.csv",
        poi_indices=POI_INDICES,
        output_prefix="bike_analysis",
        export_results=False,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0
    )
    preferences = preferences.tolist()
    
    plot_heatmap(points, preferences, method="griddata", 
                title="Bike Station Preference Heatmap", alpha=0.8, grid_resolution=200)
    
    # Plot points
    plot_points(points, "Grid Points")
    
    # Summary
    summary(lat0, lat1, lon0, lon1, points, preferences)

if __name__ == "__main__":
    main()