import datetime
import os
from functions.mcda_functions import *
from functions.map_functions import *

# =========================
# KONFIGURACJA
# =========================
CRITERIA_PATH = "../data/criteria.json"  # plik z kryteriami (pełny lub pod wodę)
SELECTED_IDS = [
    "residential_density",
    "public_transport_density",
    "park",
    "playground",
    "footway_density",
    "drinking_water"
]

# Obszar analizy – Stargard
lat0, lat1 = 53.295, 53.370
lon0, lon1 = 15.000, 15.120

# Odległość między punktami siatki (m)
distance_per_point = 500

# Plik wag
weights_file_path = "../data/water_weights.csv"

def main():
    """
    Analiza lokalizacji poidełek w Stargardzie z użyciem MCDA + GIS.
    """
    # Tworzenie folderów
    os.makedirs("output", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # Wczytanie kryteriów
    criteria, criteria_types, poi_indices = load_and_filter_criteria(
        criteria_path=CRITERIA_PATH,
        selected_ids=SELECTED_IDS,
        merge=False,
        merge_id=[]
    )

    # Tworzenie pliku wag, jeśli nie istnieje
    criteria_names = [c["name"] for c in criteria]
    print("Kryteria w analizie:", criteria_names)
    if not os.path.exists(weights_file_path):
        make_rancom_weights(criteria_names, weights_file_path)

    # Generowanie punktów siatki
    points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
    points_lat_lon = [(p[1], p[0]) for p in points]
    names = [f"Point_{i}" for i in range(len(points))]

    output_prefix = "water_analysis"

    # Analiza MCDA
    preferences, ranking = analyze_locations(
        points=points_lat_lon,
        points_names=names,
        criteria=criteria,
        criteria_types=criteria_types,
        weights_file=weights_file_path,
        poi_indices=poi_indices,
        output_prefix=output_prefix,
        export_results=False,
        chunk_size=10,
        radius=500,
        delay=1.0,
        chunk_delay=5.0,
    )
    preferences = preferences.tolist()

    # Heatmapa
    heatmap_title = "Optymalne lokalizacje poidełek – Stargard"
    plot_heatmap(
        points,
        preferences,
        method="griddata",
        title=heatmap_title,
        projection="mercator",
        alpha=0.45,
        grid_resolution=200,
    )

    # Punkty siatki
    points_title = "Punkty siatki analizy poidełek"
    plot_points(
        points,
        points_title,
    )

    # Podsumowanie
    summary(lat0, lat1, lon0, lon1, points, preferences)

if __name__ == "__main__":
    main()
