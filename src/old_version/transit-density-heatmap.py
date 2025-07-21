import numpy as np 
import json
from functions.map_functions import plot_heatmap, plot_points

def extract_stops_from_json(json_path):
    """
    Loads stops from a JSON file and returns a list of stop dictionaries.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list[dict]: List of stops with stop_id, coords, dep_count, and stop_name.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stops = []
    for stop_id, info in data.items():
        stops.append({
            'stop_id': stop_id,
            'coords': [float(info['coords'][0]), float(info['coords'][1])],  # [lat, lon]
            'dep_count': len(info['departures']),
            'stop_name': info.get('stop_name', 'brak nazwy')
        })
    return stops

def main():
    """
    Main function. Loads stops, prints top 10, plots heatmaps and statistics.
    """
    stops = extract_stops_from_json('../data/departures_all_stops.json')
    filtered = [s for s in stops if s['dep_count'] > 0]

    # Convert to (lon, lat) format for plotting
    points = np.array([[s['coords'][1], s['coords'][0]] for s in filtered])
    score = [s['dep_count'] for s in filtered]

    # Top 10 stops
    top10 = sorted(filtered, key=lambda s: s['dep_count'], reverse=True)[:10]
    print("Top 10 przystanków:")
    for stop in top10:
        print(f"{stop['stop_id']}: {stop.get('stop_name', 'brak nazwy')} - {stop['dep_count']} odjazdów")

    # Plot heatmap using KDE - shows density/intensity distribution
    plot_heatmap(points, score, method='kde', 
                title="Transit Density Heatmap - KDE", 
                alpha=0.7, grid_resolution=150)

    # For comparison - also try RBF method
    plot_heatmap(points, score, method='rbf', 
                title="Transit Heatmap - RBF Interpolation", 
                alpha=0.7, grid_resolution=150)

    # Plot points to see actual stop locations
    plot_points(points, "All Transit Stops")

    # Statistics
    print(f"\nStatistics for file: {data_file}")
    print(f"Total stops with departures: {len(filtered)}")
    print(f"Total departures: {sum(score)}")
    print(f"Average departures per stop: {np.mean(score):.2f}")
    print(f"Max departures at single stop: {max(score)}")
    print(f"Min departures at single stop: {min(score)}")

if __name__ == "__main__":
    main()
