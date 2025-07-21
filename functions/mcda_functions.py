import numpy as np
import json
import os
import time
from typing import List, Tuple, Dict, Any

from geopy.distance import geodesic
import requests
from requests.exceptions import RequestException
from pymcdm.methods import SPOTIS
from pymcdm.weights.subjective.rancom import RANCOM

# Numpy settings
np.set_printoptions(precision=4, suppress=True, linewidth=1000)


def make_rancom_weights(criteria_names: List[str], weights_file_path: str) -> None:
    """
    Makes weights for the RANCOM method.
    """
    rancom_object_names = [crit_def for crit_def in criteria_names]
    print(rancom_object_names)
    rancom = RANCOM(object_names=rancom_object_names)
    weights = rancom()
    rancom.to_csv(weights_file_path)
    

def count(lat: float, lon: float, tag_key: str, tag_value: str, 
          radius: int = 500, max_retries: int = 5, delay: float = 1.0) -> int:
    """
    Counts points of interest (POI) of a given type around a location using Overpass API.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        tag_key (str): OSM tag key (e.g., "amenity").
        tag_value (str): OSM tag value (e.g., "restaurant").
        radius (int, optional): Search radius in meters. Default is 500.
        max_retries (int, optional): Maximum number of retries. Default is 5.
        delay (float, optional): Delay between requests in seconds. Default is 1.0.

    Returns:
        int: Number of found POIs within the radius, or 0 on error.
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            
            overpass_url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            (
              node["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              way["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              relation["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
            );
            out count;
            """

            response = requests.get(overpass_url, params={'data': query}, timeout=30)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt * 5
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()

            if (data.get('elements') and 
                data['elements'][0].get('tags', {}).get('total')):
                return int(data['elements'][0]['tags']['total'])
            else:
                return 0
                
        except RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return 0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return 0

    return 0


def find_nearest_poi(lat: float, lon: float, tag_key: str, tag_value: str,
                               radius: int = 500, max_retries: int = 5, delay: float = 1.0) -> float:
    """
    Finds the distance to the nearest point of interest (POI) of a given type using Overpass API.

    Args:
        lat (float): Latitude of the starting point.
        lon (float): Longitude of the starting point.
        tag_key (str): OSM tag key (e.g., "highway", "amenity").
        tag_value (str): OSM tag value (e.g., "cycleway", "hospital").
        radius (int, optional): Maximum search radius in meters. Default is 500.
        max_retries (int, optional): Maximum number of retries. Default is 5.
        delay (float, optional): Delay between requests in seconds. Default is 1.0.

    Returns:
        float: Distance in meters to the nearest POI, or the radius if not found or on error.
    """
    for attempt in range(max_retries):
        try:
            time.sleep(delay)
            
            url = "http://overpass-api.de/api/interpreter"
            query = f"""
            [out:json][timeout:25];
            (
              node["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              way["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
              relation["{tag_key}"="{tag_value}"](around:{radius},{lat},{lon});
            );
            out center;
            """
            
            response = requests.get(url, params={'data': query}, timeout=30)
            
            if response.status_code == 429:
                wait_time = 2 ** attempt * 5
                print(f"Rate limited. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            min_dist = float('inf')
            for el in data.get('elements', []):
                poi_coord = None
                if el['type'] == 'node':
                    poi_coord = (el['lat'], el['lon'])
                elif 'center' in el:
                    poi_coord = (el['center']['lat'], el['center']['lon'])
                else:
                    continue
                    
                dist = geodesic((lat, lon), poi_coord).meters
                if dist < min_dist:
                    min_dist = dist
            
            return min_dist if min_dist < float('inf') else radius
            
        except RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return radius
        except Exception as e:
            print(f"Unexpected error: {e}")
            return radius
    
    return radius


def save_cache(data: Dict, filename: str = 'cache.json') -> None:
    """
    Saves data to a cache file in JSON format.

    Args:
        data (Dict): Dictionary with data to save.
        filename (str, optional): Name of the cache file. Defaults to 'cache.json'.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_cache(filename: str = 'cache.json') -> Dict:
    """
    Loads data from a cache file in JSON format.

    Args:
        filename (str, optional): Name of the cache file to load. Defaults to 'cache.json'.

    Returns:
        Dict: Dictionary with loaded data or empty dict if file does not exist.
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}



def get_criteria_vector(point: Tuple[float, float], criteria_list: List[Dict],
                       radius: int = 500, delay: float = 1.0) -> List[float]:
    """
    Calculates the criteria vector for a single geographical point.
    
    For each criterion in the list, it calls the appropriate function (count or find_nearest_poi)
    and returns a vector of numerical values representing all criteria.
    
    Args:
        point (Tuple[float, float]): Coordinates of the point (latitude, longitude)
        criteria_list (List[Dict]): List of dictionaries defining criteria with keys:
                                   - "method": "count" or "distance"
                                   - "api_params": tuple (tag_key, tag_value)
                                   - "name": criterion name
        radius (int, optional): Search radius in meters. Defaults to 500.
        delay (float, optional): Delay between API requests. Defaults to 1.0.
    
    Returns:
        List[float]: Vector of criterion values for the given point
    
    Example:
        >>> criteria = [{"method": "count", "api_params": ("amenity", "restaurant"), "name": "Restauracje"}]
        >>> get_criteria_vector((53.4329, 14.5479), criteria)
        [12.0]
    """
    lat, lon = point
    values = []
    
    for crit_def in criteria_list:
        tag_key, tag_val = crit_def["api_params"]
        method = crit_def["method"]
        
        try:
            if method == "distance":
                value = find_nearest_poi(lat, lon, tag_key, tag_val, radius, delay=delay)
            elif method == "count":
                value = float(count(lat, lon, tag_key, tag_val, radius, delay=delay))
            else:
                value = 0.0
            values.append(value)
        except Exception as e:
            print(f"Error processing {crit_def['name']} for point ({lat}, {lon}): {e}")
            values.append(0.0)
    
    return values


def point_to_key(point: Tuple[float, float]) -> str:
    """
    Converts a geographical point to a string key for the cache mechanism.
    
    Formats coordinates to 6 decimal places, which provides accuracy of about 11 cm at sea level.
    
    Args:
        point (Tuple[float, float]): Coordinates of the point (latitude, longitude)
    
    Returns:
        str: Key in the format "lat,lon" with 6 decimal places
        
    Example:
        >>> point_to_key((53.432961, 14.547969))
        "53.432961,14.547969"
    """
    return f"{point[0]:.6f},{point[1]:.6f}"


def process_points_in_chunks(points_list: List[Tuple[float, float]], 
                           criteria_list: List[Dict],
                           chunk_size: int = 10,
                           radius: int = 500,
                           delay: float = 1.0,
                           chunk_delay: float = 5.0,
                           cache_file: str = 'cache.json') -> np.ndarray:
    """
    Processes a list of geographical points in small chunks with caching.
    
    The function divides a large list of points into smaller chunks to avoid API overload
    and allows for regular progress saving. It uses caching to avoid repeating
    already calculated values.
    
    Args:
        points_list (List[Tuple[float, float]]): List of points to process
        criteria_list (List[Dict]): List of criterion definitions to calculate
        chunk_size (int, optional): Size of the point chunk. Defaults to 10.
        radius (int, optional): Search radius in meters. Defaults to 500.
        delay (float, optional): Delay between API requests. Defaults to 1.0.
        chunk_delay (float, optional): Delay between chunks in seconds. Defaults to 5.0.
        cache_file (str, optional): Name of the cache file. Defaults to 'cache.json'.
    
    Returns:
        np.ndarray: 2D matrix where each row represents a point, and columns are criterion values
        
    Example:
        >>> points = [(53.4329, 14.5479), (53.4350, 14.5500)]
        >>> criteria = [{"method": "count", "api_params": ("amenity", "restaurant"), "name": "Restauracje"}]
        >>> matrix = process_points_in_chunks(points, criteria, chunk_size=2)
        >>> matrix.shape
        (2, 1)
    """
    cache = load_cache(cache_file)
    all_results = []
    
    total_chunks = (len(points_list) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(0, len(points_list), chunk_size):
        chunk = points_list[chunk_idx:chunk_idx + chunk_size]
        chunk_num = chunk_idx // chunk_size + 1
        
        print(f"Processing chunk {chunk_num}/{total_chunks}")
        
        chunk_results = []
        for i, point in enumerate(chunk):
            key = point_to_key(point)
            
            if key in cache:
                chunk_results.append(cache[key])
            else:
                print(f"Processing point {chunk_idx + i + 1}: {point}")
                try:
                    result = get_criteria_vector(point, criteria_list, radius, delay)
                    chunk_results.append(result)
                    cache[key] = result
                    save_cache(cache, cache_file)
                except Exception as e:
                    print(f"Error processing point {point}: {e}")
                    default_result = [0.0] * len(criteria_list)
                    chunk_results.append(default_result)
                    cache[key] = default_result
        
        all_results.extend(chunk_results)
        
        if chunk_num < total_chunks:
            time.sleep(chunk_delay)
    
    return np.array(all_results)


# def generate_grid_points(lat_range: Tuple[float, float], 
#                         lon_range: Tuple[float, float], 
#                         grid_size: int = 10) -> List[Tuple[float, float]]:
#     """
#     Generates a regular grid of geographical points for analysis.
    
#     Creates evenly distributed points within a rectangular geographical area.
#     Useful for systematic analysis of the entire region instead of specific locations.
    
#     Args:
#         lat_range (Tuple[float, float]): Latitude range (min, max)
#         lon_range (Tuple[float, float]): Longitude range (min, max)
#         grid_size (int, optional): Number of points on each axis. Defaults to 10.
#                                   Total number of points = grid_size²
    
#     Returns:
#         List[Tuple[float, float]]: List of coordinate points (lat, lon)
        
#     Example:
#         >>> points = generate_grid_points((53.40, 53.45), (14.50, 14.55), 3)
#         >>> len(points)
#         9
#         >>> points[0]
#         (53.40, 14.50)
#     """
#     lat_grid = np.linspace(lat_range[0], lat_range[1], grid_size)
#     lon_grid = np.linspace(lon_range[0], lon_range[1], grid_size)
    
#     lon0, lat0 = np.meshgrid(lon_grid, lat_grid)
#     pts = np.column_stack([lat0.ravel(), lon0.ravel()])
    
#     return [(float(point[0]), float(point[1])) for point in pts]

def merge_poi_criteria(alts: np.ndarray, poi_indices: list, criteria_names: list = None) -> np.ndarray:
    poi_indices = sorted(poi_indices)
    if criteria_names is not None:
        print("Kryteria przed mergowaniem:")
        for i, name in enumerate(criteria_names):
            print(f"  {i}: {name}")
        print("Mergowane kryteria:")
        for i in poi_indices:
            print(f"  {i}: {criteria_names[i]}")
    merged_col = alts[:, poi_indices].sum(axis=1, keepdims=True)
    non_poi_indices = [i for i in range(alts.shape[1]) if i not in poi_indices]
    left = [i for i in non_poi_indices if i < poi_indices[0]]
    right = [i for i in non_poi_indices if i >= poi_indices[0]]
    left_cols = alts[:, left]
    right_cols = alts[:, right]
    result = np.hstack([left_cols, merged_col, right_cols])
    if criteria_names is not None:
        merged_name = " + ".join([criteria_names[i] for i in poi_indices])
        new_names = [criteria_names[i] for i in left] + [merged_name] + [criteria_names[i] for i in right]
        print("Kryteria po mergowaniu:")
        for i, name in enumerate(new_names):
            print(f"  {i}: {name}")
    return result


def calculate_bounds(data: np.ndarray, tolerance: float = 0.1) -> np.ndarray:
    """
    Calculates bounds (limits) for the SPOTIS method with additional tolerance.
    
    The SPOTIS method requires defining an ideal and worst point for each criterion.
    This function calculates the minimum and maximum values in the data and adds a tolerance margin.
    
    Args:
        data (np.ndarray): Matrix of data with criterion values
        tolerance (float, optional): Value added to the limits. Defaults to 0.1.
    
    Returns:
        np.ndarray: 2D matrix with columns [min_bound, max_bound] for each criterion
        
    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> calculate_bounds(data, tolerance=0.5)
        array([[0.5, 5.5],
               [1.5, 6.5]])
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    return np.column_stack([
        min_vals - tolerance,
        max_vals + tolerance
    ])


def export_results(preferences: np.ndarray, ranking: np.ndarray, 
                  points_names: List[str],
                  output_prefix: str = 'results') -> None:
    """
    Exports analysis results to files in CSV and JSON formats.
    
    Creates files with preferences (numerical values) and ranking with point names.
    Allows for further analysis and visualization of results.
    
    Args:
        preferences (np.ndarray): Array of preference values for each point
        ranking (np.ndarray): Array of point rankings (1-indexed)
        points_names (List[str]): List of point names corresponding to the ranking
        output_prefix (str, optional): Prefix for output file names. Defaults to 'results'.
    
    Returns:
        None
        
    Creates files:
        - {output_prefix}_preferences.csv: Preference values in CSV format
        - {output_prefix}_preferences.json: Preference values in JSON format
        - {output_prefix}_ranking.json: Complete ranking with names and preferences
    """
    np.savetxt(f'{output_prefix}_preferences.csv', preferences, delimiter=',', 
               header='preference', comments='')
    
    with open(f'{output_prefix}_preferences.json', 'w') as f:
        json.dump(preferences.tolist(), f, indent=2)
    
    ranking_data = {
        'ranking': ranking.tolist(),
        'points': points_names,
        'preferences': preferences.tolist()
    }
    
    with open(f'{output_prefix}_ranking.json', 'w') as f:
        json.dump(ranking_data, f, indent=2)


def analyze_locations(points: List[Tuple[float, float]], 
                     points_names: List[str],
                     criteria: List[Dict], 
                     criteria_types: np.ndarray,
                     weights_file: str,
                     poi_indices: List[int] = None,
                     output_prefix: str = 'results',
                     export_results: bool = True,
                     **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function for location analysis.
    
    Args:
        points: List of points (lat, lon) to analyze
        points_names: Point names
        criteria: List of criterion definitions
        criteria_types: Array of criterion types (1 = more is better, -1 = less is better)
        weights_file: Path to the weights file
        poi_indices: Indices of criteria to merge (optional)
        output_prefix: Prefix for output files
        **kwargs: Additional parameters (chunk_size, radius, delay, etc.)
    
    Returns:
        (preferences, ranking)
    """
    
    # Parameters
    chunk_size = kwargs.get('chunk_size', 10)
    radius = kwargs.get('radius', 500)
    delay = kwargs.get('delay', 1.0)
    chunk_delay = kwargs.get('chunk_delay', 5.0)
    cache_file = kwargs.get('cache_file', f'{output_prefix}_cache.json')

    # 1. Data processing
    alts = process_points_in_chunks(points, criteria, chunk_size, radius, delay, chunk_delay, cache_file)
    print(f"[DEBUG] alts shape: {alts.shape}, sample: {alts[:2]}")  # Debug: wymiary i próbka alts

    # 2. Merging POI criteria (if provided)
    if poi_indices:
        print(f"[DEBUG] Merging POI criteria with indices: {poi_indices}")
        alts_final = merge_poi_criteria(alts, poi_indices, [c['name'] for c in criteria])
        print(f"[DEBUG] After merging, alts_final shape: {alts_final.shape}, sample: {alts_final[:2]}")
    else:
        print(f"[DEBUG] No POI merging (poi_indices is {poi_indices})")
        alts_final = alts
        print(f"[DEBUG] alts_final shape (no merging): {alts_final.shape}, sample: {alts_final[:2]}")
    # 3. Loading weights
    rancom = RANCOM(filename=weights_file)
    weights = rancom()
    print(f"[DEBUG] Weights loaded from {weights_file}: {weights}, shape: {weights.shape}")

    # 4. Validate dimensions
    expected_num_criteria = alts_final.shape[1]
    print(f"[DEBUG] Expected number of criteria: {expected_num_criteria}")
    print(f"[DEBUG] criteria_types: {criteria_types}, shape: {criteria_types.shape}")
    if len(criteria_types) != expected_num_criteria:
        raise ValueError(f"Dimension mismatch: criteria_types has {len(criteria_types)} elements, "
                         f"but alts_final has {expected_num_criteria} columns")
    if len(weights) != expected_num_criteria:
        raise ValueError(f"Dimension mismatch: weights has {len(weights)} elements, "
                         f"but alts_final has {expected_num_criteria} columns")

    # 5. SPOTIS analysis
    bounds = calculate_bounds(alts_final)
    print(f"[DEBUG] Bounds shape: {bounds.shape}, sample: {bounds}")
    spotis = SPOTIS(bounds)
    preferences = spotis(alts_final, weights, criteria_types)
    ranking = spotis.rank(preferences)
    print(f"[DEBUG] Preferences shape: {preferences.shape}, sample: {preferences[:2]}")
    print(f"[DEBUG] Ranking shape: {ranking.shape}, sample: {ranking[:2]}")

    # 6. Export results
    if export_results:
        export_results(preferences, ranking, points_names, output_prefix)
    
    return preferences, ranking


# if __name__ == "__main__":
#     """
#     Main script for analyzing the best locations for bicycle infrastructure in Szczecin.
    
#     The script defines evaluation criteria, points to analyze, and conducts multi-criteria
#     decision analysis using the SPOTIS method to find the best locations for bicycle
#     infrastructure. Results are exported to JSON and CSV files.
    
#     Analysis criteria include:
#     - Public transport accessibility
#     - Proximity to universities
#     - Existing bicycle paths
#     - Access to services (supermarkets, restaurants, offices)
#     - Tourist attractions and recreational areas
#     - Residential building density
    
#     You can analyze specific points (use_grid=False) or a regular grid of points (use_grid=True).
#     """
    
#     BIKE_CRITERIA = [
#         {
#             "name": "Number of public transport platforms",
#             "id": "count_pub_trans",
#             "method": "count",
#             "api_params": ("public_transport", "platform"),
#             "type": 1,
#         },
#         {
#             "name": "Number of university areas",
#             "id": "count_uni_area",
#             "method": "count",
#             "api_params": ("amenity", "university"),
#             "type": 1,
#         },
#         {
#             "name": "Bicycle paths",
#             "id": "bi_path",
#             "method": "distance",
#             "api_params": ("highway", "cycleway"),
#             "type": -1,
#         },
#         {
#             "name": "Number of supermarkets",
#             "id": "supermarket",
#             "method": "count",
#             "api_params": ("shop", "supermarket"),
#             "type": 1,
#         },
#         {
#             "name": "Number of parks in the vicinity",
#             "id": "recreation_park_count",
#             "method": "count",
#             "api_params": ("leisure", "park"),
#             "type": 1,
#         },
#         {
#             "name": "Number of offices/workplaces",
#             "id": "office_count",
#             "method": "count",
#             "api_params": ("office", "yes"),
#             "type": 1,
#         },
#         {
#             "name": "Number of restaurants",
#             "id": "restaurant_count",
#             "method": "count",
#             "api_params": ("amenity", "restaurant"),
#             "type": 1,
#         },
#         {
#             "name": "Number of fast food",
#             "id": "fast_food_count",
#             "method": "count",
#             "api_params": ("amenity", "fast_food"),
#             "type": 1,
#         },
#         {
#             "name": "Number of tourist attractions",
#             "id": "tourism_attraction_count",
#             "method": "count",
#             "api_params": ("tourism", "attraction"),
#             "type": 1,
#         },
#         {
#             "name": "Residential building density",
#             "id": "residential_building_count",
#             "method": "count",
#             "api_params": ("building", "residential"),
#             "type": 1,
#         },
#     ]

#     # Points in Szczecin
#     BIKE_POINTS = [
#         (53.43296129639522, 14.547968868949667),  # Plac Grunwaldzki
#         (53.43209966711944, 14.555278766434393),  # Pazim/Galaxy
#         (53.44771696811074, 14.49176316404754),   # WI
#         (53.42733614288465, 14.485328899575466),  # Ster
#         (53.42777787163157, 14.53133731478859),   # Turzyn
#         (53.40365605376617, 14.499642240087212),  # Cukrowa Uniwerek
#         (53.44247491754028, 14.567364906386535),  # Fabryka Wody + Mieszkania
#         (53.427400712136276, 14.537395453240668), # Kościuszki
#         (53.422574981749726, 14.559356706858086), # Wyszyńskiego
#         (53.42451975173007, 14.550517853490351),  # Brama Portowa
#     ]

#     BIKE_POINTS_NAMES = [
#         "Plac Grunwaldzki", "Pazim/Galaxy", "WI", "Ster", "Turzyn",
#         "Cukrowa Uniwerek", "Fabryka Wody + Mieszkania", "Kościuszki",
#         "Wyszyńskiego", "Brama Portowa"
#     ]

#     # Configuration for bicycles
#     POI_INDICES = [3, 5, 6, 7, 8]  # supermarkety, biura, restauracje, fast food, atrakcje
#     CRITERIA_TYPES = np.array([1, 1, -1, 1, 1])  # after POI merging
    
#     # You can also analyze a grid of points instead of specific locations
#     use_grid = True
    
#     if use_grid:
#         analysis_points = generate_grid_points((53.40, 53.45), (14.50, 14.55), 2)
#         analysis_names = [f"Point_{i}" for i in range(len(analysis_points))]
#     else:
#         analysis_points = BIKE_POINTS
#         analysis_names = BIKE_POINTS_NAMES
    
#     # Run analysis
#     preferences, ranking = analyze_locations(
#         points=analysis_points,
#         points_names=analysis_names,
#         criteria=BIKE_CRITERIA,
#         criteria_types=CRITERIA_TYPES,
#         weights_file="as_rancom.csv",
#         poi_indices=POI_INDICES,
#         output_prefix="bike_analysis",
#         chunk_size=10,
#         radius=500,
#         delay=1.0,
#         chunk_delay=5.0
#     )
    
