from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import sys
import numpy as np
from datetime import datetime

# Add your functions directory to path
sys.path.append('functions')
from mcda_functions import *
from map_functions import *

app = Flask(__name__)
CORS(app)  # Enable CORS for web app

# Configuration
CRITERIA_PATH = "data/criteria.json"

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/api/criteria', methods=['GET'])
def get_criteria():
    """Get available criteria from JSON file"""
    try:
        with open(CRITERIA_PATH, 'r') as f:
            criteria_data = json.load(f)
        
        # Simplify criteria for frontend
        simplified_criteria = []
        for criterion in criteria_data['criteria']:
            simplified_criteria.append({
                'id': criterion['id'],
                'name': criterion['name'],
                'type': criterion.get('type', 1)
            })
        
        return jsonify({
            'success': True,
            'criteria': simplified_criteria
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate-heatmap', methods=['POST'])
def generate_heatmap():
    """Generate heatmap based on user parameters"""
    try:
        data = request.json
        
        # Extract parameters from frontend
        bounds = data.get('bounds', {})
        lat0 = float(bounds.get('lat0'))
        lat1 = float(bounds.get('lat1'))
        lon0 = float(bounds.get('lon0'))
        lon1 = float(bounds.get('lon1'))
        
        selected_criteria_ids = data.get('selectedCriteria', [])
        interpolation_method = data.get('method', 'griddata')
        distance_per_point = data.get('distance_per_point', 1000)
        
        print(f"Received request:")
        print(f"  Bounds: lat({lat0}, {lat1}), lon({lon0}, {lon1})")
        print(f"  Criteria: {selected_criteria_ids}")
        print(f"  Method: {interpolation_method}")
        
        # Ensure output and cache directories exist
        os.makedirs("output", exist_ok=True)
        os.makedirs("cache", exist_ok=True)
        
        # Load and filter criteria
        criteria, criteria_types, poi_indices = load_and_filter_criteria(
            criteria_path=CRITERIA_PATH,
            selected_ids=selected_criteria_ids,
            merge=False,
            merge_id=[]
        )
        
        # Create weights file
        criteria_names = [c["name"] for c in criteria]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_file_path = f"data/weights_{timestamp}.csv"
        
        if not os.path.exists(weights_file_path):
            make_rancom_weights(criteria_names, weights_file_path)
        
        # Generate points
        points = generate_points(lat0, lat1, lon0, lon1, distance_per_point)
        points_lat_lon = [(p[1], p[0]) for p in points]
        names = [f"Point_{i}" for i in range(len(points))]
        
        print(f"Generated {len(points)} points for analysis")
        
        # Analyze locations
        output_prefix = f"webapp_{timestamp}"
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
        
        # Generate heatmap data for frontend
        heatmap_data = []
        for i, point in enumerate(points):
            heatmap_data.append({
                'lat': point[1],  # latitude
                'lng': point[0],  # longitude
                'value': preferences[i]
            })
        
        # Calculate statistics
        stats = {
            'min_value': float(np.min(preferences)),
            'max_value': float(np.max(preferences)),
            'mean_value': float(np.mean(preferences)),
            'std_value': float(np.std(preferences))
        }
        
        # Save results for later reference
        results_file = f"output/webapp_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'bounds': bounds,
                'criteria': selected_criteria_ids,
                'method': interpolation_method,
                'heatmap_data': heatmap_data,
                'statistics': stats,
                'timestamp': timestamp
            }, f, indent=2)
        
        return jsonify({
            'success': True,
            'heatmap_data': heatmap_data,
            'statistics': stats,
            'points_analyzed': len(points),
            'criteria_used': len(selected_criteria_ids),
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get status of a processing job (for future async processing)"""
    # This would be used for async processing in production
    return jsonify({
        'success': True,
        'status': 'completed',
        'job_id': job_id
    })

if __name__ == '__main__':
    print("Starting EV Charging Station Analyzer API...")
    print("Available endpoints:")
    print("  GET  /api/criteria - Get available criteria")
    print("  POST /api/generate-heatmap - Generate heatmap")
    print("  GET  /api/status/<id> - Get job status")
    print("\nStarting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
