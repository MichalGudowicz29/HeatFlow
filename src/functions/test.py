import requests

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
radius = 200
lat = 53.431525
lon = 14.555530

# Pobierz przystanki w promieniu
stops_query = f"""
[out:json][timeout:25];
(
 node["highway"="bus_stop"](around:{radius},{lat},{lon});
 node["public_transport"="platform"](around:{radius},{lat},{lon});
);
out ids;
"""

stops_response = requests.post(OVERPASS_URL, data={'data': stops_query})
stops_response.encoding = 'utf-8'
stops_data = stops_response.json()

stop_ids = [el['id'] for el in stops_data.get('elements', []) if el['type'] == 'node']
stops = len(stop_ids) * 5
# Mnożymy przez 5 i zwracamy

print(stops)
