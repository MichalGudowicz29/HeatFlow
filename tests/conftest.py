import pytest

from heatflow import geo

# Mały obszar testowy w Szczecinie (~1,3 × 1,1 km)
BBOX = (14.540, 53.425, 14.560, 53.435)  # w, s, e, n
AREA = {"type": "Polygon", "coordinates": [[
    [BBOX[0], BBOX[1]], [BBOX[2], BBOX[1]], [BBOX[2], BBOX[3]], [BBOX[0], BBOX[3]], [BBOX[0], BBOX[1]],
]]}


def node(i, lon, lat, **tags):
    return {"type": "node", "id": i, "lon": lon, "lat": lat, "tags": tags}


def way_geom(i, coords, **tags):
    return {"type": "way", "id": i, "tags": tags,
            "geometry": [{"lon": lo, "lat": la} for lo, la in coords]}


class FakeOverpass:
    """Zwraca syntetyczne dane zależnie od treści zapytania."""

    def __init__(self):
        self.queries = []

    def query(self, q, notify=None):
        self.queries.append(q)
        if 'way["highway"~"^(footway' in q:
            return walking_grid()
        if '"amenity"="charging_station"' in q:
            return {"elements": [node(1, 14.545, 53.428), node(2, 14.556, 53.433)]}
        if '"amenity"="parking"' in q:
            return {"elements": [{"type": "node", "id": 10 + k, "lon": 14.541 + 0.002 * k, "lat": 53.426 + 0.001 * k}
                                 for k in range(8)]}
        if '"leisure"="park"' in q:
            return {"elements": [way_geom(100, [(14.548, 53.429), (14.552, 53.429), (14.552, 53.431),
                                               (14.548, 53.431), (14.548, 53.429)], leisure="park")]}
        if any(t in q for t in ('"natural"="water"', '"waterway"="riverbank"', '"landuse"~"^(reservoir|basin)$"',
                                '"railway"="rail"', '"highway"~"^(motorway|motorway_link)$"', '"barrier"~')):
            return {"elements": []}  # bariery dla dojść pieszych – brak w danych testowych
        if '"amenity"="nothing"' in q:
            return {"elements": []}
        if '["shop"]' in q:
            return {"elements": [{"type": "way", "id": 200 + k, "center": {"lon": 14.55, "lat": 53.43 + 0.0005 * k}}
                                 for k in range(5)]}
        raise AssertionError(f"Nieoczekiwane zapytanie: {q}")


def walking_grid(step=0.001):
    """Regularna sieć ulic co ~66 m (lon) / ~111 m (lat) wokół obszaru testowego."""
    els, nid = [], 1
    ids = {}
    lons = [round(14.530 + step * i, 6) for i in range(41)]
    lats = [round(53.418 + step * j, 6) for j in range(25)]
    for la in lats:
        for lo in lons:
            ids[(lo, la)] = nid
            els.append({"type": "node", "id": nid, "lon": lo, "lat": la})
            nid += 1
    wid = 10_000
    for la in lats:
        els.append({"type": "way", "id": wid, "nodes": [ids[(lo, la)] for lo in lons]})
        wid += 1
    for lo in lons:
        els.append({"type": "way", "id": wid, "nodes": [ids[(lo, la)] for la in lats]})
        wid += 1
    return {"elements": els}


@pytest.fixture
def area_ll():
    return geo.parse_area(AREA)


@pytest.fixture
def fake_overpass():
    return FakeOverpass()


@pytest.fixture(autouse=True)
def heatflow_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HEATFLOW_HOME", str(tmp_path / "hf"))
    return tmp_path / "hf"
