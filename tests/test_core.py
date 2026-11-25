import numpy as np
import pytest
import shapely
from pymcdm.methods import SPOTIS
from shapely.geometry import LineString, Point, Polygon

from heatflow import geo, osm
from heatflow.criteria import CriterionError, count_within, euclidean_distance, parse_criterion
from heatflow.features import DatasetError, overpass_to_geometries, parse_csv, parse_geojson
from heatflow.mcda import MCDAError, evaluate
from heatflow.network import build_graph, walking_distance
from heatflow.weights import WeightsError, compute_weights, entropy_weights, rancom_weights

from .conftest import AREA, walking_grid


# ---------------------------------------------------------------- geometria i siatka
def test_grid_spacing_and_mask(area_ll):
    g = geo.make_grid(area_ll, 100)
    lon, lat = g.node_lonlat()
    x, y = g.proj.forward(lon, lat)
    assert g.n_points == g.nx * g.ny  # prostokąt – wszystkie punkty w środku
    assert np.allclose(np.diff(np.unique(np.round(x, 3))), 100, atol=0.01)
    assert np.allclose(np.diff(np.unique(np.round(y, 3))), 100, atol=0.01)


def test_grid_polygon_mask():
    tri = {"type": "Polygon", "coordinates": [[[14.54, 53.425], [14.56, 53.425], [14.54, 53.435], [14.54, 53.425]]]}
    a = geo.parse_area(tri)
    g = geo.make_grid(a, 100)
    full = g.nx * g.ny
    assert 0.35 * full < g.n_points < 0.65 * full


def test_grid_limits(area_ll):
    with pytest.raises(geo.GeoError):
        geo.make_grid(area_ll, 1, max_points=1000)


def test_projection_accuracy():
    """Łuk równoleżnika na φ0 (dokładnie N·cosφ·Δλ na elipsoidzie WGS84) i łuk południka (M·Δφ)."""
    import math
    a, e2, lat = 6378137.0, 6.69437999014e-3, 53.43
    s2 = math.sin(math.radians(lat)) ** 2
    n = a / math.sqrt(1 - e2 * s2)
    m = a * (1 - e2) / (1 - e2 * s2) ** 1.5
    p = geo.LocalProjection(14.55, lat)
    x, _ = p.forward(14.55 + 0.1, lat)
    assert x == pytest.approx(n * math.cos(math.radians(lat)) * math.radians(0.1), rel=1e-9)
    _, y = p.forward(14.55, lat + 0.01)
    assert y == pytest.approx(m * math.radians(0.01), rel=1e-6)
    # ok. 1 km na wschód: znana długość łuku równoleżnika 53,43°N to ok. 66,6 km/stopień
    assert 66_400 < p.kx < 66_800


def test_auto_spacing(area_ll):
    sp = geo.auto_spacing(area_ll, target_points=100)
    assert sp in geo.NICE_SPACINGS
    assert geo.make_grid(area_ll, sp).n_points <= 200


def test_parse_area_rejects_points():
    with pytest.raises(geo.GeoError):
        geo.parse_area({"type": "Point", "coordinates": [14.5, 53.4]})


# ---------------------------------------------------------------- Overpass
def test_tag_filter():
    assert osm.tag_filter("shop", None) == '["shop"]'
    assert osm.tag_filter("amenity", ["cafe"]) == '["amenity"="cafe"]'
    assert osm.tag_filter("highway", ["trunk", "primary"]) == '["highway"~"^(trunk|primary)$"]'
    with pytest.raises(ValueError):
        osm.tag_filter('amenity"];out;', ["x"])
    with pytest.raises(ValueError):
        osm.tag_filter("shop", ["*"])


class _Resp:
    def __init__(self, code, data=None, text=""):
        self.status_code, self._data, self.text = code, data, text

    def json(self):
        if self._data is None:
            raise ValueError
        return self._data


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = 0

    def post(self, url, data, timeout):
        self.calls += 1
        return self.responses.pop(0)


def test_overpass_retry_and_cache(tmp_path):
    sess = _Session([_Resp(504), _Resp(429), _Resp(200, {"elements": [1]})])
    waits = []
    c = osm.OverpassClient(cache_dir=tmp_path, endpoints=["https://a/api", "https://b/api"], session=sess,
                           sleep=waits.append)
    msgs = []
    assert c.query("Q", notify=msgs.append) == {"elements": [1]}
    assert waits == [10] and msgs
    # drugi raz z cache – bez zapytań
    assert c.query("Q") == {"elements": [1]} and sess.calls == 3


def test_overpass_gives_up(tmp_path):
    sess = _Session([_Resp(502)] * 50)
    c = osm.OverpassClient(cache_dir=tmp_path, endpoints=["https://a/api"], session=sess, sleep=lambda s: None,
                           budget_s=30)
    with pytest.raises(osm.OverpassError):
        c.query("Q")


# ---------------------------------------------------------------- obiekty
def test_overpass_geometries():
    data = {"elements": [
        {"type": "node", "id": 1, "lon": 1, "lat": 2, "tags": {}},
        {"type": "way", "id": 2, "tags": {"leisure": "park"},
         "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}, {"lon": 1, "lat": 1}, {"lon": 0, "lat": 0}]},
        {"type": "way", "id": 3, "tags": {"highway": "pedestrian"},
         "geometry": [{"lon": 0, "lat": 0}, {"lon": 1, "lat": 0}, {"lon": 1, "lat": 1}, {"lon": 0, "lat": 0}]},
    ]}
    g = overpass_to_geometries(data, "leisure")
    assert [x.geom_type for x in g] == ["Point", "Polygon", "Polygon"]
    g2 = overpass_to_geometries(data, "highway")
    assert g2[2].geom_type == "LineString"


def test_parse_user_files():
    gj = b'{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[14.5,53.4]}}]}'
    assert parse_geojson(gj)[0].geom_type == "Point"
    csv_raw = "nazwa;lat;lon\nA;53,40;14,50\nB;53.41;14.51\n".encode()
    pts = parse_csv(csv_raw)
    assert len(pts) == 2 and pts[0].x == pytest.approx(14.50)
    with pytest.raises(DatasetError):
        parse_csv(b"a,b\n1,2\n")


# ---------------------------------------------------------------- kryteria
def test_euclidean_distance_geometries():
    geoms = [Point(0, 0), LineString([(100, -50), (100, 50)]), Polygon([(300, 0), (400, 0), (400, 100), (300, 100)])]
    pts = np.array([[0, 10], [90, 0], [350, 50], [5000, 5000]])
    d = euclidean_distance(geoms, pts, max_distance=1000)
    assert d.tolist() == pytest.approx([10, 10, 0, 1000])
    assert euclidean_distance([], pts, 500).tolist() == [500] * 4


def test_count_within():
    geoms = [Point(0, 0), Point(30, 0), Point(200, 0)]
    c = count_within(geoms, np.array([[0, 0], [150, 0]]), radius=50)
    assert c.tolist() == [2, 1]


def test_parse_criterion_validation():
    c = parse_criterion({"name": "Sklepy", "source": {"type": "osm", "key": "shop", "values": ["*"]},
                         "method": "count", "direction": "benefit"}, 0)
    assert c.source["values"] is None  # „*” = dowolna wartość
    with pytest.raises(CriterionError):
        parse_criterion({"name": "X", "source": {"type": "osm", "key": "a"}, "method": "area"}, 0)
    with pytest.raises(CriterionError):
        parse_criterion({"name": "X", "source": {"type": "osm", "key": "a"}, "radius": 1}, 0)


# ---------------------------------------------------------------- odległość piesza
def test_walking_distance_follows_network():
    a = geo.parse_area(AREA)
    proj = geo.projection_for(a)
    graph = build_graph(walking_grid(), proj)
    # obiekt w węźle sieci; punkt na ukos – odległość piesza ≈ suma składowych (metryka „miejska”)
    tx, ty = proj.forward(14.550, 53.430)
    px, py = proj.forward(14.553, 53.433)
    target = [Point(float(tx), float(ty))]
    pts = np.array([[float(px), float(py)]])
    walk, stats = walking_distance(graph, target, pts, max_distance=5000)
    eu = euclidean_distance(target, pts, 5000)
    manhattan = abs(px - tx) + abs(py - ty)
    assert walk[0] == pytest.approx(manhattan, rel=0.02)
    assert walk[0] > eu[0] * 1.2
    assert stats["n_sources"] >= 1


def test_walking_distance_no_targets():
    a = geo.parse_area(AREA)
    graph = build_graph(walking_grid(), geo.projection_for(a))
    walk, _ = walking_distance(graph, [], np.array([[0.0, 0.0]]), 800)
    assert walk.tolist() == [800]


# ---------------------------------------------------------------- wagi
def test_rancom_matches_reference():
    assert rancom_weights([1, 2, 4, 3]) == pytest.approx([0.4375, 0.3125, 0.0625, 0.1875])


def test_entropy_handles_zero_columns():
    rng = np.random.default_rng(1)
    m = np.column_stack([rng.uniform(0, 100, 200), rng.integers(0, 3, 200)])  # druga kolumna z zerami
    w = entropy_weights(m)
    assert w.sum() == pytest.approx(1)
    assert 0 < w[1] < 0.99  # pymcdm dałby tu wagę 1.0


def test_compute_weights_modes():
    m = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0]])
    assert compute_weights({"mode": "manual", "values": [1, 3]}, m, 2).tolist() == [0.25, 0.75]
    assert compute_weights({"mode": "equal"}, m, 2).tolist() == [0.5, 0.5]
    for mode in ("entropy", "critic", "std"):
        assert compute_weights({"mode": mode}, m, 2).sum() == pytest.approx(1)
    with pytest.raises(WeightsError):
        compute_weights({"mode": "manual", "values": [0, 0]}, m, 2)
    with pytest.raises(WeightsError):
        compute_weights({"mode": "rancom", "ranking": [1]}, m, 2)


# ---------------------------------------------------------------- MCDA
def test_spotis_contributions_sum_to_score():
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 1000, (50, 3))
    w = np.array([0.5, 0.3, 0.2])
    t = np.array([1, -1, 1])
    ev = evaluate(m, w, t, "spotis")
    assert ev.contributions.sum(axis=1) == pytest.approx(ev.scores)
    bounds = np.column_stack([m.min(0), m.max(0)])
    assert ev.scores == pytest.approx(SPOTIS(bounds)(m, w, t))
    assert ev.ranks[np.argmin(ev.scores)] == 1


def test_constant_criterion_excluded():
    m = np.array([[1.0, 500], [2.0, 500], [3.0, 500]])
    ev = evaluate(m, np.array([0.5, 0.5]), np.array([1, -1]), "spotis")
    assert ev.used.tolist() == [0] and ev.weights.tolist() == [1.0]
    with pytest.raises(MCDAError):
        evaluate(m[:, [1]], np.array([1.0]), np.array([1]), "spotis")


def test_topsis_direction():
    m = np.array([[10.0, 1], [5.0, 5], [1.0, 10]])
    ev = evaluate(m, np.array([0.5, 0.5]), np.array([1, -1]), "topsis")
    assert ev.ranks.tolist() == [1, 2, 3]
    assert not ev.method.lower_is_better


def test_walking_does_not_jump_across_river():
    """Dwa brzegi bez połączenia w pobliżu; most 1 km dalej. Punkt 60 m od celu przez rzekę."""
    a = geo.parse_area(AREA)
    proj = geo.projection_for(a)
    els, nid, wid = [], 1, 1
    def street(coords):
        nonlocal nid, wid
        ids = []
        for lo, la in coords:
            els.append({"type": "node", "id": nid, "lon": lo, "lat": la}); ids.append(nid); nid += 1
        els.append({"type": "way", "id": wid, "nodes": ids}); wid += 1
        return ids
    lons = [round(14.540 + 0.0005 * i, 6) for i in range(61)]          # ~2 km wzdłuż rzeki
    west = street([(lo, 53.4300) for lo in lons])                      # brzeg południowy
    east = street([(lo, 53.4310) for lo in lons])                      # brzeg północny (~111 m dalej)
    # most na końcu (lon 14.570): łączymy ostatnie węzły obu brzegów
    els.append({"type": "way", "id": wid, "nodes": [west[-1], east[-1]]})
    graph = build_graph({"elements": els}, proj)
    tx, ty = proj.forward(14.545, 53.4310)            # cel na brzegu północnym
    px, py = proj.forward(14.545, 53.43018)           # punkt na brzegu południowym, 20 m od ulicy, ~91 m od celu
    walk, st = walking_distance(graph, [Point(float(tx), float(ty))], np.array([[float(px), float(py)]]), 5000)
    # musi iść przez most: ~1,7 km w jedną stronę + 111 m + ~1,7 km z powrotem
    assert walk[0] > 3000


def test_multipolygon_keeps_holes():
    ring = lambda x0, y0, x1, y1: [{"lon": x0, "lat": y0}, {"lon": x1, "lat": y0}, {"lon": x1, "lat": y1},
                                   {"lon": x0, "lat": y1}, {"lon": x0, "lat": y0}]
    rel = {"type": "relation", "id": 1, "tags": {"type": "multipolygon", "leisure": "park"}, "members": [
        {"type": "way", "role": "outer", "geometry": ring(0, 0, 10, 10)},
        {"type": "way", "role": "inner", "geometry": ring(4, 4, 6, 6)},
    ]}
    g = overpass_to_geometries({"elements": [rel]}, "leisure")[0]
    assert g.area == pytest.approx(100 - 4)
    assert not g.contains(Point(5, 5)) and g.contains(Point(1, 1))


def test_objective_weights_ignore_constant_column(fake_overpass):
    from heatflow.analysis import Analysis
    rng = np.random.default_rng(3)
    m3 = rng.uniform(0, 100, (60, 3))
    m3[:, 1] = m3[:, 0] * 0.8 + rng.uniform(0, 10, 60)  # skorelowane kolumny
    m4 = np.column_stack([m3, np.full(60, 500.0)])
    crit = lambda i: {"id": f"k{i}", "name": f"K{i}", "source": {"type": "osm", "key": "amenity", "values": ["x"]}}
    a3 = Analysis({"area": AREA, "criteria": [crit(i) for i in range(3)]}, client=fake_overpass)
    a4 = Analysis({"area": AREA, "criteria": [crit(i) for i in range(4)]}, client=fake_overpass)
    a3.matrix, a4.matrix = m3, m4
    for mode in ("critic", "entropy", "std"):
        w3 = a3.evaluate("spotis", {"mode": mode})["weights"]
        w4 = a4.evaluate("spotis", {"mode": mode})["weights"]
        assert [w3[f"k{i}"] for i in range(3)] == pytest.approx([w4[f"k{i}"] for i in range(3)])
        assert w4["k3"] == 0


def _street_graph(proj, streets):
    """Graf z listy ulic (listy punktów lon/lat); wspólne punkty = wspólne węzły."""
    els, ids, nid, wid = [], {}, 1, 1
    for coords in streets:
        refs = []
        for lo, la in coords:
            key = (round(lo, 7), round(la, 7))
            if key not in ids:
                ids[key] = nid
                els.append({"type": "node", "id": nid, "lon": lo, "lat": la})
                nid += 1
            refs.append(ids[key])
        els.append({"type": "way", "id": wid, "nodes": refs})
        wid += 1
    return build_graph({"elements": els}, proj)


def test_walking_snaps_to_edge_not_node():
    """Długi chodnik (bez węzłów pośrednich); punkt i cel przy jego środku → odległość wzdłuż chodnika."""
    a = geo.parse_area(AREA)
    proj = geo.projection_for(a)
    graph = _street_graph(proj, [[(14.540, 53.430), (14.552, 53.430)]])  # ~800 m, dwa węzły
    tx, ty = proj.forward(14.5465, 53.4302)   # cel 22 m od chodnika
    px, py = proj.forward(14.5455, 53.42995)  # punkt 6 m od chodnika, ~70 m dalej wzdłuż
    walk, _ = walking_distance(graph, [Point(float(tx), float(ty))], np.array([[float(px), float(py)]]), 1500)
    assert 85 < walk[0] < 110   # 6 + ~67 + 22; wersja „do węzłów” dawała ~800 m


def test_walking_snap_respects_barrier():
    """Ulica tuż za rzeką jest bliżej niż ulica po tej samej stronie – dociągnięcie nie może przejść przez wodę."""
    a = geo.parse_area(AREA)
    proj = geo.projection_for(a)
    south = [(14.540 + 0.001 * i, 53.4300) for i in range(21)]
    north = [(14.540 + 0.001 * i, 53.4310) for i in range(21)]
    bridge = [south[-1], north[-1]]
    graph = _street_graph(proj, [south, north, bridge])
    river_ll = shapely.box(14.535, 53.4303, 14.565, 53.43085)  # rzeka między ulicami
    graph.set_barriers([proj.geom_to_xy(river_ll)])
    tx, ty = proj.forward(14.545, 53.4311)    # cel przy ulicy północnej
    px, py = proj.forward(14.545, 53.43055)   # punkt na brzegu (na wodzie byłby wykluczony) – bliżej ulicy płn.?
    px2, py2 = proj.forward(14.545, 53.43027)  # punkt na lądzie po stronie południowej, 30 m od ulicy płd.
    walk, _ = walking_distance(graph, [Point(float(tx), float(ty))], np.array([[float(px2), float(py2)]]), 5000)
    assert walk[0] > 2000  # musi iść przez most na końcu ulic


def test_objective_weights_respect_direction():
    """Odwrócenie wartości kryterium razem z jego kierunkiem nie może zmienić wag (CRITIC, entropia, std)."""
    from heatflow.weights import critic_weights, entropy_weights, std_weights
    rng = np.random.default_rng(7)
    m = rng.uniform(0, 100, (80, 3))
    m[:, 1] = 0.7 * m[:, 0] + rng.uniform(0, 30, 80)
    flipped = m.copy()
    flipped[:, 1] = 500 - m[:, 1]            # ta sama informacja, przeciwna skala
    for fn in (critic_weights, entropy_weights, std_weights):
        a = fn(m, np.array([1, 1, -1]))
        b = fn(flipped, np.array([1, -1, -1]))
        assert a == pytest.approx(b), fn.__name__


def test_rancom_matrix_validation_and_scoring():
    from heatflow.weights import rancom_details
    with pytest.raises(WeightsError):
        rancom_details({"matrix": [[1, 1], [0, 0]]}, 2)          # zła przekątna
    with pytest.raises(WeightsError):
        rancom_details({"matrix": [[0.5, 0.996], [0, 0.5]]}, 2)  # wartość spoza {0, ½, 1}
    with pytest.raises(WeightsError):
        rancom_details({"matrix": [[0.5, 1], [1, 0.5]]}, 2)      # brak wzajemności
    w = rancom_details({"scoring": [100, 50, 50, 0]}, 4)["weights"]  # zero i remis
    assert w == pytest.approx([7 / 16, 4 / 16, 4 / 16, 1 / 16])


def test_defaults_match_ui(fake_overpass):
    from heatflow.analysis import Analysis
    base = {"area": AREA, "criteria": [{"name": "Sklepy", "source": {"type": "osm", "key": "shop"}, "method": "count"}]}
    a = Analysis(base, client=fake_overpass)
    assert a.exclude == ["water"]                      # brak pola = jak w interfejsie
    assert a.criteria[0].direction == "benefit"        # liczność: więcej = lepiej
    assert Analysis(dict(base, exclude=[]), client=fake_overpass).exclude == []


def test_walking_ignores_tiny_fragments_and_crosses_fence_last():
    a = geo.parse_area(AREA)
    proj = geo.projection_for(a)
    main = [(14.540 + 0.001 * i, 53.4300) for i in range(21)]            # główna ulica (21 węzłów)
    side = [(14.550, 53.4300 + 0.0002 * k) for k in range(40)]            # odnoga na północ (razem ≥ 50 węzłów)
    frag = [(14.5452, 53.4309), (14.5454, 53.4309)]                       # odcięty fragment 2-węzłowy
    graph = _street_graph(proj, [main, side, frag])
    tx, ty = proj.forward(14.545, 53.4300)
    target = [Point(float(tx), float(ty))]
    px, py = proj.forward(14.5453, 53.43085)  # 5 m od fragmentu, ~94 m od ulicy głównej
    walk, _ = walking_distance(graph, target, np.array([[float(px), float(py)]]), 1500)
    assert walk[0] < 200                       # dociągnięcie do sieci głównej, nie „brak trasy”
    # ogrodzenie między punktem a ulicą: brak innej drogi → przekroczenie w ostateczności
    fence = shapely.LineString([proj.forward(14.53, 53.4304), proj.forward(14.57, 53.4304)])
    graph.set_barriers([], [fence])
    walk2, _ = walking_distance(graph, target, np.array([[float(px), float(py)]]), 1500)
    assert walk2[0] < 200
