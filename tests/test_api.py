import io
import json
import time

import numpy as np
import pytest

from heatflow.server import create_app

from .conftest import AREA


@pytest.fixture
def client(fake_overpass):
    app = create_app(client=fake_overpass)
    app.testing = True
    return app.test_client()


CRITERIA = [
    {"id": "ch", "name": "Stacje ładowania", "source": {"type": "osm", "key": "amenity", "values": ["charging_station"]},
     "method": "distance", "metric": "euclidean", "direction": "benefit"},
    {"id": "pk", "name": "Parkingi", "source": {"type": "osm", "key": "amenity", "values": ["parking"]},
     "method": "count", "radius": 300, "direction": "benefit"},
    {"id": "park", "name": "Park (pieszo)", "source": {"type": "osm", "key": "leisure", "values": ["park"]},
     "method": "distance", "metric": "walking", "direction": "cost"},
    {"id": "sh", "name": "Sklepy", "source": {"type": "osm", "key": "shop", "values": None},
     "method": "count", "radius": 500, "direction": "benefit"},
    {"id": "none", "name": "Brak obiektów", "source": {"type": "osm", "key": "amenity", "values": ["nothing"]},
     "method": "distance", "direction": "cost"},
]


def run(client, criteria=CRITERIA, spacing=100):
    r = client.post("/api/analyses", json={"name": "Test", "area": AREA, "grid": {"spacing": spacing},
                                           "max_distance": 1000, "criteria": criteria})
    assert r.status_code == 202, r.get_json()
    jid = r.get_json()["id"]
    for _ in range(200):
        st = client.get(f"/api/analyses/{jid}").get_json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert st["status"] == "done", st
    return jid, client.get(f"/api/analyses/{jid}?result=1").get_json()["result"]


def test_catalog_and_options(client):
    cat = client.get("/api/catalog").get_json()
    assert len(cat["layers"]) > 50
    assert not any(l["osm"]["values"] == ["*"] for l in cat["layers"])
    ids = [m["id"] for m in client.get("/api/options").get_json()["methods"]]
    assert ids == ["spotis", "topsis", "vikor"]


def test_grid_preview(client):
    r = client.post("/api/grid-preview", json={"area": AREA, "spacing": "auto"}).get_json()
    assert r["n_points"] > 50 and r["area_km2"] == pytest.approx(1.47, rel=0.05)


def test_full_analysis_flow(client):
    jid, res = run(client)
    n = res["grid"]["n_points"]
    assert len(res["values"]) == 5 and all(len(v) == n for v in res["values"])
    assert res["grid"]["mask"].count("1") == n
    assert any("Brak obiektów" in w for w in res["warnings"])
    assert res["criteria"][0]["stats"]["n_features"] == 2

    ev = client.post(f"/api/analyses/{jid}/evaluate",
                     json={"method": "spotis", "weights": {"mode": "manual", "values": [35, 20, 25, 15, 5]}}).get_json()
    assert ev["excluded"] == ["none"]
    assert sum(ev["weights"].values()) == pytest.approx(1)
    assert ev["weights"]["none"] == 0
    assert len(ev["scores"]) == n and min(ev["ranks"]) == 1
    assert len(ev["contributions"][0]) == 4

    # zmiana kierunku bez ponownego pobierania
    ev2 = client.post(f"/api/analyses/{jid}/evaluate", json={
        "method": "spotis", "weights": {"mode": "equal"}, "directions": {"ch": "cost"}}).get_json()
    ev3 = client.post(f"/api/analyses/{jid}/evaluate", json={
        "method": "spotis", "weights": {"mode": "equal"}, "directions": {"ch": "benefit"}}).get_json()
    assert ev2["scores"] != ev3["scores"]

    for mode in ({"mode": "rancom", "ranking": [1, 2, 2, 3, 4]}, {"mode": "entropy"}, {"mode": "critic"}):
        r = client.post(f"/api/analyses/{jid}/evaluate", json={"method": "topsis", "weights": mode})
        assert r.status_code == 200, r.get_json()

    # eksport wskazanej wersji oceny – z kierunkiem użytym w tej ocenie
    gj2 = json.loads(client.get(f"/api/analyses/{jid}/export.geojson?eval={ev2['eval_id']}").data)
    ch = next(c for c in gj2["heatflow"]["criteria"] if c["id"] == "ch")
    assert ch["direction"] == "cost" and "mniej = lepiej" in ch["description"]
    assert [f["properties"]["score"] for f in gj2["features"]] == ev2["scores"]
    assert client.get(f"/api/analyses/{jid}/export.csv?eval=nieistnieje").status_code == 404

    gj = json.loads(client.get(f"/api/analyses/{jid}/export.geojson?eval={ev['eval_id']}").data)
    assert len(gj["features"]) == n and "score" in gj["features"][0]["properties"]
    csv_text = client.get(f"/api/analyses/{jid}/export.csv").data.decode()
    assert csv_text.splitlines()[0].startswith("lon,lat,id,ch,pk")


def test_walking_ge_euclidean(client):
    crit = [dict(CRITERIA[2]), dict(CRITERIA[2], id="park_e", metric="euclidean")]
    _, res = run(client, crit)
    walk, eu = res["values"]
    assert all(w >= e - 0.11 for w, e in zip(walk, eu))
    assert sum(walk) > sum(eu)
    # punkty wewnątrz parku mają odległość 0 także pieszo
    inside = [w for w, e in zip(walk, eu) if e == 0]
    assert inside and all(w == 0 for w in inside)


def test_user_dataset_criterion(client):
    csv_raw = b"lat,lon\n53.430,14.550\n53.426,14.542\n"
    r = client.post("/api/datasets", data={"file": (io.BytesIO(csv_raw), "punkty.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 201, r.get_json()
    ds = r.get_json()
    assert ds["n_features"] == 2
    assert client.get("/api/datasets").get_json()[0]["id"] == ds["id"]
    crit = [{"id": "u", "name": "Moje punkty", "source": {"type": "dataset", "dataset_id": ds["id"]},
             "method": "distance", "direction": "cost"}, CRITERIA[0]]
    _, res = run(client, crit)
    assert min(res["values"][0]) < 60
    assert client.delete(f"/api/datasets/{ds['id']}").status_code == 200


def test_validation_errors(client):
    r = client.post("/api/analyses", json={"area": AREA, "criteria": []})
    assert r.status_code == 400 and "kryterium" in r.get_json()["error"]
    r = client.post("/api/analyses", json={"criteria": CRITERIA[:1]})
    assert r.status_code == 400
    r = client.post("/api/datasets", data={"file": (io.BytesIO(b"x"), "a.shp")}, content_type="multipart/form-data")
    assert r.status_code == 400
    bad = b'{"type":"FeatureCollection","features":[null]}'
    r = client.post("/api/datasets", data={"file": (io.BytesIO(bad), "a.geojson")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "nr 1" in r.get_json()["error"]


def test_examples_are_valid(client):
    ex = client.get("/api/examples").get_json()
    assert ex and ex[0]["file"].endswith(".json")
    s = client.get(f"/api/examples/{ex[0]['file']}").get_json()
    assert s["weights"]["values"] == [0.20, 0.25, 0.35, 0.15, 0.05]
    assert client.get("/api/examples/..%2Fpyproject.toml").status_code == 404


def test_frontend_served(client):
    assert b"HeatFlow" in client.get("/").data
    assert client.get("/app.js").status_code == 200


def test_rancom_pairwise_matches_pymcdm(client):
    from pymcdm.weights.subjective import RANCOM
    # odpowiedzi jak w RANCOM(object_names=...): pary (0,1), (0,2), (1,2)
    mac = [[0.5, 1, 1], [0, 0.5, 0.5], [0, 0.5, 0.5]]
    r = client.post("/api/weights/rancom", json={"n": 3, "matrix": mac}).get_json()
    assert r["weights"] == pytest.approx(RANCOM(matrix=np.array(mac))().tolist())
    assert r["scw"] == [2.5, 1.0, 1.0] and r["inconsistent_triads"] == 0
    bad = [[0.5, 1, 1], [1, 0.5, 0.5], [0, 0.5, 0.5]]  # a[1,0] powinno być 0
    assert client.post("/api/weights/rancom", json={"n": 3, "matrix": bad}).status_code == 400
    assert client.post("/api/weights/rancom", json={"n": 3, "matrix": [[0.5]]}).status_code == 400
    sc = client.post("/api/weights/rancom", json={"n": 3, "scoring": [10, 90, 50]}).get_json()
    assert sc["weights"] == pytest.approx(RANCOM(scoring=[10, 90, 50])().tolist())


def test_evaluate_with_rancom_matrix(client):
    jid, res = run(client, CRITERIA[:3])
    mac = [[0.5, 0, 1], [1, 0.5, 1], [0, 0, 0.5]]
    ev = client.post(f"/api/analyses/{jid}/evaluate",
                     json={"method": "spotis", "weights": {"mode": "rancom", "matrix": mac}}).get_json()
    w = [ev["weights_input"][k] for k in ("ch", "pk", "park")]
    assert w == pytest.approx([1.5 / 4.5, 2.5 / 4.5, 0.5 / 4.5])
