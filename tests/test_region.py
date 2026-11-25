"""Import lokalnego regionu z (syntetycznego) pliku OSM i analiza bez Overpass."""
import time

import numpy as np
import pytest

pytest.importorskip("osmium")

from heatflow import region as region_mod  # noqa: E402
from heatflow.analysis import Analysis, AnalysisError, grid_preview  # noqa: E402

LON0, LAT0 = 14.50, 53.40
D = 0.001  # ~66 m (lon) / ~111 m (lat)


def build_osm(path):
    nodes, ways, rels = [], [], []
    nid = [1]

    def node(lon, lat, **tags):
        i = nid[0]
        nid[0] += 1
        nodes.append((i, lon, lat, tags))
        return i

    # granica miasta: kwadrat 20×20 komórek
    b = [node(LON0, LAT0), node(LON0 + 20 * D, LAT0), node(LON0 + 20 * D, LAT0 + 20 * D), node(LON0, LAT0 + 20 * D)]
    ways.append((1000, b + [b[0]], {}))
    rels.append((5000, [("way", 1000, "outer")], {"type": "boundary", "boundary": "administrative",
                                                    "admin_level": "7", "name": "Testowo"}))
    # osiedle w środku (admin_level 10)
    o = [node(LON0 + 2 * D, LAT0 + 2 * D), node(LON0 + 8 * D, LAT0 + 2 * D), node(LON0 + 8 * D, LAT0 + 8 * D),
         node(LON0 + 2 * D, LAT0 + 8 * D)]
    ways.append((1001, o + [o[0]], {}))
    rels.append((5001, [("way", 1001, "outer")], {"type": "boundary", "boundary": "administrative",
                                                    "admin_level": "10", "name": "Osiedle Zielone"}))
    # sieć ulic co 2 komórki
    wid = 2000
    for j in range(0, 21, 2):
        ids = [node(LON0 + i * D, LAT0 + j * D) for i in range(21)]
        ways.append((wid, ids, {"highway": "residential", "name": f"Ulica {j}"}))
        wid += 1
    # ulica poprzeczna łącząca wszystkie (wspólne węzły na skrzyżowaniach)
    rows = {}
    for i_, lo, la, t in nodes:
        if abs(lo - LON0) < 1e-9 and not t:
            rows[round(la, 7)] = i_
    ways.append((wid, [rows[k] for k in sorted(rows)], {"highway": "footway"}))
    # park jako multipoligon z otworem (staw)
    po = [node(LON0 + 12 * D, LAT0 + 12 * D), node(LON0 + 18 * D, LAT0 + 12 * D), node(LON0 + 18 * D, LAT0 + 18 * D),
          node(LON0 + 12 * D, LAT0 + 18 * D)]
    pi = [node(LON0 + 14 * D, LAT0 + 14 * D), node(LON0 + 16 * D, LAT0 + 14 * D), node(LON0 + 16 * D, LAT0 + 16 * D),
          node(LON0 + 14 * D, LAT0 + 16 * D)]
    ways.append((3000, po + [po[0]], {}))
    ways.append((3001, pi + [pi[0]], {}))
    rels.append((6000, [("way", 3000, "outer"), ("way", 3001, "inner")], {"type": "multipolygon", "leisure": "park",
                                                                         "name": "Park Testowy"}))
    # obiekty punktowe
    node(LON0 + 5 * D, LAT0 + 5 * D, amenity="charging_station")
    for k in range(6):
        node(LON0 + (3 + k) * D, LAT0 + 10 * D, shop="bakery" if k % 2 else "convenience")

    def tags_xml(t):
        return "".join(f'<tag k="{k}" v="{v}"/>' for k, v in t.items())

    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<osm version="0.6" generator="test"><bounds minlat="53.3" minlon="14.4" maxlat="53.5" maxlon="14.7"/>']
    for i, lon, lat, t in nodes:
        xml.append(f'<node id="{i}" version="1" lat="{lat:.7f}" lon="{lon:.7f}">{tags_xml(t)}</node>')
    for i, refs, t in ways:
        xml.append(f'<way id="{i}" version="1">' + "".join(f'<nd ref="{r}"/>' for r in refs) + tags_xml(t) + "</way>")
    for i, mem, t in rels:
        xml.append(f'<relation id="{i}" version="1">' +
                   "".join(f'<member type="{ty}" ref="{r}" role="{ro}"/>' for ty, r, ro in mem) + tags_xml(t) + "</relation>")
    xml.append("</osm>")
    path.write_text("\n".join(xml), encoding="utf-8")


@pytest.fixture
def test_region(tmp_path, heatflow_home):
    osm = tmp_path / "test.osm"
    build_osm(osm)
    cfg = {"name": "Testowo", "pbf_url": "", "boundary_relation": 5000, "buffer_m": 3000,
           "district_admin_levels": ["10"]}
    region_mod.import_region("testowo", pbf=str(osm), log=lambda m: None, cfg=cfg)
    region_mod._cache.clear()
    return region_mod.active_region()


def test_import_and_queries(test_region):
    r = test_region
    assert r.name == "Testowo"
    bbox = (LAT0 - 0.01, LON0 - 0.01, LAT0 + 0.03, LON0 + 0.03)
    parks = r.layer_geoms("leisure", ["park"], bbox)
    assert len(parks) == 1 and parks[0].geom_type in ("Polygon", "MultiPolygon")
    from shapely.geometry import Point
    assert not parks[0].contains(Point(LON0 + 15 * D, LAT0 + 15 * D))  # staw = otwór
    assert len(r.layer_geoms("shop", None, bbox)) == 6
    assert len(r.layer_geoms("shop", ["bakery"], bbox)) == 3
    roads = r.layer_geoms("highway", ["residential"], bbox)
    assert all(g.geom_type == "LineString" for g in roads)
    lon, lat, u, v = r.walk_arrays(bbox)
    assert len(u) >= 11 * 20
    res = r.search("zielone")
    assert res and res[0]["type"] == "osiedle" and res[0]["boundary"]
    assert any(x["type"] == "ulica" for x in r.search("Ulica 4"))


def test_analysis_uses_local_region(test_region, fake_overpass):
    area = {"bbox": [LON0 - 0.01, LAT0 - 0.01, LON0 + 0.01, LAT0 + 0.01]}  # częściowo poza miastem
    cfg = {"name": "T", "area": area, "grid": {"spacing": 50}, "max_distance": 1000, "criteria": [
        {"id": "ch", "name": "Ładowarki", "source": {"type": "osm", "key": "amenity", "values": ["charging_station"]},
         "method": "distance", "metric": "walking", "direction": "benefit"},
        {"id": "sh", "name": "Sklepy", "source": {"type": "osm", "key": "shop"}, "method": "count", "radius": 300,
         "direction": "benefit"},
    ]}
    a = Analysis(cfg, client=fake_overpass)
    assert a.region is test_region and a.clipped
    assert any("przycięto" in w for w in a.warnings)
    lon, lat = a.grid.node_lonlat()
    assert lon.min() >= LON0 - 1e-9 and lat.min() >= LAT0 - 1e-9
    a.run()
    assert fake_overpass.queries == []  # żadnych zapytań do Overpass
    assert a.matrix.shape[1] == 2 and a.matrix[:, 1].max() > 0
    assert a.result_json()["data_source"].startswith("lokalne dane")
    ev = a.evaluate("spotis", {"mode": "equal"})
    assert len(ev["scores"]) == a.grid.n_points

    p = grid_preview(area, 100)
    assert p["clipped"] and p["area"]["type"] in ("Polygon", "MultiPolygon")

    with pytest.raises(AnalysisError):
        Analysis(dict(cfg, area={"bbox": [15.0, 54.0, 15.01, 54.01]}), client=fake_overpass)
    with pytest.raises(AnalysisError):  # bufor danych za mały dla 5 km
        Analysis(dict(cfg, max_distance=5000), client=fake_overpass)


def test_specific_objects(test_region, fake_overpass):
    r = test_region
    objs = r.layer_objects("leisure", ["park"])
    assert [o["name"] for o in objs["items"]] == ["Park Testowy"]
    shops = r.layer_objects("shop", None)
    assert shops["items"] == []  # sklepy testowe nie mają nazw
    assert objs["items"][0]["refs"]
    ref = objs["items"][0]["refs"][0]
    bbox = (LAT0, LON0, LAT0 + 0.001, LON0 + 0.001)  # obiekt poza obwiednią – refs ignorują filtr obwiedni
    assert len(r.layer_geoms("leisure", ["park"], bbox, refs=[ref])) == 1
    assert r.layer_geoms("leisure", ["park"], bbox, refs=["w999999"]) == []
    cfg = {"area": {"bbox": [LON0, LAT0, LON0 + 0.02, LAT0 + 0.02]}, "grid": {"spacing": 100}, "criteria": [
        {"id": "p", "name": "Park", "source": {"type": "osm", "key": "leisure", "values": ["park"], "refs": [ref],
                                               "ref_names": ["Park Testowy"]}, "method": "distance"}]}
    a = Analysis(cfg, client=fake_overpass)
    a.run()
    assert a.crit_stats[0]["n_features"] == 1
    assert "wybrane obiekty (1): Park Testowy" in a.criteria[0].describe()


def test_water_excluded(test_region, fake_overpass):
    # park testowy (multipoligon) traktujemy jak „wodę” przez zmianę definicji wykluczenia
    from heatflow import analysis as an
    old = an.EXCLUSIONS["water"]
    an.EXCLUSIONS["water"] = ("Wody", [("leisure", ["park"])])
    try:
        cfg = {"area": {"bbox": [LON0, LAT0, LON0 + 0.02, LAT0 + 0.02]}, "grid": {"spacing": 100}, "exclude": ["water"],
               "criteria": [{"id": "s", "name": "Sklepy", "source": {"type": "osm", "key": "shop"}, "method": "count",
                             "radius": 300, "direction": "benefit"}]}
        a = Analysis(cfg, client=fake_overpass)
        before = a.grid.n_points
        a.run()
        res = a.result_json()
        assert 0 < a.n_excluded < before and a.grid.n_points == before - a.n_excluded
        assert res["excluded"]["geometry"]["type"] in ("Polygon", "MultiPolygon")
        assert a.matrix.shape[0] == a.grid.n_points
        # punkty w stawie (otwór) nie są wykluczone
        from shapely.geometry import Point, shape
        g = shape(res["excluded"]["geometry"])
        assert not g.contains(Point(LON0 + 15 * D, LAT0 + 15 * D))
    finally:
        an.EXCLUSIONS["water"] = old
    with pytest.raises(AnalysisError):
        Analysis(dict(cfg, exclude=["mars"]), client=fake_overpass)
