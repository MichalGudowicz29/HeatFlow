"""Lokalny region danych OSM: import z pliku PBF i zapytania bez Overpass API.

Region = granica miasta (obszar analizy) + bufor (dane z otoczenia, żeby odległości przy
granicy nie były zawyżone). Dane trafiają do SQLite z indeksem przestrzennym R-tree
i indeksem tagów; graf pieszy zapisywany jest osobno jako tablice NumPy.

    heatflow import-region            # Szczecin (domyślnie)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import shapely
from shapely import wkb
from shapely.geometry import LineString, Polygon, mapping

from . import storage
from .features import LINEAR_KEYS
from .geo import LocalProjection
from .osm import USER_AGENT, WALKABLE_HIGHWAYS

REGIONS = {
    "szczecin": {
        "name": "Szczecin",
        "pbf_url": "https://download.geofabrik.de/europe/poland/zachodniopomorskie-latest.osm.pbf",
        "boundary_relation": 2873415,   # gmina miejska Szczecin (boundary=administrative, admin_level=7)
        "buffer_m": 6000,
        # osiedla Szczecina (jednostki pomocnicze) – do wyszukiwarki i wyboru obszaru
        "district_admin_levels": ["9", "10"],
    },
}

# Tagi pomijane przy imporcie (tłumaczenia nazw, metadane edycji, adresy)
SKIP_PREFIXES = ("name:", "alt_name:", "official_name:", "old_name:", "source", "note", "fixme", "FIXME",
                 "wikipedia", "wikidata", "wikimedia", "created_by", "check_date", "addr:", "contact:",
                 "ref:", "description", "image", "website", "url", "phone", "email", "opening_hours",
                 "survey", "mapillary", "operator:", "brand:", "network:", "not:", "teryt:")
WALKABLE = set(WALKABLE_HIGHWAYS)

KIND_POINT, KIND_LINE, KIND_CLOSED, KIND_AREA = 0, 1, 2, 3

# Polskie etykiety typów obiektów w wyszukiwarce
SEARCH_LABELS = {
    "highway": "ulica", "bus_stop": "przystanek", "tram_stop": "przystanek", "platform": "przystanek",
    "stop_position": "przystanek", "stop": "przystanek", "halt": "przystanek kolejowy", "station": "stacja",
    "school": "szkoła", "viewpoint": "punkt widokowy", "construction": "budowa",
    "bicycle_rental": "wypożyczalnia rowerów", "attraction": "atrakcja", "townhall": "urząd",
    "kindergarten": "przedszkole", "university": "uczelnia", "hospital": "szpital", "clinic": "przychodnia",
    "pharmacy": "apteka", "park": "park", "playground": "plac zabaw", "parking": "parking",
    "supermarket": "supermarket", "mall": "centrum handlowe", "restaurant": "restauracja", "cafe": "kawiarnia",
    "church": "kościół", "place_of_worship": "miejsce kultu", "suburb": "dzielnica", "neighbourhood": "osiedle",
    "quarter": "osiedle", "residential": "zabudowa mieszkaniowa", "apartments": "budynek mieszkalny",
    "museum": "muzeum", "theatre": "teatr", "cinema": "kino", "library": "biblioteka", "stadium": "stadion",
    "sports_centre": "obiekt sportowy", "cemetery": "cmentarz", "forest": "las", "water": "woda",
    "hotel": "hotel", "fuel": "stacja paliw", "charging_station": "stacja ładowania", "office": "biuro",
    "yes": "budynek",
}


def regions_dir() -> Path:
    p = storage.home() / "regions"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================ import
def _keep_tag(k: str) -> bool:
    return not k.startswith(SKIP_PREFIXES)


def _download(url: str, dest: Path, log) -> None:
    if dest.exists() and time.time() - dest.stat().st_mtime < 24 * 3600:
        log(f"Używam pobranego pliku {dest.name}")
        return
    log(f"Pobieranie {url} …")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def _boundary(pbf: Path, relation_id: int, district_levels, log):
    import osmium

    wkbf = osmium.geom.WKBFactory()
    city, districts = None, []
    fp = (osmium.FileProcessor(str(pbf))
          .with_areas(osmium.filter.TagFilter(("boundary", "administrative")))
          .with_filter(osmium.filter.EntityFilter(osmium.osm.AREA)))
    for a in fp:
        if a.from_way():
            continue
        try:
            g = wkb.loads(wkbf.create_multipolygon(a), hex=True)
        except Exception:  # noqa: BLE001 – uszkodzony multipoligon
            continue
        if a.orig_id() == relation_id:
            city = g
        elif a.tags.get("admin_level") in district_levels and a.tags.get("name"):
            districts.append((a.tags.get("name"), a.tags.get("admin_level"), g))
    if city is None:
        raise RuntimeError(f"Nie znaleziono granicy (relacja {relation_id}) w pliku {pbf.name}.")
    districts = [(n, lvl, g) for n, lvl, g in districts if g.representative_point().within(city)]
    log(f"Granica miasta: {len(districts)} osiedli/dzielnic w środku")
    return city, districts


def import_region(region_id: str = "szczecin", pbf: str | None = None, log=print, cfg: dict | None = None) -> dict:
    import osmium

    import shutil
    import uuid as _uuid

    cfg = cfg or REGIONS[region_id]
    final = regions_dir() / region_id
    # import do katalogu tymczasowego i podmiana całego regionu naraz – działający serwer
    # nigdy nie widzi mieszanki starych i nowych plików
    out = regions_dir() / f".{region_id}.staging-{_uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True)
    src = Path(pbf) if pbf else storage.home() / "sources" / Path(cfg["pbf_url"]).name
    if not pbf:
        _download(cfg["pbf_url"], src, log)
    t0 = time.time()
    header = osmium.io.Reader(str(src), osmium.osm.osm_entity_bits.NOTHING).header()
    osm_ts = header.get("osmosis_replication_timestamp") or None

    log("Wyznaczanie granicy miasta…")
    city, districts = _boundary(src, cfg["boundary_relation"], cfg["district_admin_levels"], log)
    c = city.centroid
    proj = LocalProjection(c.x, c.y)
    coverage = proj.geom_to_ll(proj.geom_to_xy(city).buffer(cfg["buffer_m"], quad_segs=8))
    w, s, e, n = coverage.bounds
    shapely.prepare(coverage)

    db_path = out / "osm.sqlite"
    tmp_db = out / "osm.sqlite.tmp"
    tmp_db.unlink(missing_ok=True)
    db = sqlite3.connect(tmp_db)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE objects(id INTEGER PRIMARY KEY, osm TEXT, kind INTEGER, geom BLOB, name TEXT);
        CREATE TABLE tags(obj INTEGER, k TEXT, v TEXT);
        CREATE VIRTUAL TABLE objects_rtree USING rtree(id, minx, maxx, miny, maxy);
    """)
    wkbf = osmium.geom.WKBFactory()
    rows, tag_rows, rt_rows = [], [], []
    next_id = 1
    node_index: dict[int, int] = {}
    node_lon, node_lat, eu, ev = [], [], [], []

    def flush():
        db.executemany("INSERT INTO objects VALUES (?,?,?,?,?)", rows)
        db.executemany("INSERT INTO tags VALUES (?,?,?)", tag_rows)
        db.executemany("INSERT INTO objects_rtree VALUES (?,?,?,?,?)", rt_rows)
        rows.clear(), tag_rows.clear(), rt_rows.clear()

    def add(osm_ref, kind, geom, tags):
        nonlocal next_id
        oid = next_id
        next_id += 1
        b = geom.bounds
        rows.append((oid, osm_ref, kind, geom.wkb, tags.get("name")))
        tag_rows.extend((oid, k, v) for k, v in tags.items() if _keep_tag(k))
        rt_rows.append((oid, b[0], b[2], b[1], b[3]))
        if len(rows) >= 20000:
            flush()

    def inside_bbox(lon, lat):
        return w <= lon <= e and s <= lat <= n

    log("Import obiektów OSM (to potrwa ok. minuty)…")
    fp = (osmium.FileProcessor(str(src))
          .with_locations()
          .with_areas()  # wszystkie relacje multipolygon/boundary – działa dla dowolnego tagu
          .with_filter(osmium.filter.EmptyTagFilter()))
    n_obj = 0
    for o in fp:
        if o.is_node():
            loc = o.location
            if loc.valid() and inside_bbox(loc.lon, loc.lat):
                tags = {t.k: t.v for t in o.tags}
                add(f"n{o.id}", KIND_POINT, shapely.Point(loc.lon, loc.lat), tags)
                n_obj += 1
        elif o.is_way():
            coords, ids = [], []
            for nd in o.nodes:
                if nd.location.valid():
                    coords.append((nd.location.lon, nd.location.lat))
                    ids.append(nd.ref)
            if len(coords) < 2:
                continue
            if not any(inside_bbox(x, y) for x, y in coords):
                continue
            tags = {t.k: t.v for t in o.tags}
            closed = len(coords) >= 4 and o.nodes[0].ref == o.nodes[-1].ref
            add(f"w{o.id}", KIND_CLOSED if closed else KIND_LINE, LineString(coords), tags)
            n_obj += 1
            hw = tags.get("highway")
            foot = tags.get("foot")
            foot_ok = foot in ("yes", "designated", "permissive", "official") or (
                foot not in ("no", "private", "use_sidepath") and tags.get("access") not in ("no", "private"))
            if hw in WALKABLE and foot_ok and tags.get("area") != "yes":
                idx = []
                for ref, (x, y) in zip(ids, coords):
                    k = node_index.get(ref)
                    if k is None:
                        k = len(node_lon)
                        node_index[ref] = k
                        node_lon.append(x)
                        node_lat.append(y)
                    idx.append(k)
                eu.extend(idx[:-1])
                ev.extend(idx[1:])
        elif o.is_area():
            if o.from_way():
                continue  # zamknięte drogi zapisane jako linie – poligon tworzony przy zapytaniu
            try:
                g = wkb.loads(wkbf.create_multipolygon(o), hex=True)
            except Exception:  # noqa: BLE001
                continue
            bw, bs, be, bn = g.bounds
            if be < w or bw > e or bn < s or bs > n:
                continue
            add(f"r{o.orig_id()}", KIND_AREA, g, {t.k: t.v for t in o.tags})
            n_obj += 1
    flush()
    log("Budowa indeksów…")
    db.executescript("""
        CREATE INDEX tags_kv ON tags(k, v, obj);
        CREATE INDEX tags_obj ON tags(obj);
        CREATE INDEX objects_name ON objects(name) WHERE name IS NOT NULL;
    """)
    db.commit()
    db.close()
    tmp_db.replace(db_path)

    np.savez_compressed(out / "walk.npz", lon=np.asarray(node_lon), lat=np.asarray(node_lat),
                        u=np.asarray(eu, dtype=np.int32), v=np.asarray(ev, dtype=np.int32))
    meta = {
        "id": region_id,
        "name": cfg["name"],
        "osm_timestamp": osm_ts,
        "source": cfg["pbf_url"] if not pbf else str(pbf),
        "imported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "buffer_m": cfg["buffer_m"],
        "boundary": mapping(city),
        "coverage_bbox": [w, s, e, n],
        "n_objects": n_obj,
        "walk_nodes": len(node_lon),
        "walk_edges": len(eu),
        "districts": [{"name": nm, "admin_level": lvl, "geometry": mapping(shapely.simplify(g, 0.00002))}
                      for nm, lvl, g in sorted(districts)],
    }
    (out / "region.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    old = None
    if final.exists():
        old = final.with_name(f".{region_id}.old-{_uuid.uuid4().hex[:8]}")
        final.rename(old)
    out.rename(final)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)  # otwarte pliki (działające analizy) zostają czytelne
    log(f"Gotowe w {time.time() - t0:.0f} s: {n_obj} obiektów, sieć piesza {len(node_lon)} węzłów, "
        f"{len(eu)} odcinków, {len(districts)} osiedli. Stan OSM: {osm_ts}")
    return meta


# ============================================================================ odczyt
class LocalRegion:
    """Źródło danych dla analiz w granicach zaimportowanego regionu."""

    def __init__(self, path: Path):
        import threading

        self.path = Path(path)
        self.version = (self.path / "region.json").stat().st_mtime_ns
        self.meta = json.loads((self.path / "region.json").read_text(encoding="utf-8"))
        self.boundary = shapely.geometry.shape(self.meta["boundary"])
        shapely.prepare(self.boundary)
        self.coverage_bbox = tuple(self.meta["coverage_bbox"])
        # jedno połączenie i graf wczytane od razu: cała analiza korzysta z tej samej wersji
        # danych, nawet jeśli w trakcie zostanie zaimportowany nowy region
        self._con = sqlite3.connect(f"file:{self.path / 'osm.sqlite'}?mode=ro", uri=True, check_same_thread=False)
        self._lock = threading.Lock()
        z = np.load(self.path / "walk.npz")
        self._walk = {k: z[k] for k in ("lon", "lat", "u", "v")}

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def osm_timestamp(self) -> str | None:
        return self.meta.get("osm_timestamp")

    def _query(self, sql, params):
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def covers(self, bbox) -> bool:
        s, w, n, e = bbox
        cw, cs, ce, cn = self.coverage_bbox
        return w >= cw and e <= ce and s >= cs and n <= cn

    def layer_geoms(self, key: str, values, bbox, refs=None) -> list:
        """Geometrie (lon/lat) obiektów z tagiem key[=values] w obwiedni (s, w, n, e).

        `refs` (np. ["w123", "n45"]) zawęża warstwę do wskazanych obiektów (bez filtra obwiedni).
        """
        s, w, n, e = bbox
        sql = ("SELECT o.kind, o.geom, (SELECT v FROM tags a WHERE a.obj = o.id AND a.k = 'area') "
               "FROM tags t JOIN objects o ON o.id = t.obj JOIN objects_rtree r ON r.id = o.id "
               "WHERE t.k = ?")
        params: list = [key]
        if refs:
            sql += f" AND o.osm IN ({','.join('?' * len(refs))})"
            params += list(refs)
        else:
            sql += " AND r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"
            params += [w, e, s, n]
        if values:
            sql += f" AND t.v IN ({','.join('?' * len(values))})"
            params += list(values)
        out = []
        for kind, blob, area_tag in self._query(sql, params):
            g = wkb.loads(blob)
            if kind == KIND_CLOSED and ((key not in LINEAR_KEYS and area_tag != "no") or area_tag == "yes"):
                poly = Polygon(g.coords)
                g = poly if poly.is_valid else shapely.make_valid(poly)
            out.append(g)
        return out

    def walk_arrays(self, bbox):
        """Węzły (lon, lat) i krawędzie sieci pieszej w obwiedni (s, w, n, e)."""
        d = self._walk
        s, w, n, e = bbox
        inside = (d["lon"] >= w) & (d["lon"] <= e) & (d["lat"] >= s) & (d["lat"] <= n)
        keep = inside[d["u"]] | inside[d["v"]]
        return d["lon"], d["lat"], d["u"][keep], d["v"][keep]

    def layer_objects(self, key: str, values, q: str = "", limit: int = 300) -> dict:
        """Nazwane obiekty warstwy w granicach miasta – do wskazania konkretnych obiektów."""
        sql = ("SELECT o.osm, o.name, o.geom FROM tags t JOIN objects o ON o.id = t.obj "
               "WHERE t.k = ? AND o.name IS NOT NULL")
        params: list = [key]
        if values:
            sql += f" AND t.v IN ({','.join('?' * len(values))})"
            params += list(values)
        if q:
            sql += " AND lower(o.name) LIKE ?"
            params.append(f"%{q.lower()}%")
        sql += " ORDER BY o.name LIMIT 2000"
        groups: dict[str, dict] = {}
        rows = self._query(sql, params)
        cnt = "SELECT count(*) FROM tags t WHERE t.k = ?" + (
            f" AND t.v IN ({','.join('?' * len(values))})" if values else "")
        total = self._query(cnt, [key] + list(values or []))[0][0]
        for ref, name, blob in rows:
            g = wkb.loads(blob)
            if not g.intersects(self.boundary):
                continue
            # obiekty o tej samej nazwie (np. budynki jednej uczelni) wybiera się razem
            grp = groups.get(name)
            if grp is None:
                if len(groups) >= limit:
                    continue
                c = g.representative_point()
                grp = groups[name] = {"name": name, "refs": [], "lat": round(c.y, 6), "lon": round(c.x, 6)}
            grp["refs"].append(ref)
        items = list(groups.values())
        return {"items": items, "total_in_data": total, "truncated": len(items) >= limit}

    def search(self, q: str, limit: int = 8) -> list[dict]:
        ql = q.strip().lower()
        out = []
        for dist in self.meta.get("districts", []):
            if ql in dist["name"].lower():
                g = shapely.geometry.shape(dist["geometry"])
                w, s, e, n = g.bounds
                out.append({"name": f"{dist['name']}, {self.name}", "type": "osiedle", "category": "boundary",
                            "lat": g.centroid.y, "lon": g.centroid.x, "bbox": [w, s, e, n],
                            "boundary": dist["geometry"]})
        if len(out) < limit:
            rows = self._query(
                "SELECT o.name, o.kind, o.geom, "
                "(SELECT k || '=' || v FROM tags t WHERE t.obj = o.id AND t.k IN "
                "('place','railway','public_transport','highway','amenity','shop','leisure','tourism',"
                "'landuse','building') ORDER BY CASE t.k WHEN 'place' THEN 0 WHEN 'railway' THEN 1 "
                "WHEN 'public_transport' THEN 2 ELSE 3 END LIMIT 1) "
                "FROM objects o WHERE o.name IS NOT NULL AND lower(o.name) LIKE ? LIMIT 400",
                (f"%{ql}%",))
            seen = set()
            for name, kind, blob, tag in sorted(rows, key=lambda r: (not r[0].lower().startswith(ql), len(r[0]))):
                g = wkb.loads(blob)
                if not g.intersects(self.boundary):
                    continue
                k_, _, v_ = (tag or "=").partition("=")
                if k_ == "highway" and v_ not in ("bus_stop", "platform"):
                    label = "ulica"
                else:
                    label = SEARCH_LABELS.get(v_, SEARCH_LABELS.get(k_, v_.replace("_", " ") or "obiekt"))
                k = (name.lower(), label)
                if k in seen:
                    continue
                seen.add(k)
                w, s, e, n = g.bounds
                pad = 0.002
                c = g.representative_point()
                out.append({"name": f"{name}, {self.name}", "type": label, "category": "local",
                            "lat": c.y, "lon": c.x, "bbox": [w - pad, s - pad, e + pad, n + pad], "boundary": None})
                if len(out) >= limit:
                    break
        return out[:limit]

    def info(self) -> dict:
        m = self.meta
        return {"id": m["id"], "name": m["name"], "osm_timestamp": m.get("osm_timestamp"),
                "imported": m.get("imported"), "boundary": m["boundary"], "buffer_m": m.get("buffer_m"),
                "n_objects": m.get("n_objects"), "n_districts": len(m.get("districts", []))}


_cache: dict[str, LocalRegion] = {}


def active_region() -> LocalRegion | None:
    """Region wskazany zmienną HEATFLOW_REGION (domyślnie: jedyny/pierwszy zaimportowany)."""
    rid = os.environ.get("HEATFLOW_REGION")
    if rid == "none":
        return None
    base = regions_dir()
    cands = [base / rid] if rid else sorted(p for p in base.iterdir()
                                            if not p.name.startswith(".") and (p / "region.json").exists())
    for p in cands:
        if (p / "region.json").exists() and (p / "osm.sqlite").exists():
            key = str(p)
            cached = _cache.get(key)
            if cached is None or cached.version != (p / "region.json").stat().st_mtime_ns:
                _cache[key] = LocalRegion(p)  # nowy import → nowa wersja regionu
            return _cache[key]
    return None
