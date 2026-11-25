"""Zamiana danych źródłowych (Overpass JSON, GeoJSON, CSV) na geometrie shapely."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import shapely
from shapely.geometry import LineString, MultiLineString, Point, Polygon, shape
from shapely.ops import polygonize, unary_union

# Klucze, dla których zamknięta linia NIE oznacza powierzchni
LINEAR_KEYS = {"highway", "railway", "power", "waterway", "barrier", "route", "aerialway"}


class DatasetError(ValueError):
    """Błąd pliku z danymi użytkownika."""


def _way_geometry(coords, tags, key):
    if len(coords) < 2:
        return None
    closed = len(coords) >= 4 and coords[0] == coords[-1]
    if closed and ((key not in LINEAR_KEYS and tags.get("area") != "no") or tags.get("area") == "yes"):
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        return poly
    return LineString(coords)


def _relation_geometry(el, key):
    lines = []
    outer, inner = [], []
    for m in el.get("members", []):
        g = m.get("geometry")
        if m.get("type") == "way" and g:
            coords = [(p["lon"], p["lat"]) for p in g if p]
            if len(coords) >= 2:
                ls = LineString(coords)
                lines.append(ls)
                (inner if m.get("role") == "inner" else outer).append(ls)
        elif m.get("type") == "node" and "lon" in m:
            lines.append(Point(m["lon"], m["lat"]))
    if not lines:
        if "center" in el:
            return Point(el["center"]["lon"], el["center"]["lat"])
        return None
    rel_type = el.get("tags", {}).get("type")
    if outer and rel_type in ("multipolygon", "boundary") and key not in LINEAR_KEYS:
        outer_polys = list(polygonize(unary_union(outer)))
        if outer_polys:
            area = unary_union(outer_polys)
            inner_polys = list(polygonize(unary_union(inner))) if inner else []
            if inner_polys:
                area = area.difference(unary_union(inner_polys))  # otwory (np. jezioro w parku)
            if not area.is_empty:
                return area
    return unary_union(lines)


def overpass_to_geometries(data: dict, key: str | None = None) -> list:
    """Geometrie (lon/lat) obiektów z odpowiedzi Overpass (`out geom` lub `out center`)."""
    geoms = []
    for el in data.get("elements", []):
        t = el.get("type")
        tags = el.get("tags", {})
        g = None
        if t == "node" and "lon" in el:
            g = Point(el["lon"], el["lat"])
        elif t == "way":
            if "geometry" in el:
                coords = [(p["lon"], p["lat"]) for p in el["geometry"] if p]
                g = _way_geometry(coords, tags, key)
            elif "center" in el:
                g = Point(el["center"]["lon"], el["center"]["lat"])
        elif t == "relation":
            if "members" in el and any("geometry" in m for m in el["members"]):
                g = _relation_geometry(el, key)
            elif "center" in el:
                g = Point(el["center"]["lon"], el["center"]["lat"])
        if g is not None and not g.is_empty:
            geoms.append(g)
    return geoms


def representative_points_xy(geoms_xy) -> np.ndarray:
    """Punkt reprezentatywny każdego obiektu (do zliczania w promieniu)."""
    if not geoms_xy:
        return np.empty((0, 2))
    pts = shapely.point_on_surface(np.asarray(geoms_xy, dtype=object))
    return np.column_stack([shapely.get_x(pts), shapely.get_y(pts)])


# --- dane użytkownika -------------------------------------------------------

LAT_NAMES = ("lat", "latitude", "y", "szerokosc", "szerokość", "szer", "wgs84_lat")
LON_NAMES = ("lon", "lng", "long", "longitude", "x", "dlugosc", "długość", "dl", "wgs84_lon")


def parse_geojson(raw: bytes) -> list:
    try:
        obj = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Plik nie jest poprawnym GeoJSON: {exc}") from exc
    t = obj.get("type") if isinstance(obj, dict) else None
    if t == "FeatureCollection":
        feats = obj.get("features")
        if not isinstance(feats, list):
            raise DatasetError("FeatureCollection nie zawiera listy 'features'.")
        items = []
        for i, f in enumerate(feats):
            if not isinstance(f, dict):
                raise DatasetError(f"Obiekt nr {i + 1} nie jest poprawnym elementem Feature.")
            items.append(f.get("geometry"))
    elif t == "Feature":
        items = [obj.get("geometry")]
    elif t in ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon", "GeometryCollection"):
        items = [obj]
    else:
        raise DatasetError("Nieobsługiwany typ GeoJSON (oczekiwano FeatureCollection, Feature lub Geometry).")
    geoms = []
    for i, g in enumerate(items):
        if not g:
            continue
        if not isinstance(g, dict):
            raise DatasetError(f"Błędna geometria obiektu nr {i + 1}.")
        try:
            geom = shape(g)
        except Exception as exc:  # noqa: BLE001 – komunikat dla użytkownika
            raise DatasetError(f"Błędna geometria obiektu nr {i + 1}: {exc}") from exc
        if not geom.is_empty:
            geoms.append(geom)
    return geoms


def parse_csv(raw: bytes) -> list:
    text = raw.decode("utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise DatasetError("Plik CSV nie ma nagłówka.")
    norm = {f.strip().lower(): f for f in reader.fieldnames}
    lat_col = next((norm[n] for n in LAT_NAMES if n in norm), None)
    lon_col = next((norm[n] for n in LON_NAMES if n in norm), None)
    if not lat_col or not lon_col:
        raise DatasetError(
            "Nie znaleziono kolumn ze współrzędnymi. Nazwij je np. 'lat' i 'lon' (WGS84)."
        )
    geoms = []
    for i, row in enumerate(reader, start=2):
        try:
            lat = float(str(row[lat_col]).replace(",", "."))
            lon = float(str(row[lon_col]).replace(",", "."))
        except (TypeError, ValueError):
            raise DatasetError(f"Wiersz {i}: niepoprawne współrzędne.") from None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise DatasetError(f"Wiersz {i}: współrzędne poza zakresem WGS84.")
        geoms.append(Point(lon, lat))
    return geoms


def parse_dataset(filename: str, raw: bytes) -> list:
    name = (filename or "").lower()
    if name.endswith((".geojson", ".json")):
        geoms = parse_geojson(raw)
    elif name.endswith((".csv", ".txt", ".tsv")):
        geoms = parse_csv(raw)
    else:
        raise DatasetError("Obsługiwane formaty: GeoJSON (.geojson, .json) i CSV (.csv).")
    if not geoms:
        raise DatasetError("Plik nie zawiera żadnych obiektów.")
    w, s, e, n = shapely.total_bounds(np.asarray(geoms, dtype=object))
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        raise DatasetError("Współrzędne poza zakresem – plik musi być w WGS84 (EPSG:4326).")
    return geoms


def flatten_lines(geom) -> list:
    """Wszystkie linie (także obwody poligonów) jako listy LineString."""
    if geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "LineString":
        return [geom]
    if gt == "LinearRing":
        return [LineString(geom.coords)]
    if gt == "Polygon":
        return [LineString(geom.exterior.coords)] + [LineString(r.coords) for r in geom.interiors]
    if gt in ("MultiLineString", "MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms:
            out.extend(flatten_lines(g))
        return out
    return []


__all__ = [
    "DatasetError", "overpass_to_geometries", "representative_points_xy", "parse_dataset",
    "parse_geojson", "parse_csv", "flatten_lines", "MultiLineString",
]
