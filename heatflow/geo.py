"""Geometria: lokalna projekcja metryczna, obszar analizy i regularna siatka punktów."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import shape, mapping

EARTH_RADIUS = 6_371_008.8  # m, średni promień Ziemi
WGS84_A = 6_378_137.0
WGS84_E2 = 6.69437999014e-3

# Dozwolone rozstawy siatki w trybie automatycznym (m)
NICE_SPACINGS = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000]
MAX_POINTS = 20_000


class GeoError(ValueError):
    """Błąd danych przestrzennych zgłaszany użytkownikowi."""


@dataclass(frozen=True)
class LocalProjection:
    """Projekcja równoodległościowa wokół środka obszaru.

    x = N(φ0)·cos(φ0)·(λ−λ0), y = M(φ0)·(φ−φ0), gdzie N i M to promienie krzywizny
    elipsoidy WGS84 w środku obszaru. Przekształcenie jest liniowe w lon/lat, więc siatka
    regularna w metrach jest też regularna w stopniach. Błąd odległości względem geodezji
    rośnie z odległością od środka (stały cos φ0): dla Szczecina z buforem 6 km wynosi
    maks. ok. 0,4% (ok. 0,25% w granicach miasta), czyli kilka metrów na 1,5 km.
    """

    lon0: float
    lat0: float

    @property
    def _sin2(self) -> float:
        return math.sin(math.radians(self.lat0)) ** 2

    @property
    def kx(self) -> float:
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * self._sin2)
        return n * math.cos(math.radians(self.lat0)) * math.pi / 180.0

    @property
    def ky(self) -> float:
        m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * self._sin2) ** 1.5
        return m * math.pi / 180.0

    def forward(self, lon, lat):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def inverse(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return x / self.kx + self.lon0, y / self.ky + self.lat0

    def geom_to_xy(self, geom):
        return shapely.transform(geom, lambda c: np.column_stack(self.forward(c[:, 0], c[:, 1])))

    def geom_to_ll(self, geom):
        return shapely.transform(geom, lambda c: np.column_stack(self.inverse(c[:, 0], c[:, 1])))


def parse_area(area: dict):
    """Zwraca poligon (lon/lat) z GeoJSON (Geometry/Feature/FeatureCollection) albo bbox."""
    if not isinstance(area, dict):
        raise GeoError("Nie wskazano obszaru analizy.")
    if "bbox" in area and area.get("type") is None:
        w, s, e, n = (float(v) for v in area["bbox"])
        geom = shapely.box(min(w, e), min(s, n), max(w, e), max(s, n))
    else:
        t = area.get("type")
        if t == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in area.get("features", []) if f.get("geometry")]
            geom = shapely.union_all(geoms) if geoms else None
        elif t == "Feature":
            geom = shape(area["geometry"])
        else:
            geom = shape(area)
    if geom is None or geom.is_empty:
        raise GeoError("Obszar analizy jest pusty.")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise GeoError("Obszar analizy musi być poligonem (prostokąt, wielokąt lub granica miejscowości).")
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
        polys = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = shapely.union_all(polys)
    w, s, e, n = geom.bounds
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -85 <= s <= 85 and -85 <= n <= 85):
        raise GeoError("Współrzędne obszaru są poza zakresem (oczekiwano lon/lat WGS84).")
    return geom


def projection_for(area_ll) -> LocalProjection:
    c = area_ll.envelope.centroid
    return LocalProjection(lon0=c.x, lat0=c.y)


def area_km2(area_ll) -> float:
    proj = projection_for(area_ll)
    return proj.geom_to_xy(area_ll).area / 1e6


def auto_spacing(area_ll, target_points: int = 3000) -> int:
    """Rozstaw siatki dobrany tak, by obszar pokryło ok. `target_points` punktów."""
    a = area_km2(area_ll) * 1e6
    raw = math.sqrt(a / max(target_points, 1))
    for s in NICE_SPACINGS:
        if s >= raw:
            return s
    return NICE_SPACINGS[-1]


@dataclass
class Grid:
    """Regularna siatka w lon/lat (stały krok w metrach) z maską obszaru."""

    lon_min: float
    lat_min: float
    dlon: float
    dlat: float
    nx: int
    ny: int
    spacing: float
    mask: np.ndarray  # (ny, nx) bool – punkt wewnątrz obszaru
    proj: LocalProjection

    @property
    def n_points(self) -> int:
        return int(self.mask.sum())

    def node_lonlat(self):
        """Współrzędne węzłów wewnątrz obszaru, kolejność wierszami (od południa)."""
        jj, ii = np.nonzero(self.mask)
        lon = self.lon_min + ii * self.dlon
        lat = self.lat_min + jj * self.dlat
        return lon, lat

    def node_xy(self):
        lon, lat = self.node_lonlat()
        return self.proj.forward(lon, lat)

    def to_json(self) -> dict:
        return {
            "lon_min": self.lon_min,
            "lat_min": self.lat_min,
            "dlon": self.dlon,
            "dlat": self.dlat,
            "nx": self.nx,
            "ny": self.ny,
            "spacing": self.spacing,
            "n_points": self.n_points,
            # maska jako ciąg 0/1 wierszami od południa
            "mask": "".join("1" if v else "0" for v in self.mask.ravel()),
        }


def make_grid(area_ll, spacing: float, max_points: int = MAX_POINTS) -> Grid:
    if spacing is None or spacing <= 0:
        raise GeoError("Rozstaw siatki musi być dodatni.")
    proj = projection_for(area_ll)
    area_xy = proj.geom_to_xy(area_ll)
    minx, miny, maxx, maxy = area_xy.bounds
    nx = int(math.floor((maxx - minx) / spacing)) + 1
    ny = int(math.floor((maxy - miny) / spacing)) + 1
    if nx * ny > max_points * 4:
        raise GeoError(
            f"Siatka byłaby zbyt gęsta ({nx}×{ny} węzłów). Zwiększ rozstaw lub zmniejsz obszar."
        )
    # Siatka wyśrodkowana w obwiedni obszaru
    offx = ((maxx - minx) - (nx - 1) * spacing) / 2
    offy = ((maxy - miny) - (ny - 1) * spacing) / 2
    xs = minx + offx + np.arange(nx) * spacing
    ys = miny + offy + np.arange(ny) * spacing
    X, Y = np.meshgrid(xs, ys)
    shapely.prepare(area_xy)
    inside = shapely.contains_xy(area_xy, X.ravel(), Y.ravel()).reshape(ny, nx)
    n_inside = int(inside.sum())
    if n_inside == 0:
        raise GeoError("Żaden punkt siatki nie leży w obszarze. Zmniejsz rozstaw siatki.")
    if n_inside > max_points:
        raise GeoError(
            f"Siatka miałaby {n_inside} punktów (limit {max_points}). Zwiększ rozstaw lub zmniejsz obszar."
        )
    lon_min, lat_min = proj.inverse(xs[0], ys[0])
    lon_1, lat_1 = proj.inverse(xs[0] + spacing, ys[0] + spacing)
    return Grid(
        lon_min=float(lon_min),
        lat_min=float(lat_min),
        dlon=float(lon_1 - lon_min),
        dlat=float(lat_1 - lat_min),
        nx=nx,
        ny=ny,
        spacing=float(spacing),
        mask=inside,
        proj=proj,
    )


def buffered_bbox_ll(area_ll, buffer_m: float):
    """Obwiednia obszaru powiększona o bufor w metrach: (south, west, north, east)."""
    proj = projection_for(area_ll)
    minx, miny, maxx, maxy = proj.geom_to_xy(area_ll).bounds
    w, s = proj.inverse(minx - buffer_m, miny - buffer_m)
    e, n = proj.inverse(maxx + buffer_m, maxy + buffer_m)
    return float(s), float(w), float(n), float(e)


def area_to_geojson(area_ll) -> dict:
    return mapping(area_ll)
