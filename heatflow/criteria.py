"""Definicje kryteriów i obliczanie ich wartości w punktach siatki."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial import cKDTree

from .features import representative_points_xy

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
if not CATALOG_PATH.exists():  # instalacja jako pakiet
    CATALOG_PATH = Path(__file__).resolve().parent / "data" / "catalog.json"

METHODS = ("distance", "count")
DIRECTIONS = ("benefit", "cost")
METRICS = ("euclidean", "walking")
DEFAULT_RADIUS = 500.0
DEFAULT_MAX_DISTANCE = 1500.0


class CriterionError(ValueError):
    pass


def load_catalog(path: Path = CATALOG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Criterion:
    id: str
    name: str
    method: str
    direction: str
    source: dict
    metric: str = "euclidean"
    radius: float = DEFAULT_RADIUS
    max_distance: float = DEFAULT_MAX_DISTANCE
    meta: dict = field(default_factory=dict)

    @property
    def type_sign(self) -> int:
        """Kierunek w konwencji pymcdm: 1 = więcej lepiej, -1 = mniej lepiej."""
        return 1 if self.direction == "benefit" else -1

    @property
    def unit(self) -> str:
        return "m" if self.method == "distance" else "szt."

    def fetch_buffer(self) -> float:
        if self.method == "count":
            return self.radius
        return self.max_distance

    def describe(self) -> str:
        src = self.source
        if src["type"] == "osm":
            vals = src.get("values")
            tag = f'{src["key"]}={"|".join(vals)}' if vals else f'{src["key"]}=*'
            if src.get("refs"):
                names = src.get("ref_names") or []
                tag += f" – wybrane obiekty ({len(src['refs'])}): " + ", ".join(names[:5]) + ("…" if len(names) > 5 else "")
        else:
            tag = f'plik: {src.get("name") or src.get("dataset_id")}'
        if self.method == "count":
            how = f"liczba obiektów w promieniu {self.radius:g} m"
        else:
            how = ("odległość piesza" if self.metric == "walking" else "odległość w linii prostej") + \
                  f" do najbliższego (maks. {self.max_distance:g} m)"
        kier = "więcej = lepiej" if self.direction == "benefit" else "mniej = lepiej"
        return f"{tag}; {how}; {kier}"

    def to_json(self) -> dict:
        return {
            "id": self.id, "name": self.name, "method": self.method, "direction": self.direction,
            "metric": self.metric, "radius": self.radius, "max_distance": self.max_distance,
            "source": self.source, "unit": self.unit, "description": self.describe(),
        }


def parse_criterion(d: dict, index: int, default_max_distance: float = DEFAULT_MAX_DISTANCE) -> Criterion:
    if not isinstance(d, dict):
        raise CriterionError(f"Kryterium nr {index + 1}: niepoprawna definicja.")
    name = str(d.get("name") or "").strip() or f"Kryterium {index + 1}"
    method = d.get("method", "distance")
    # domyślny kierunek jak w interfejsie: liczność – więcej = lepiej, odległość – bliżej = lepiej
    direction = d.get("direction") or ("benefit" if method == "count" else "cost")
    metric = d.get("metric", "euclidean") if method == "distance" else "euclidean"
    if method not in METHODS:
        raise CriterionError(f"„{name}”: nieznana metoda {method!r}.")
    if direction not in DIRECTIONS:
        raise CriterionError(f"„{name}”: nieznany kierunek {direction!r}.")
    if metric not in METRICS:
        raise CriterionError(f"„{name}”: nieznana metryka odległości {metric!r}.")
    src = d.get("source") or {}
    if src.get("type") == "osm":
        key = str(src.get("key") or "").strip()
        if not key:
            raise CriterionError(f"„{name}”: brak klucza tagu OSM.")
        vals = src.get("values")
        if isinstance(vals, str):
            vals = [v.strip() for v in vals.split("|")]
        vals = [str(v).strip() for v in (vals or []) if str(v).strip() and str(v).strip() != "*"] or None
        source = {"type": "osm", "key": key, "values": vals}
        refs = src.get("refs")
        if refs:
            if not isinstance(refs, list) or len(refs) > 200 or not all(
                    isinstance(r, str) and len(r) > 1 and r[0] in "nwr" and r[1:].isdigit() for r in refs):
                raise CriterionError(f"„{name}”: niepoprawna lista wybranych obiektów.")
            source["refs"] = refs
            source["ref_names"] = [str(x)[:120] for x in (src.get("ref_names") or [])][:200]
    elif src.get("type") == "dataset":
        if not src.get("dataset_id"):
            raise CriterionError(f"„{name}”: brak wskazanego pliku z danymi.")
        source = {"type": "dataset", "dataset_id": str(src["dataset_id"]), "name": src.get("name")}
    else:
        raise CriterionError(f"„{name}”: źródło danych musi być tagiem OSM albo wgranym plikiem.")
    try:
        radius = float(d.get("radius") or DEFAULT_RADIUS)
        max_distance = float(d.get("max_distance") or default_max_distance)
    except (TypeError, ValueError):
        raise CriterionError(f"„{name}”: promień i maks. odległość muszą być liczbami.") from None
    if not (10 <= radius <= 10_000):
        raise CriterionError(f"„{name}”: promień musi mieścić się w zakresie 10–10 000 m.")
    if not (50 <= max_distance <= 20_000):
        raise CriterionError(f"„{name}”: maks. odległość musi mieścić się w zakresie 50–20 000 m.")
    return Criterion(
        id=str(d.get("id") or f"c{index + 1}"), name=name, method=method, direction=direction,
        source=source, metric=metric, radius=radius, max_distance=max_distance,
    )


def euclidean_distance(geoms_xy, points_xy: np.ndarray, max_distance: float) -> np.ndarray:
    """Odległość (m) od punktów do najbliższej geometrii; wewnątrz poligonu = 0."""
    out = np.full(len(points_xy), float(max_distance))
    if not geoms_xy:
        return out
    tree = shapely.STRtree(geoms_xy)
    pts = shapely.points(points_xy)
    (pi, _), d = tree.query_nearest(pts, max_distance=max_distance, return_distance=True, all_matches=False)
    out[pi] = np.minimum(d, max_distance)
    return out


def count_within(geoms_xy, points_xy: np.ndarray, radius: float) -> np.ndarray:
    rep = representative_points_xy(geoms_xy)
    if rep.size == 0:
        return np.zeros(len(points_xy))
    tree = cKDTree(rep)
    return np.asarray(tree.query_ball_point(points_xy, r=radius, return_length=True), dtype=float)
