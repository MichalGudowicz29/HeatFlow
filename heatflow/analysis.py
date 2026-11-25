"""Przebieg analizy: obszar → siatka → dane → macierz decyzyjna → ocena MCDA."""
from __future__ import annotations

import time
import uuid
from typing import Callable

import numpy as np
import shapely

from . import geo, storage
from .criteria import (
    DEFAULT_MAX_DISTANCE, Criterion, CriterionError, count_within, euclidean_distance, parse_criterion,
)
from .features import overpass_to_geometries, representative_points_xy
from .mcda import METHODS, constant_columns, evaluate
from .network import SNAP_MAX, build_graph, walking_distance
from .network import build_graph_arrays
from .osm import OverpassClient, layer_query, osm_timestamp, walking_network_query
from .region import LocalRegion, active_region
from .weights import MODES, OBJECTIVE, compute_weights

MAX_CRITERIA = 15

# Bariery dla dojścia do sieci pieszej (odcinek dociągnięcia nie może ich przecinać)
WALK_BARRIERS = [("natural", ["water"]), ("waterway", ["riverbank"]), ("landuse", ["reservoir", "basin"]),
                 ("railway", ["rail"]), ("highway", ["motorway", "motorway_link"])]
# Bariery miękkie: omijane, gdy się da; przekraczane w ostateczności (OSM rzadko ma wszystkie bramy)
WALK_SOFT_BARRIERS = [("barrier", ["fence", "wall", "city_wall", "retaining_wall"])]

# Tereny, które można wyłączyć z oceny (punkty nie są oceniane, na mapie są szare)
EXCLUSIONS = {
    "water": ("Wody", [("natural", ["water"]), ("waterway", ["riverbank"]), ("landuse", ["reservoir", "basin"])]),
    "forest": ("Lasy", [("landuse", ["forest"]), ("natural", ["wood"])]),
    "cemetery": ("Cmentarze", [("landuse", ["cemetery"]), ("amenity", ["grave_yard"])]),
}
PREVIEW_LIMIT = 4000
WALKING_MAX_AREA_KM2 = 400.0


class AnalysisError(ValueError):
    pass


def _project(geoms_ll, proj):
    if not geoms_ll:
        return []
    arr = np.asarray(geoms_ll, dtype=object)
    out = shapely.transform(arr, lambda c: np.column_stack(proj.forward(c[:, 0], c[:, 1])))
    return list(out)


def grid_preview(area: dict, spacing) -> dict:
    """Szybki podgląd siatki (bez pobierania danych) – do formularza."""
    area_ll, clipped = clip_to_region(geo.parse_area(area), active_region())
    km2 = geo.area_km2(area_ll)
    sp = geo.auto_spacing(area_ll) if spacing in (None, "auto", "") else float(spacing)
    grid = geo.make_grid(area_ll, sp)
    return {"area_km2": round(km2, 3), "spacing": sp, "n_points": grid.n_points, "clipped": clipped,
            "area": geo.area_to_geojson(area_ll) if clipped else None}


def clip_to_region(area_ll, region: LocalRegion | None):
    """Obszar analizy przycięty do granicy regionu (np. miasta)."""
    if region is None:
        return area_ll, False
    clipped = area_ll.intersection(region.boundary)
    polys = [g for g in getattr(clipped, "geoms", [clipped]) if g.geom_type in ("Polygon", "MultiPolygon")]
    clipped = shapely.union_all(polys) if polys else None
    if clipped is None or clipped.is_empty or clipped.area < 1e-10:
        raise AnalysisError(f"Wybrany obszar leży poza granicą miasta {region.name}.")
    return clipped, clipped.area < area_ll.area * 0.999


class Analysis:
    def __init__(self, config: dict, client: OverpassClient | None = None, region: LocalRegion | None | bool = True):
        self.config = config or {}
        self.region = active_region() if region is True else (region or None)
        self.client = client or OverpassClient(cache_dir=storage.cache_dir())
        self.name = str(self.config.get("name") or "Analiza HeatFlow")[:200]
        self.goal = str(self.config.get("goal") or "")[:2000]
        self.area_ll, self.clipped = clip_to_region(geo.parse_area(self.config.get("area")), self.region)
        g = self.config.get("grid") or {}
        spacing = g.get("spacing", "auto")
        self.spacing = geo.auto_spacing(self.area_ll) if spacing in (None, "auto", "") else float(spacing)
        try:
            default_max = float(self.config.get("max_distance") or DEFAULT_MAX_DISTANCE)
        except (TypeError, ValueError):
            raise AnalysisError("Maksymalna odległość musi być liczbą.") from None
        crits = self.config.get("criteria") or []
        if not crits:
            raise AnalysisError("Wybierz co najmniej jedno kryterium.")
        if len(crits) > MAX_CRITERIA:
            raise AnalysisError(f"Maksymalnie {MAX_CRITERIA} kryteriów w jednej analizie.")
        self.criteria: list[Criterion] = [parse_criterion(c, i, default_max) for i, c in enumerate(crits)]
        ids = [c.id for c in self.criteria]
        if len(set(ids)) != len(ids):
            raise AnalysisError("Identyfikatory kryteriów muszą być unikalne.")
        if any(c.metric == "walking" for c in self.criteria):
            km2 = geo.area_km2(self.area_ll)
            if km2 > WALKING_MAX_AREA_KM2:
                raise AnalysisError(
                    f"Odległość piesza jest dostępna dla obszarów do {WALKING_MAX_AREA_KM2:g} km² "
                    f"(wybrany: {km2:.0f} km²). Użyj odległości w linii prostej lub zmniejsz obszar."
                )
        self.grid = geo.make_grid(self.area_ll, self.spacing)
        # brak pola = wartość domyślna interfejsu (wody wyłączone); jawne [] = bez wykluczeń
        excl = self.config["exclude"] if "exclude" in self.config else ["water"]
        excl = excl or []
        if not isinstance(excl, list) or any(e not in EXCLUSIONS for e in excl):
            raise AnalysisError(f"Nieznany rodzaj terenu do wykluczenia. Dostępne: {', '.join(EXCLUSIONS)}.")
        self.exclude: list[str] = list(dict.fromkeys(excl))
        self.excluded_geojson: dict | None = None
        self.n_excluded = 0
        self.matrix: np.ndarray | None = None
        self.crit_stats: list[dict] = []
        self.previews: dict[str, list] = {}
        self.warnings: list[str] = []
        self.duration_s: float | None = None
        self.osm_timestamps: set[str] = set()
        if self.region is not None:
            if self.region.osm_timestamp:
                self.osm_timestamps.add(self.region.osm_timestamp)
            need = max(c.fetch_buffer() for c in self.criteria) + SNAP_MAX
            if not self.region.covers(geo.buffered_bbox_ll(self.area_ll, need)):
                raise AnalysisError(
                    f"Zasięg odległości/promienia wykracza poza zaimportowane dane regionu "
                    f"(bufor {self.region.meta.get('buffer_m', 0):g} m od granicy). Zmniejsz maks. odległość."
                )
            if self.clipped:
                self.warnings.append(f"Obszar przycięto do granicy miasta {self.region.name}.")

    @property
    def data_source(self) -> str:
        if self.region is not None:
            return f"lokalne dane OSM – {self.region.name}"
        return "OpenStreetMap (Overpass API)"

    # ------------------------------------------------------------------
    def _geoms_for(self, c: Criterion, notify=None):
        if c.source["type"] == "dataset":
            try:
                return storage.load_dataset_geoms(c.source["dataset_id"])
            except FileNotFoundError:
                raise AnalysisError(f"„{c.name}”: nie znaleziono wgranego pliku z danymi.") from None
        bbox = geo.buffered_bbox_ll(self.area_ll, c.fetch_buffer())
        if self.region is not None:
            return self.region.layer_geoms(c.source["key"], c.source.get("values"), bbox, c.source.get("refs"))
        if c.source.get("refs"):
            raise AnalysisError(f"„{c.name}”: wskazanie konkretnych obiektów wymaga lokalnych danych regionu "
                                "(heatflow import-region).")
        q = layer_query(c.source["key"], c.source.get("values"), bbox, geometry=(c.method == "distance"))
        data = self.client.query(q, notify=notify)
        if ts := osm_timestamp(data):
            self.osm_timestamps.add(ts)
        return overpass_to_geometries(data, c.source["key"])

    def _layer_polys_lines(self, specs, bbox, report, polygons_only=False) -> list:
        """Geometrie (lon/lat) dla listy (klucz, wartości); zamknięte linie → poligony."""
        out = []
        for key, values in specs:
            if self.region is not None:
                geoms = self.region.layer_geoms(key, values, bbox)
            else:
                q = layer_query(key, values, bbox, geometry=True)
                geoms = overpass_to_geometries(self.client.query(q, notify=lambda m: report(0.01, m)), key)
            for g in geoms:
                if g.geom_type == "LineString" and g.is_closed and len(g.coords) >= 4 \
                        and key not in ("railway", "highway", "barrier"):  # bariery liniowe zostają liniami
                    g = shapely.Polygon(g.coords)  # np. waterway=riverbank zapisane jako zamknięta linia
                if g.geom_type in ("Polygon", "MultiPolygon"):
                    out.append(g if g.is_valid else shapely.make_valid(g))
                elif not polygons_only and g.geom_type in ("LineString", "MultiLineString"):
                    out.append(g)
        return out

    def _apply_exclusions(self, report) -> None:
        """Usuwa z oceny punkty leżące na wykluczonych terenach (np. na wodzie)."""
        if not self.exclude:
            return
        report(0.01, "Wyznaczanie terenów wyłączonych z oceny…")
        proj = self.grid.proj
        bbox = geo.buffered_bbox_ll(self.area_ll, 50)
        polys = []
        for ex in self.exclude:
            polys += self._layer_polys_lines(EXCLUSIONS[ex][1], bbox, report, polygons_only=True)
        if not polys:
            return
        ex_ll = shapely.union_all(polys).intersection(self.area_ll)
        if ex_ll.is_empty:
            return
        ex_xy = proj.geom_to_xy(ex_ll)
        shapely.prepare(ex_xy)
        jj, ii = np.nonzero(self.grid.mask)
        x, y = self.grid.node_xy()
        inside = shapely.contains_xy(ex_xy, x, y)
        if inside.all():
            raise AnalysisError("Wszystkie punkty analizy leżą na terenach wyłączonych z oceny.")
        self.grid.mask[jj[inside], ii[inside]] = False
        self.n_excluded = int(inside.sum())
        simp = shapely.simplify(ex_ll, 0.00005, preserve_topology=True)
        self.excluded_geojson = shapely.geometry.mapping(simp)

    def run(self, progress: Callable[[float, str], None] | None = None) -> None:
        t0 = time.time()
        report = progress or (lambda f, m: None)
        self._apply_exclusions(report)
        proj = self.grid.proj
        pts_xy = np.column_stack(self.grid.node_xy())
        n_steps = len(self.criteria) + (1 if any(c.metric == "walking" for c in self.criteria) else 0)
        step = 0
        graph = None
        if any(c.metric == "walking" for c in self.criteria):
            report(0.02, "Sieć dróg i ścieżek pieszych…")
            buf = max(c.max_distance for c in self.criteria if c.metric == "walking") + SNAP_MAX
            wbbox = geo.buffered_bbox_ll(self.area_ll, buf)
            if self.region is not None:
                graph = build_graph_arrays(*self.region.walk_arrays(wbbox), proj)
            else:
                data = self.client.query(walking_network_query(wbbox), notify=lambda m: report(0.02, m))
                if ts := osm_timestamp(data):
                    self.osm_timestamps.add(ts)
                graph = build_graph(data, proj)
            report(0.03, "Bariery dla dojść do sieci (wody, tory, autostrady)…")
            graph.set_barriers(_project(self._layer_polys_lines(WALK_BARRIERS, wbbox, report), proj),
                               _project(self._layer_polys_lines(WALK_SOFT_BARRIERS, wbbox, report), proj))
            step += 1

        columns = []
        for c in self.criteria:
            frac = 0.05 + 0.9 * step / n_steps
            report(frac, f"Kryterium „{c.name}”: pobieranie danych…")
            geoms_ll = self._geoms_for(c, notify=lambda m, f=frac: report(f, m))
            geoms_xy = _project(geoms_ll, proj)
            report(0.05 + 0.9 * (step + 0.6) / n_steps, f"Kryterium „{c.name}”: obliczenia…")
            extra = {}
            if c.method == "count":
                vals = count_within(geoms_xy, pts_xy, c.radius)
            elif c.metric == "walking":
                vals, extra = walking_distance(graph, geoms_xy, pts_xy, c.max_distance)
                eu = euclidean_distance(geoms_xy, pts_xy, c.max_distance)
                # punkt wewnątrz obiektu docelowego (np. parku) jest już u celu
                vals = np.where(eu <= 0, 0.0, np.maximum(vals, eu))
            else:
                vals = euclidean_distance(geoms_xy, pts_xy, c.max_distance)
            columns.append(vals)
            st = {
                "n_features": len(geoms_ll),
                "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()),
                "constant": bool(np.ptp(vals) <= 1e-9 * max(abs(vals).max(), 1.0)),
            }
            uniq, cnt = np.unique(vals, return_counts=True)
            st["mode_value"] = float(uniq[cnt.argmax()])
            st["mode_share"] = float(cnt.max() / len(vals))
            st.update(extra)
            if c.method == "distance":
                at_limit = vals >= c.max_distance - 1e-6   # liczone z wartości końcowych
                st["capped_share"] = float(at_limit.mean())
                st["at_limit"] = int(at_limit.sum())
            st.pop("beyond_limit_or_no_route", None)
            self.crit_stats.append(st)
            self._warn_for(c, st)
            if geoms_xy:
                rep = representative_points_xy(geoms_xy)
                lon, lat = proj.inverse(rep[:, 0], rep[:, 1])
                ll = np.column_stack([lon, lat])
                if len(ll) > PREVIEW_LIMIT:
                    ll = ll[np.random.default_rng(0).choice(len(ll), PREVIEW_LIMIT, replace=False)]
                self.previews[c.id] = np.round(ll, 6).tolist()
            step += 1
        self.matrix = np.column_stack(columns)
        self.duration_s = time.time() - t0
        report(1.0, "Gotowe")

    def _warn_for(self, c: Criterion, st: dict) -> None:
        if st["n_features"] == 0:
            self.warnings.append(f"„{c.name}”: brak obiektów w obszarze i jego otoczeniu – kryterium nie różnicuje lokalizacji.")
        elif st["constant"]:
            self.warnings.append(f"„{c.name}”: wartość jest taka sama we wszystkich punktach – kryterium zostanie pominięte w ocenie.")
        elif c.method == "count" and st.get("mode_share", 0) >= 0.8:
            self.warnings.append(
                f"„{c.name}”: {st['mode_share']:.0%} punktów ma tę samą wartość ({st['mode_value']:g} szt. w promieniu "
                f"{c.radius:g} m), więc kryterium różnicuje tylko niewielką część obszaru, a pozostałym punktom daje "
                "jednakową stratę. Dla rzadkich obiektów lepiej sprawdza się odległość albo większy promień."
            )
        elif c.method == "distance" and c.metric == "walking" and st.get("at_limit", 0):
            self.warnings.append(
                f"„{c.name}”: {st['at_limit']} punktów ({st['capped_share']:.0%}) osiągnęło limit {c.max_distance:g} m "
                "albo nie ma trasy pieszej do obiektu – przyjęto wartość graniczną; punkty na limicie są równorzędne."
            )
        elif c.method == "distance" and st.get("capped_share", 0) > 0.5:
            self.warnings.append(
                f"„{c.name}”: {st['at_limit']} punktów ({st['capped_share']:.0%}) ma najbliższy obiekt dalej niż "
                f"{c.max_distance:g} m – przyjęto wartość graniczną; punkty na limicie są równorzędne."
            )
        if st.get("no_access", 0):
            self.warnings.append(
                f"„{c.name}”: {st['no_access']} punktów nie ma dojścia do sieci pieszej w promieniu 500 m bez "
                "przekraczania wody, torów lub autostrady (np. wyspy, tereny zamknięte) – mają wartość graniczną "
                f"{c.max_distance:g} m. Rozważ wyłączenie tych terenów z oceny."
            )
        if st.get("far_from_network", 0) > 0.02 * self.grid.n_points:
            self.warnings.append(
                f"„{c.name}”: {st['far_from_network']} punktów leży dalej niż 200 m od sieci pieszej "
                "(np. woda, lasy, tereny zamknięte) – odległość piesza jest tam przybliżona."
            )

    # ------------------------------------------------------------------
    def result_json(self) -> dict:
        crits = []
        for c, st in zip(self.criteria, self.crit_stats):
            d = c.to_json()
            d["stats"] = st
            crits.append(d)
        decimals = [1 if c.method == "distance" else 0 for c in self.criteria]
        values = [np.round(self.matrix[:, j], decimals[j]).tolist() for j in range(self.matrix.shape[1])]
        return {
            "name": self.name,
            "goal": self.goal,
            "area": geo.area_to_geojson(self.area_ll),
            "area_km2": round(geo.area_km2(self.area_ll), 3),
            "grid": self.grid.to_json(),
            "excluded": {
                "kinds": [{"id": e, "name": EXCLUSIONS[e][0]} for e in self.exclude],
                "n_points": self.n_excluded,
                "geometry": self.excluded_geojson,
            },
            "criteria": crits,
            "values": values,  # kolumnami: values[j][i] = kryterium j w punkcie i
            "previews": self.previews,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s or 0, 1),
            "osm_timestamp": max(self.osm_timestamps) if self.osm_timestamps else None,
            "data_source": self.data_source,
        }

    def evaluate(self, method_id: str, weights_spec: dict, directions: dict | None = None) -> dict:
        """Ocena MCDA na policzonej macierzy. `directions` pozwala zmienić kierunek
        preferencji kryterium (benefit/cost) bez ponownego pobierania danych."""
        if self.matrix is None:
            raise AnalysisError("Analiza nie została jeszcze policzona.")
        n = len(self.criteria)
        directions = directions or {}
        dirs = []
        for c in self.criteria:
            d = directions.get(c.id, c.direction)
            if d not in ("benefit", "cost"):
                raise AnalysisError(f"Nieznany kierunek preferencji: {d!r}.")
            dirs.append(d)
        types = np.array([1 if d == "benefit" else -1 for d in dirs])
        mode = (weights_spec or {}).get("mode", "equal")
        if mode in OBJECTIVE:
            # wagi obiektywne tylko z kryteriów, które różnicują obszar – stała kolumna
            # nie może zmieniać wag pozostałych (np. przez korelacje w CRITIC)
            var = ~constant_columns(self.matrix)
            w_all = np.zeros(n)
            if var.any():
                w_all[var] = compute_weights(weights_spec, self.matrix[:, var], int(var.sum()), types[var])
        else:
            w_all = compute_weights(weights_spec, self.matrix, n)
        ev = evaluate(self.matrix, w_all, types, method_id)
        used_ids = [self.criteria[j].id for j in ev.used]
        excluded = [c.id for j, c in enumerate(self.criteria) if j not in set(ev.used.tolist())]
        weights_used = {cid: 0.0 for cid in (c.id for c in self.criteria)}
        for cid, w in zip(used_ids, ev.weights):
            weights_used[cid] = float(w)
        crit_desc = []
        for c, d in zip(self.criteria, dirs):
            cj = c.to_json()
            cj["direction"] = d
            cj["description"] = Criterion(**{**{k: getattr(c, k) for k in (
                "id", "name", "method", "source", "metric", "radius", "max_distance")}, "direction": d}).describe()
            crit_desc.append(cj)
        out = {
            "eval_id": uuid.uuid4().hex[:12],
            "criteria": crit_desc,
            "directions": {c.id: d for c, d in zip(self.criteria, dirs)},
            "method": {
                "id": ev.method.id, "name": ev.method.name, "lower_is_better": ev.method.lower_is_better,
                "score_label": ev.method.score_label, "params": ev.method.params,
            },
            "weights_mode": (weights_spec or {}).get("mode", "equal"),
            "weights_input": {c.id: float(w) for c, w in zip(self.criteria, w_all)},
            "weights": weights_used,
            "used": used_ids,
            "excluded": excluded,
            "scores": ev.scores.tolist(),  # pełna precyzja – zaokrąglenie tworzyłoby sztuczne remisy
            "ranks": ev.ranks.tolist(),
            "contributions": None if ev.contributions is None else ev.contributions.tolist(),
            "summary": {
                "min": float(ev.scores.min()), "max": float(ev.scores.max()),
                "mean": float(ev.scores.mean()), "n_points": int(ev.scores.size),
            },
        }
        return out

    # ------------------------------------------------------------------
    def export_rows(self, evaluation: dict | None):
        """Wiersze do eksportu: (lon, lat, {kolumny})."""
        lon, lat = self.grid.node_lonlat()
        for i in range(len(lon)):
            props = {"id": i}
            for j, c in enumerate(self.criteria):
                props[c.id] = float(self.matrix[i, j])
            if evaluation:
                props["score"] = evaluation["scores"][i]
                props["rank"] = evaluation["ranks"][i]
            yield float(lon[i]), float(lat[i]), props


def describe_options() -> dict:
    return {
        "methods": [
            {"id": m.id, "name": m.name, "lower_is_better": m.lower_is_better,
             "score_label": m.score_label, "description": m.description}
            for m in METHODS.values()
        ],
        "weight_modes": [{"id": k, "name": v, "objective": k in OBJECTIVE} for k, v in MODES.items()],
        "exclusions": [{"id": k, "name": v[0]} for k, v in EXCLUSIONS.items()],
        "limits": {
            "max_points": geo.MAX_POINTS, "max_criteria": MAX_CRITERIA,
            "walking_max_area_km2": WALKING_MAX_AREA_KM2,
        },
        "defaults": {"max_distance": DEFAULT_MAX_DISTANCE, "radius": 500.0},
    }


__all__ = ["Analysis", "AnalysisError", "CriterionError", "grid_preview", "describe_options"]
