"""Odległość piesza po sieci dróg i ścieżek OSM.

Graf pieszy (węzły OSM, krawędzie = odcinki dróg w metrach) i wieloźródłowa Dijkstra
od obiektów docelowych: jedno drzewo najkrótszych ścieżek na kryterium. Do grafu
dodajemy wirtualne źródło połączone z końcami odcinków, do których dociągnięto obiekty.

Dociąganie (punkt siatki → sieć, obiekt → sieć) odbywa się do najbliższego ODCINKA
(rzut prostopadły), nie do węzła, a odcinek dociągnięcia nie może przecinać barier
(wody, tory kolejowe, autostrady). Gdy punkt i obiekt leżą przy tym samym odcinku,
odległość liczona jest bezpośrednio wzdłuż niego.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .features import flatten_lines

SNAP_MAX = 300.0       # m – maks. dojście od obiektu docelowego do sieci pieszej
SAMPLE_STEP = 20.0     # m – próbkowanie linii i obwodów obiektów docelowych
CANDIDATE_EXTRA = 10.0  # m – kandydaci do dociągnięcia: odcinki najwyżej o tyle dalsze od najbliższego
FALLBACK_RADIUS = 500.0  # m – szukanie odcinka bez bariery, gdy najbliższe są za barierą
SOFT_DETOUR = 100.0      # m – o ile dalszy odcinek bez barier wolimy od przejścia przez ogrodzenie
MIN_COMPONENT = 50       # węzłów – mniejsze, odcięte fragmenty sieci nie służą do dociągania


class NetworkError(RuntimeError):
    pass


@dataclass
class WalkGraph:
    xy: np.ndarray          # (N, 2) współrzędne węzłów w metrach
    rows: np.ndarray        # krawędzie (nieskierowane, bez duplikatów)
    cols: np.ndarray
    lengths: np.ndarray
    edges: np.ndarray       # geometrie odcinków (LineString) w kolejności rows/cols
    tree: shapely.STRtree   # indeks przestrzenny odcinków
    big_edge: np.ndarray | None = None   # odcinek należy do składowej ≥ MIN_COMPONENT węzłów
    big_tree: shapely.STRtree | None = None  # indeks tylko po tych odcinkach (najbliższy odcinek sieci)
    barriers: shapely.STRtree | None = None
    barrier_geoms: list = field(default_factory=list)
    barrier_array: np.ndarray | None = None
    soft_array: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.xy)

    def set_barriers(self, hard_xy, soft_xy=()) -> None:
        """Bariery dla odcinków dociągnięcia.

        Twarde (wody, tory, autostrady) nigdy nie są przekraczane. Miękkie (ogrodzenia, mury)
        są omijane, gdy w pobliżu jest odcinek sieci bez nich, a przekraczane tylko
        w ostateczności – OSM rzadko zaznacza wszystkie bramy i furtki.
        """
        def split(geoms):
            parts = []
            for g in geoms:
                if not g.is_empty:
                    parts.extend(getattr(g, "geoms", [g]))
            parts = [p for p in parts if p.geom_type in ("Polygon", "LineString", "LinearRing")]
            for p in parts:
                shapely.prepare(p)
            return parts
        hard, soft = split(hard_xy), split(soft_xy)
        self.barrier_geoms = hard + soft
        self.barrier_array = np.asarray(hard, dtype=object) if hard else None
        self.soft_array = np.asarray(soft, dtype=object) if soft else None
        self.barriers = shapely.STRtree(self.barrier_geoms) if self.barrier_geoms else None


def build_graph(data: dict, proj) -> WalkGraph:
    nodes = {}
    ways = []
    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "way" and el.get("nodes"):
            ways.append(el["nodes"])
    if not nodes or not ways:
        raise NetworkError("Brak sieci dróg i ścieżek pieszych w obszarze – nie można policzyć odległości pieszej.")
    ids = np.fromiter(nodes.keys(), dtype=np.int64, count=len(nodes))
    ll = np.array(list(nodes.values()), dtype=float)
    x, y = proj.forward(ll[:, 0], ll[:, 1])
    xy = np.column_stack([x, y])
    order = np.argsort(ids)
    ids_sorted = ids[order]

    a_list, b_list = [], []
    for refs in ways:
        r = np.asarray(refs, dtype=np.int64)
        pos = np.searchsorted(ids_sorted, r)
        pos = np.clip(pos, 0, len(ids_sorted) - 1)
        ok = ids_sorted[pos] == r
        idx = np.where(ok, order[pos], -1)
        a, b = idx[:-1], idx[1:]
        keep = (a >= 0) & (b >= 0) & (a != b)
        a_list.append(a[keep])
        b_list.append(b[keep])
    rows = np.concatenate(a_list) if a_list else np.empty(0, dtype=np.int64)
    cols = np.concatenate(b_list) if b_list else np.empty(0, dtype=np.int64)
    return _graph_from_edges(xy, rows, cols)


def build_graph_arrays(lon, lat, u, v, proj) -> WalkGraph:
    """Graf z tablic (węzły lon/lat, krawędzie u–v) – używane przez lokalny region."""
    if len(u) == 0:
        raise NetworkError("Brak sieci dróg i ścieżek pieszych w obszarze – nie można policzyć odległości pieszej.")
    used = np.unique(np.concatenate([u, v]))
    remap = np.full(len(lon), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    x, y = proj.forward(np.asarray(lon)[used], np.asarray(lat)[used])
    rows, cols = remap[u], remap[v]
    keep = rows != cols
    return _graph_from_edges(np.column_stack([x, y]), rows[keep], cols[keep])


def _graph_from_edges(xy, rows, cols) -> WalkGraph:
    if rows.size == 0:
        raise NetworkError("Sieć piesza w obszarze nie zawiera odcinków.")
    a, b = np.minimum(rows, cols), np.maximum(rows, cols)
    keep = a != b
    pairs = np.unique(np.column_stack([a[keep], b[keep]]), axis=0)  # bez zdublowanych odcinków
    rows, cols = pairs[:, 0], pairs[:, 1]
    lengths = np.maximum(np.hypot(*(xy[rows] - xy[cols]).T), 0.01)  # 0 w csgraph = brak krawędzi
    edges = shapely.linestrings(np.stack([xy[rows], xy[cols]], axis=1))
    g = coo_matrix((lengths, (rows, cols)), shape=(len(xy), len(xy)))
    _, labels = connected_components(g, directed=False)
    sizes = np.bincount(labels)
    big_edge = sizes[labels[rows]] >= min(MIN_COMPONENT, sizes.max())
    return WalkGraph(xy=xy, rows=rows, cols=cols, lengths=lengths, edges=edges, tree=shapely.STRtree(edges),
                     big_edge=big_edge, big_tree=shapely.STRtree(edges[big_edge]))


def _target_samples(geoms_xy) -> np.ndarray:
    pts = []
    for g in geoms_xy:
        if g.geom_type == "Point":
            pts.append((g.x, g.y))
        elif g.geom_type == "MultiPoint":
            pts.extend((p.x, p.y) for p in g.geoms)
        else:
            for line in flatten_lines(g):
                dense = shapely.segmentize(line, SAMPLE_STEP)
                pts.extend(shapely.get_coordinates(dense).tolist())
            if g.geom_type == "GeometryCollection":
                pts.extend((p.x, p.y) for p in g.geoms if p.geom_type == "Point")
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _candidates(graph: WalkGraph, pts, pts_xy, idx, radius):
    """Pary (punkt, odcinek) w promieniu `radius` z odległością, pozycją i oceną barier."""
    qi, ce = graph.tree.query(pts[idx], predicate="dwithin", distance=radius)
    cp = idx[qi]
    edges = graph.edges[ce]
    t = shapely.line_locate_point(edges, pts[cp])
    proj = shapely.line_interpolate_point(edges, t)
    d = shapely.distance(pts[cp], proj)
    hard_ok = np.ones(len(cp), dtype=bool)
    soft_ok = np.ones(len(cp), dtype=bool)
    if len(cp) and (graph.barrier_array is not None or graph.soft_array is not None):
        seg = shapely.linestrings(np.stack([pts_xy[cp], shapely.get_coordinates(proj)], axis=1))
        # indeks po odcinkach dociągnięcia, a bariery jako zapytanie: GEOS przygotowuje geometrie
        # zapytania, więc duże poligony wód są testowane szybko (odwrotny kierunek był ~30× wolniejszy)
        stree = shapely.STRtree(seg)
        if graph.barrier_array is not None:
            hard_ok[np.unique(stree.query(graph.barrier_array, predicate="intersects")[1])] = False
        if graph.soft_array is not None:
            soft_ok[np.unique(stree.query(graph.soft_array, predicate="intersects")[1])] = False
    big = graph.big_edge[ce] if graph.big_edge is not None else np.ones(len(cp), dtype=bool)
    return cp, ce, d, t, hard_ok & big, soft_ok


def _snap(graph: WalkGraph, pts_xy: np.ndarray, max_snap: float | None):
    """Dociągnięcia do sieci: (indeks punktu, indeks odcinka, odl., pozycja na odcinku, nearest).

    Kolejne poziomy (pierwszy, który da kandydata, wygrywa; odległości względem najbliższego odcinka):
    1. bez żadnej bariery, do +CANDIDATE_EXTRA,
    2. bez żadnej bariery, do +SOFT_DETOUR,
    3. tylko przez barierę miękką (ogrodzenie, mur), do +CANDIDATE_EXTRA,
    4. bez żadnej bariery, do FALLBACK_RADIUS (lub `max_snap`),
    5. tylko przez barierę miękką, do FALLBACK_RADIUS (lub `max_snap`),
    zawsze w składowej sieci ≥ MIN_COMPONENT węzłów i nigdy przez wodę, tory ani autostradę.
    Gdy brak kandydata, punkt nie dostaje dociągnięcia (brak trasy).
    """
    pts = shapely.points(pts_xy)
    tree = graph.big_tree if graph.big_tree is not None else graph.tree
    (pi, _), d0 = tree.query_nearest(pts, return_distance=True, all_matches=False, max_distance=max_snap)
    nearest = np.full(len(pts_xy), np.inf)
    nearest[pi] = d0
    has = np.nonzero(np.isfinite(nearest))[0]
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
    if has.size == 0:
        return (*empty, nearest)
    radius = nearest[has] + CANDIDATE_EXTRA
    if max_snap is not None:
        radius = np.minimum(radius, max_snap)
    cp, ce, d, t, hard_ok, soft_ok = _candidates(graph, pts, pts_xy, has, radius)
    ok = hard_ok & soft_ok
    parts = [(cp[ok], ce[ok], d[ok], t[ok])]
    covered = np.zeros(len(pts_xy), dtype=bool)
    covered[cp[ok]] = True
    missing = has[~covered[has]]
    if missing.size:
        far = FALLBACK_RADIUS if max_snap is None else max_snap
        fcp, fce, fd, ft, fhard, fsoft = _candidates(graph, pts, pts_xy, missing, far)
        near_lim = nearest[fcp]
        tiers = (
            fhard & fsoft & (fd <= near_lim + SOFT_DETOUR),   # 2: bez barier, niewielki objazd
            fhard & (fd <= near_lim + CANDIDATE_EXTRA),       # 3: tylko przez ogrodzenie, tuż obok
            fhard & fsoft,                                    # 4: bez barier, dalej
            fhard,                                            # 5: przez ogrodzenie, dalej
        )
        for mask in tiers:
            m = mask & ~covered[fcp]
            if not m.any():
                continue
            best = np.full(len(pts_xy), np.inf)
            np.minimum.at(best, fcp[m], fd[m])
            keep = m & (fd <= best[fcp] + CANDIDATE_EXTRA)
            parts.append((fcp[keep], fce[keep], fd[keep], ft[keep]))
            covered[fcp[keep]] = True
    cp, ce, d, t = (np.concatenate(x) for x in zip(*parts))
    return cp, ce, d, t, nearest


def walking_distance(graph: WalkGraph, geoms_xy, points_xy: np.ndarray, max_distance: float):
    """Odległość piesza (m) z każdego punktu do najbliższego obiektu, obcięta do max_distance.

    Zwraca (wartości, statystyki).
    """
    n = graph.n_nodes
    L = graph.lengths
    stats: dict = {}
    src = {}                       # węzeł -> koszt dojścia od obiektu
    same_edge: dict[int, list] = {}  # odcinek -> [(pozycja t, koszt dociągnięcia obiektu)]

    samples = _target_samples(geoms_xy)
    if samples.size:
        sp, se, sd, st, _ = _snap(graph, samples, SNAP_MAX)
        # każdy poprawny odcinek w zasięgu (najbliższy + CANDIDATE_EXTRA) jest możliwym dojściem
        # do obiektu – np. chodnik przy ogrodzeniu szkoły, a nie tylko ścieżka wewnątrz jej terenu
        for k in range(len(sp)):
            e, dd, tt = int(se[k]), float(sd[k]), float(st[k])
            u, v = int(graph.rows[e]), int(graph.cols[e])
            for node, w in ((u, dd + tt), (v, dd + L[e] - tt)):
                if w < src.get(node, np.inf):
                    src[node] = w
            same_edge.setdefault(e, []).append((tt, dd))
    # węzły sieci wewnątrz poligonów docelowych (np. park) mają odległość 0
    polys = [g for g in geoms_xy if g.geom_type in ("Polygon", "MultiPolygon")]
    if polys:
        ptree = shapely.STRtree(polys)
        inside = ptree.query(shapely.points(graph.xy), predicate="within")[0]
        for node in np.unique(inside):
            src[int(node)] = 0.0

    stats["n_sources"] = len(src)
    if not src:
        stats["beyond_limit_or_no_route"] = int(len(points_xy))
        stats["far_from_network"] = 0
        return np.full(len(points_xy), float(max_distance)), stats

    s = n
    src_nodes = np.fromiter(src.keys(), dtype=np.int64)
    src_w = np.maximum(np.fromiter(src.values(), dtype=float), 0.01)
    rows = np.concatenate([graph.rows, np.full(src_nodes.size, s)])
    cols = np.concatenate([graph.cols, src_nodes])
    w = np.concatenate([L, src_w])
    g = coo_matrix((w, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    dist = dijkstra(g, directed=False, indices=s, limit=max_distance + SNAP_MAX)[:n]

    cp, ce, cd, ct, nearest = _snap(graph, points_xy, None)
    u, v = graph.rows[ce], graph.cols[ce]
    cost = cd + np.minimum(ct + dist[u], L[ce] - ct + dist[v])
    if same_edge:
        for k in np.nonzero(np.isin(ce, np.fromiter(same_edge.keys(), dtype=np.int64)))[0]:
            for tt, dd in same_edge[int(ce[k])]:
                cost[k] = min(cost[k], cd[k] + abs(ct[k] - tt) + dd)
    values = np.full(len(points_xy), np.inf)
    np.minimum.at(values, cp, cost)
    unreachable = ~np.isfinite(values)
    values = np.where(unreachable, max_distance, values)
    stats["far_from_network"] = int((nearest > 200).sum())
    has_access = np.zeros(len(points_xy), dtype=bool)
    has_access[cp] = True
    stats["no_access"] = int((~has_access).sum())  # brak dojścia do sieci bez przekraczania wody/torów
    stats["beyond_limit_or_no_route"] = int((values >= max_distance - 1e-9).sum())
    return np.minimum(values, max_distance), stats
