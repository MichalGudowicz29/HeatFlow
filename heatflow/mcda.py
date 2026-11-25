"""Ocena wielokryterialna punktów siatki.

Rejestr metod pozwala dodać kolejny algorytm bez zmian w reszcie systemu:
wystarczy funkcja (macierz, wagi, typy) -> (wyniki, wkłady | None) i wpis w METHODS.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pymcdm.methods import SPOTIS, TOPSIS, VIKOR
from scipy.stats import rankdata


class MCDAError(ValueError):
    pass


@dataclass(frozen=True)
class Method:
    id: str
    name: str
    lower_is_better: bool
    score_label: str
    description: str
    fn: Callable
    params: dict | None = None


def _spotis(matrix, weights, types):
    bounds = np.column_stack([matrix.min(axis=0), matrix.max(axis=0)])
    ideal = np.where(types > 0, bounds[:, 1], bounds[:, 0])
    span = bounds[:, 1] - bounds[:, 0]
    contrib = weights * np.abs(matrix - ideal) / span
    scores = SPOTIS(bounds)(matrix, weights, types)
    return np.asarray(scores, dtype=float), contrib


def _topsis(matrix, weights, types):
    return np.asarray(TOPSIS()(matrix, weights, types), dtype=float), None


def _vikor(matrix, weights, types):
    q = np.asarray(VIKOR()(matrix, weights, types), dtype=float)
    if not np.all(np.isfinite(q)):
        raise MCDAError("VIKOR nie daje rozstrzygnięcia dla tych danych (zerowy rozrzut). Wybierz SPOTIS lub TOPSIS.")
    return q, None


METHODS = {
    "spotis": Method(
        "spotis", "SPOTIS", True, "Odległość od rozwiązania idealnego",
        "Ważona, znormalizowana odległość od punktu idealnego (0 = ideał). Granice kryteriów = "
        "zakres wartości w analizowanym obszarze, więc ocena jest względna dla tego obszaru "
        "(inny obszar = inny punkt odniesienia). Wynik rozkłada się na straty poszczególnych kryteriów.",
        _spotis, {"bounds": "min–max wartości w obszarze"},
    ),
    "topsis": Method(
        "topsis", "TOPSIS", False, "Współczynnik bliskości do ideału",
        "Względna bliskość do rozwiązania idealnego i oddalenie od antyidealnego (1 = najlepiej). "
        "Normalizacja min–max.",
        _topsis, {"normalization": "minmax"},
    ),
    "vikor": Method(
        "vikor", "VIKOR", True, "Indeks kompromisu Q",
        "Rozwiązanie kompromisowe łączące użyteczność grupy i największy żal (0 = najlepiej, v = 0,5).",
        _vikor, {"v": 0.5},
    ),
}


@dataclass
class Evaluation:
    method: Method
    scores: np.ndarray          # (n_points,)
    ranks: np.ndarray           # 1 = najlepszy
    contributions: np.ndarray | None  # (n_points, n_used) – tylko SPOTIS
    used: np.ndarray            # indeksy kryteriów użytych w ocenie
    weights: np.ndarray         # wagi kryteriów użytych (suma 1)


def constant_columns(matrix: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    span = matrix.max(axis=0) - matrix.min(axis=0)
    scale = np.maximum(np.abs(matrix).max(axis=0), 1.0)
    return span <= rtol * scale


def evaluate(matrix: np.ndarray, weights: np.ndarray, types: np.ndarray, method_id: str) -> Evaluation:
    if method_id not in METHODS:
        raise MCDAError(f"Nieznana metoda MCDA: {method_id!r}.")
    method = METHODS[method_id]
    matrix = np.asarray(matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)
    types = np.asarray(types, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise MCDAError("Do oceny potrzebne są co najmniej 2 punkty.")
    const = constant_columns(matrix)
    used = np.nonzero(~const)[0]
    if used.size == 0:
        raise MCDAError(
            "Żadne kryterium nie różnicuje lokalizacji w tym obszarze (wszystkie mają stałą wartość)."
        )
    w = weights[used]
    if w.sum() <= 0:
        raise MCDAError("Kryteria, które różnicują obszar, mają zerowe wagi.")
    w = w / w.sum()
    m, t = matrix[:, used], types[used]
    with warnings.catch_warnings():
        # Na siatce przestrzennej punkty zdominowane są czymś normalnym – ostrzeżenia pymcdm pomijamy.
        warnings.simplefilter("ignore", UserWarning)
        scores, contrib = method.fn(m, w, t)
    ranks = rankdata(scores if method.lower_is_better else -scores, method="min").astype(int)
    return Evaluation(method=method, scores=scores, ranks=ranks, contributions=contrib, used=used, weights=w)
