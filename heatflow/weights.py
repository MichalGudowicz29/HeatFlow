"""Wagi kryteriów: ręczne, RANCOM (z rankingu) i obiektywne (z macierzy decyzyjnej).

Wagi obiektywne liczone są na macierzy znormalizowanej min–max ze stopniami spełnienia
kryteriów (kryteria kosztowe odwrócone: 1 = najlepsza wartość w obszarze), bo kryteria mają
różne jednostki (metry, liczby obiektów), a CRITIC mierzy konflikt między stopniami spełnienia
(Diakoulaki i in., 1995). Entropia używa konwencji 0·ln 0 = 0
(implementacja w pymcdm przypisuje kolumnie z jakimkolwiek zerem maksymalną wagę,
co dla danych o liczności obiektów daje błędne wyniki).
"""
from __future__ import annotations

import numpy as np
from pymcdm.weights.subjective import RANCOM

MODES = {
    "manual": "Ręczne",
    "rancom": "RANCOM (ranking ekspercki)",
    "entropy": "Entropia (obiektywne)",
    "critic": "CRITIC (obiektywne)",
    "std": "Odchylenie standardowe (obiektywne)",
    "equal": "Równe",
}
OBJECTIVE = {"entropy", "critic", "std"}


class WeightsError(ValueError):
    pass


def _minmax(matrix: np.ndarray, types=None) -> np.ndarray:
    """Stopień spełnienia kryterium w [0, 1]; dla kryteriów kosztowych (typ −1) odwrócony."""
    mn = matrix.min(axis=0)
    rng = matrix.max(axis=0) - mn
    rng[rng == 0] = 1.0
    norm = (matrix - mn) / rng
    if types is not None:
        cost = np.asarray(types) < 0
        norm[:, cost] = 1.0 - norm[:, cost]
    return norm


def entropy_weights(matrix: np.ndarray, types=None) -> np.ndarray:
    m = matrix.shape[0]
    if m < 2:
        raise WeightsError("Za mało punktów do policzenia entropii.")
    norm = _minmax(matrix, types)
    col_sum = norm.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    p = norm / col_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > 0, p * np.log(p), 0.0)
    e = -plogp.sum(axis=0) / np.log(m)
    d = 1.0 - e
    if d.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return d / d.sum()


def critic_weights(matrix: np.ndarray, types=None) -> np.ndarray:
    norm = _minmax(matrix, types)
    std = norm.std(axis=0, ddof=1)
    if norm.shape[1] == 1:
        return np.ones(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(norm, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    c = std * (1.0 - corr).sum(axis=0)
    if c.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return c / c.sum()


def std_weights(matrix: np.ndarray, types=None) -> np.ndarray:
    std = _minmax(matrix, types).std(axis=0, ddof=1)  # odwrócenie nie zmienia odchylenia
    if std.sum() <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return std / std.sum()


def rancom_weights(ranking) -> np.ndarray:
    r = np.asarray(ranking, dtype=float)
    if r.ndim != 1 or r.size == 0 or np.any(~np.isfinite(r)) or np.any(r < 1):
        raise WeightsError("Ranking RANCOM musi zawierać pozycje 1, 2, 3… (remisy dozwolone).")
    if r.size == 1:
        return np.ones(1)
    return np.asarray(RANCOM(ranking=r)(), dtype=float)


def rancom_from_spec(spec: dict, n: int):
    """Obiekt pymcdm RANCOM z odpowiedzi eksperta.

    - `matrix`: macierz MAC z porównań parami (pytania jak w `RANCOM(object_names=...)`),
      wartości 0 / 0,5 / 1, przekątna 0,5, a[j, i] = 1 − a[i, j] – walidowana przez pymcdm,
    - `ranking`: pozycje kryteriów (1 = najważniejsze, remisy dozwolone),
    - `scoring`: punktacja kryteriów (więcej = ważniejsze).
    """
    if n == 1:
        return None
    if spec.get("matrix") is not None:
        m = np.asarray(spec["matrix"], dtype=float)
        if m.shape != (n, n):
            raise WeightsError(f"Macierz porównań RANCOM ma wymiar {m.shape}, oczekiwano {n}×{n}. "
                               "Odpowiedz na pytania dla wszystkich par kryteriów.")
        # walidacja ściślejsza niż w pymcdm (która zaokrągla wartości i nie sprawdza przekątnej)
        if not np.all(np.isin(m, (0.0, 0.5, 1.0))):
            raise WeightsError("Macierz porównań RANCOM może zawierać tylko 0, ½ i 1.")
        if not np.allclose(np.diag(m), 0.5):
            raise WeightsError("Na przekątnej macierzy porównań RANCOM musi być ½.")
        if not np.allclose(m + m.T, 1.0):
            raise WeightsError("Macierz porównań RANCOM musi być wzajemna: a[j, i] = 1 − a[i, j].")
        try:
            return RANCOM(matrix=m)
        except ValueError as exc:
            raise WeightsError(f"Niepoprawna macierz porównań RANCOM: {exc}") from None
    if spec.get("scoring") is not None:
        sc = np.asarray(spec["scoring"], dtype=float)
        if sc.size != n or not np.all(np.isfinite(sc)):
            raise WeightsError(f"Punktacja RANCOM ma {sc.size} pozycji dla {n} kryteriów.")
        # Punktacja → ranking (więcej punktów = wyższa pozycja, remisy zachowane). Nie używamy
        # RANCOM(scoring=...), bo w pymcdm 1.4.0 przy powtórzonych wartościach przestawia pozycje.
        from scipy.stats import rankdata
        return RANCOM(ranking=rankdata(-sc, method="min"))
    r = spec.get("ranking") or []
    if len(r) != n:
        raise WeightsError(f"Ranking RANCOM ma {len(r)} pozycji dla {n} kryteriów.")
    rancom_weights(r)  # walidacja
    return RANCOM(ranking=np.asarray(r, dtype=float))


def inconsistent_triads(mac: np.ndarray) -> int:
    """Liczba trójek kryteriów z niespójnymi (nieprzechodnimi) odpowiedziami."""
    n = mac.shape[0]
    bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx = [i, j, k]
                sub = mac[np.ix_(idx, idx)]
                s = sub.sum(axis=1)
                expect = np.where(s[:, None] > s[None, :], 1.0, np.where(s[:, None] < s[None, :], 0.0, 0.5))
                if np.any(np.abs(expect - sub) > 1e-9):
                    bad += 1
    return bad


def rancom_details(spec: dict, n: int) -> dict:
    """Wagi RANCOM (pymcdm) wraz z macierzą MAC – do podglądu w interfejsie."""
    if n == 1:
        return {"weights": [1.0], "matrix": [[0.5]], "inconsistent_triads": 0, "n_triads": 0}
    r = rancom_from_spec(spec, n)
    w = np.asarray(r(), dtype=float)
    mac = np.asarray(r.matrix, dtype=float)
    return {"weights": w.tolist(), "matrix": mac.tolist(), "scw": mac.sum(axis=1).tolist(),
            "inconsistent_triads": inconsistent_triads(mac), "n_triads": n * (n - 1) * (n - 2) // 6}


def compute_weights(spec: dict, matrix: np.ndarray, n: int, types=None) -> np.ndarray:
    """Wagi dla `n` kryteriów wg specyfikacji {mode, values|ranking}."""
    mode = (spec or {}).get("mode", "equal")
    if mode not in MODES:
        raise WeightsError(f"Nieznany sposób wyznaczania wag: {mode!r}.")
    if mode == "equal":
        return np.full(n, 1.0 / n)
    if mode == "manual":
        v = np.asarray(spec.get("values") or [], dtype=float)
        if v.size != n:
            raise WeightsError(f"Podano {v.size} wag dla {n} kryteriów.")
        if np.any(v < 0) or not np.all(np.isfinite(v)):
            raise WeightsError("Wagi nie mogą być ujemne.")
        if v.sum() <= 0:
            raise WeightsError("Suma wag musi być dodatnia.")
        return v / v.sum()
    if mode == "rancom":
        r = rancom_from_spec(spec, n)
        return np.ones(1) if r is None else np.asarray(r(), dtype=float)
    if matrix is None or matrix.shape[1] != n:
        raise WeightsError("Wagi obiektywne wymagają policzonej macierzy decyzyjnej.")
    fn = {"entropy": entropy_weights, "critic": critic_weights, "std": std_weights}[mode]
    return fn(np.asarray(matrix, dtype=float), types)
