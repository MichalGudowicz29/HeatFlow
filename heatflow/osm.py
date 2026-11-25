"""Pobieranie danych z OpenStreetMap przez Overpass API.

Jedno zapytanie na warstwę danych dla całego obszaru (z buforem), cache dyskowy
kluczowany treścią zapytania, kilka serwerów lustrzanych i jawne błędy zamiast
cichych wartości zastępczych.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import requests

DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
USER_AGENT = "HeatFlow/2.0 (spatial multi-criteria analysis tool)"
CACHE_TTL_S = 7 * 24 * 3600

# Drogi, po których można iść pieszo (bez autostrad i dróg ekspresowych)
WALKABLE_HIGHWAYS = [
    "footway", "pedestrian", "path", "steps", "living_street", "residential",
    "service", "unclassified", "tertiary", "tertiary_link", "secondary", "secondary_link",
    "primary", "primary_link", "track", "cycleway", "corridor", "road", "bridleway",
]

_SAFE = re.compile(r"^[\w:\- ]+$", re.UNICODE)


class OverpassError(RuntimeError):
    """Nie udało się pobrać danych z Overpass (komunikat dla użytkownika)."""


def _check_token(value: str, what: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE.match(value):
        raise ValueError(f"Niedozwolona {what} tagu OSM: {value!r}")
    return value


def tag_filter(key: str, values: list[str] | None) -> str:
    """Filtr Overpass QL: sam klucz (dowolna wartość), jedna wartość albo lista wartości."""
    key = _check_token(key, "nazwa klucza")
    if not values:
        return f'["{key}"]'
    vals = [_check_token(v, "wartość") for v in values]
    if len(vals) == 1:
        return f'["{key}"="{vals[0]}"]'
    alt = "|".join(vals)  # _SAFE wyklucza znaki specjalne regex
    return f'["{key}"~"^({alt})$"]'


def layer_query(key: str, values: list[str] | None, bbox, geometry: bool) -> str:
    s, w, n, e = bbox
    out = "out geom qt;" if geometry else "out center qt;"
    return (
        f"[out:json][timeout:180][maxsize:1073741824][bbox:{s:.6f},{w:.6f},{n:.6f},{e:.6f}];\n"
        f"nwr{tag_filter(key, values)};\n{out}"
    )


def osm_timestamp(data: dict) -> str | None:
    """Stan bazy OSM, z którego pochodzi odpowiedź Overpass."""
    return (data.get("osm3s") or {}).get("timestamp_osm_base")


def walking_network_query(bbox) -> str:
    s, w, n, e = bbox
    hw = "|".join(WALKABLE_HIGHWAYS)
    return (
        f"[out:json][timeout:300][maxsize:2147483648][bbox:{s:.6f},{w:.6f},{n:.6f},{e:.6f}];\n"
        f'(way["highway"~"^({hw})$"]["foot"!~"^(no|private|use_sidepath)$"]["access"!~"^(no|private)$"]["area"!="yes"];\n'
        f' way["highway"~"^({hw})$"]["foot"~"^(yes|designated|permissive|official)$"]["area"!="yes"];)->.w;\n'
        f"(.w;node(w.w););\nout skel qt;"
    )


class OverpassClient:
    """Klient Overpass z cache, serwerami lustrzanymi i ponawianiem przy przeciążeniu.

    Serwer można wskazać zmienną środowiskową HEATFLOW_OVERPASS_URL (np. własna instancja
    Overpass dla dużych analyz). `budget_s` ogranicza łączny czas oczekiwania na jedno zapytanie.
    """

    RETRY_STATUS = {429, 502, 503, 504}

    def __init__(self, cache_dir: str | os.PathLike | None = None, endpoints=None,
                 read_timeout: float = 330, budget_s: float = 900, session=None, sleep=time.sleep):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        env = os.environ.get("HEATFLOW_OVERPASS_URL")
        self.endpoints = list(endpoints or ([env] if env else DEFAULT_ENDPOINTS))
        self.read_timeout = read_timeout
        self.budget_s = budget_s
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._sleep = sleep
        self._dead: set[str] = set()  # serwery, które nie odpowiadają (pomijane do końca sesji)

    # --- cache -----------------------------------------------------------
    def _cache_path(self, query: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{h}.json.gz"

    def _cache_get(self, query: str):
        p = self._cache_path(query)
        if p and p.exists() and time.time() - p.stat().st_mtime < CACHE_TTL_S:
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                p.unlink(missing_ok=True)
        return None

    def _cache_put(self, query: str, data) -> None:
        p = self._cache_path(query)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{uuid.uuid4().hex}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        tmp.replace(p)

    # --- zapytanie -------------------------------------------------------
    def query(self, query: str, notify=None) -> dict:
        """Wynik zapytania. `notify(msg)` dostaje komunikaty o oczekiwaniu na serwer."""
        cached = self._cache_get(query)
        if cached is not None:
            return cached
        notify = notify or (lambda msg: None)
        errors: list[str] = []
        t0 = time.monotonic()
        attempt = 0
        while True:
            alive = [u for u in self.endpoints if u not in self._dead] or self.endpoints
            for url in alive:
                host = url.split("/")[2]
                remaining = self.budget_s - (time.monotonic() - t0)
                if remaining <= 5:
                    break
                try:
                    r = self.session.post(url, data={"data": query},
                                          timeout=(10, max(5.0, min(self.read_timeout, remaining))))
                except requests.ConnectionError as exc:
                    errors.append(f"{host}: brak połączenia")
                    self._dead.add(url)
                    continue
                except requests.RequestException as exc:
                    errors.append(f"{host}: {exc.__class__.__name__}")
                    continue
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError:
                        errors.append(f"{host}: niepoprawna odpowiedź")
                        continue
                    remark = data.get("remark") or ""
                    if any(k in remark for k in ("runtime error", "timed out", "out of memory")):
                        errors.append(f"{host}: {remark.strip()[:160]}")
                        if "timed out" in remark or "out of memory" in remark:
                            raise OverpassError(
                                "Serwer Overpass przerwał zapytanie (za dużo danych). Zmniejsz obszar, "
                                "maks. odległość lub zawęź tag OSM."
                            )
                        continue
                    self._cache_put(query, data)
                    return data
                if r.status_code == 400:
                    raise OverpassError(f"Serwer Overpass odrzucił zapytanie: {r.text[:300]}")
                errors.append(f"{host}: HTTP {r.status_code}")
                if r.status_code == 429:
                    # limit dla naszego adresu – czekamy zamiast przechodzić na inny serwer
                    retry_after = r.headers.get("Retry-After") if hasattr(r, "headers") else None
                    try:
                        pause = min(float(retry_after), 120.0) if retry_after else 0.0
                    except ValueError:
                        pause = 0.0
                    if pause:
                        notify(f"Serwer OpenStreetMap ogranicza liczbę zapytań – czekam {pause:.0f} s…")
                        self._sleep(pause)
                    break
            attempt += 1
            wait = min(10 * 2 ** (attempt - 1), 120)
            if time.monotonic() - t0 + wait > self.budget_s:
                break
            notify(f"Serwery OpenStreetMap są przeciążone – ponawiam za {wait} s (próba {attempt + 1})…")
            self._sleep(wait)
        raise OverpassError(
            "Nie udało się pobrać danych z OpenStreetMap (Overpass API) – serwery są przeciążone lub "
            "niedostępne. Spróbuj ponownie za kilka minut. Szczegóły: " + "; ".join(errors[-4:])
        )
