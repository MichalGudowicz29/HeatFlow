"""Katalog roboczy HeatFlow: cache Overpass i wgrane zbiory danych użytkownika."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import mapping

from .features import parse_dataset

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def home() -> Path:
    p = Path(os.environ.get("HEATFLOW_HOME") or Path.cwd() / ".heatflow")
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    return home() / "overpass-cache"


def datasets_dir() -> Path:
    p = home() / "datasets"
    p.mkdir(parents=True, exist_ok=True)
    return p


_ID = re.compile(r"^[a-f0-9]{12}$")


def save_dataset(filename: str, raw: bytes) -> dict:
    geoms = parse_dataset(filename, raw)
    ds_id = uuid.uuid4().hex[:12]
    kinds = sorted({g.geom_type for g in geoms})
    meta = {
        "id": ds_id,
        "name": Path(filename).name,
        "n_features": len(geoms),
        "geometry_types": kinds,
        "uploaded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(g)} for g in geoms
    ]}
    d = datasets_dir()
    (d / f"{ds_id}.geojson").write_text(json.dumps(fc), encoding="utf-8")
    (d / f"{ds_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def list_datasets() -> list[dict]:
    out = []
    for p in sorted(datasets_dir().glob("*.meta.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(out, key=lambda m: m.get("uploaded", ""), reverse=True)


def load_dataset_geoms(ds_id: str) -> list:
    if not _ID.match(ds_id or ""):
        raise FileNotFoundError(ds_id)
    p = datasets_dir() / f"{ds_id}.geojson"
    if not p.exists():
        raise FileNotFoundError(ds_id)
    return parse_dataset(p.name, p.read_bytes())


def delete_dataset(ds_id: str) -> bool:
    if not _ID.match(ds_id or ""):
        return False
    found = False
    for suffix in (".geojson", ".meta.json"):
        p = datasets_dir() / f"{ds_id}{suffix}"
        if p.exists():
            p.unlink()
            found = True
    return found
