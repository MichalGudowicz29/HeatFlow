"""Wiersz poleceń HeatFlow.

    heatflow import-region [szczecin]       – lokalne dane OSM (bez Overpass API)
    heatflow serve [--port 5000]            – interfejs WWW
    heatflow run scenariusz.json -o wyniki/ – analiza bez interfejsu (GeoJSON + CSV)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .analysis import Analysis


def _run(args) -> int:
    cfg = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    analysis = Analysis(cfg)
    analysis.run(lambda f, m: print(f"[{f:4.0%}] {m}", file=sys.stderr))
    excl = f" (wyłączono {analysis.n_excluded}: {', '.join(analysis.exclude)})" if analysis.n_excluded else ""
    print(f"{analysis.name}: {analysis.grid.n_points} ocenianych punktów{excl}, rozstaw {analysis.grid.spacing:g} m, "
          f"{len(analysis.criteria)} kryteriów; dane: {analysis.data_source}")
    for w in analysis.warnings:
        print("UWAGA:", w, file=sys.stderr)
    ev = analysis.evaluate(cfg.get("method", "spotis"), cfg.get("weights") or {"mode": "equal"})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": p}
             for lon, lat, p in analysis.export_rows(ev)]
    meta = {"name": analysis.name, "criteria": ev["criteria"], "method": ev["method"], "weights": ev["weights"],
            "directions": ev["directions"], "excluded_criteria": ev["excluded"],
            "excluded_terrain": analysis.exclude, "n_excluded_points": analysis.n_excluded,
            "grid_spacing_m": analysis.grid.spacing, "data_source": analysis.data_source,
            "osm_timestamp": max(analysis.osm_timestamps) if analysis.osm_timestamps else None}
    (out / "wyniki.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "heatflow": meta, "features": feats}, ensure_ascii=False),
        encoding="utf-8")
    rows = list(analysis.export_rows(ev))
    with open(out / "wyniki.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(rows[0][2].keys())
        w.writerow(["lon", "lat"] + cols)
        for lon, lat, p in rows:
            w.writerow([f"{lon:.6f}", f"{lat:.6f}"] + [p[k] for k in cols])
    print(f"Metoda {ev['method']['name']}, wagi: " +
          ", ".join(f"{k}={v:.3f}" for k, v in ev["weights"].items()))
    print(f"Zapisano {out / 'wyniki.geojson'} i {out / 'wyniki.csv'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="heatflow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="uruchom interfejs WWW")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    imp = sub.add_parser("import-region", help="pobierz i zaimportuj lokalne dane OSM regionu (domyślnie Szczecin)")
    imp.add_argument("region", nargs="?", default="szczecin")
    imp.add_argument("--pbf", help="lokalny plik .osm.pbf zamiast pobierania z Geofabrik")
    r = sub.add_parser("run", help="policz scenariusz z pliku JSON")
    r.add_argument("scenario")
    r.add_argument("-o", "--output", default="wyniki")
    args = p.parse_args(argv)
    from .server import USER_ERRORS
    if args.cmd == "serve":
        from .server import create_app
        print(f"HeatFlow: http://{args.host}:{args.port}")
        create_app().run(host=args.host, port=args.port, threaded=True)
        return 0
    if args.cmd == "import-region":
        from .region import REGIONS, import_region
        if args.region not in REGIONS:
            print(f"Nieznany region {args.region!r}. Dostępne: {', '.join(REGIONS)}", file=sys.stderr)
            return 2
        import_region(args.region, pbf=args.pbf)
        return 0
    try:
        return _run(args)
    except USER_ERRORS as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
