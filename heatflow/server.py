"""Serwer HeatFlow: API analizy i interfejs WWW."""
from __future__ import annotations

import csv
import io
import json
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

from . import storage
from .analysis import Analysis, AnalysisError, describe_options, grid_preview
from .criteria import CriterionError, load_catalog
from .features import DatasetError
from .geo import GeoError
from .mcda import MCDAError
from .network import NetworkError
from .osm import USER_AGENT, OverpassError, tag_filter
from .region import active_region
from .weights import WeightsError, rancom_details

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if not WEB_DIR.exists():
    WEB_DIR = Path(__file__).resolve().parent / "web"
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

USER_ERRORS = (AnalysisError, CriterionError, GeoError, MCDAError, WeightsError, DatasetError,
               OverpassError, NetworkError)
MAX_JOBS = 20
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class Job:
    def __init__(self, analysis: Analysis):
        self.id = uuid.uuid4().hex[:16]
        self.analysis = analysis
        self.status = "queued"
        self.progress = 0.0
        self.message = "W kolejce…"
        self.error: str | None = None
        self.created = time.time()
        self.evals: OrderedDict[str, dict] = OrderedDict()  # ostatnie oceny, do eksportu

    def remember(self, ev: dict) -> None:
        self.evals[ev["eval_id"]] = ev
        while len(self.evals) > 30:
            self.evals.popitem(last=False)

    def to_json(self, include_result: bool = False) -> dict:
        d = {"id": self.id, "status": self.status, "progress": round(self.progress, 3),
             "message": self.message, "error": self.error}
        if include_result and self.status == "done":
            d["result"] = self.analysis.result_json()
        return d


class JobStore:
    def __init__(self):
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(2)  # równoległe analizy (ograniczenie obciążenia Overpass)

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > MAX_JOBS:
                self._jobs.popitem(last=False)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: Job) -> None:
        def run():
            with self._sem:
                job.status = "running"

                def progress(frac, msg):
                    job.progress, job.message = frac, msg

                try:
                    job.analysis.run(progress)
                    job.status = "done"
                except USER_ERRORS as exc:
                    job.status, job.error = "error", str(exc)
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    job.status, job.error = "error", f"Nieoczekiwany błąd: {exc}"

        threading.Thread(target=run, daemon=True).start()


def create_app(client=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = storage.MAX_UPLOAD_BYTES
    jobs = JobStore()
    catalog = load_catalog()
    http = requests.Session()
    http.headers.update({"User-Agent": USER_AGENT})
    geo_lock = threading.Lock()
    geo_state = {"last": 0.0}
    geo_cache: OrderedDict[str, list] = OrderedDict()

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    @app.errorhandler(413)
    def too_large(_):
        return err("Plik jest za duży (limit 50 MB).", 413)

    # --- frontend -------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/<path:path>")
    def static_files(path):
        if path.startswith("api/"):
            return err("Nie znaleziono.", 404)
        return send_from_directory(WEB_DIR, path)

    # --- konfiguracja ---------------------------------------------------
    @app.get("/api/catalog")
    def get_catalog():
        return jsonify(catalog)

    @app.get("/api/region")
    def get_region():
        r = active_region()
        return jsonify(r.info() if r else None)

    @app.post("/api/weights/rancom")
    def post_rancom():
        """Wagi RANCOM liczone przez pymcdm z odpowiedzi eksperta (macierz MAC, ranking lub punktacja)."""
        body = request.get_json(silent=True) or {}
        try:
            n = int(body.get("n") or 0)
        except (TypeError, ValueError):
            n = 0
        if not 1 <= n <= 15:
            return err("Podaj od 1 do 15 kryteriów.")
        try:
            return jsonify(rancom_details(body, n))
        except WeightsError as exc:
            return err(str(exc))

    @app.get("/api/layer-objects")
    def get_layer_objects():
        r = active_region()
        if r is None:
            return err("Wskazywanie konkretnych obiektów wymaga lokalnych danych regionu.", 404)
        key = (request.args.get("key") or "").strip()
        values = [v for v in (request.args.get("values") or "").split("|") if v]
        try:
            tag_filter(key, values or None)  # walidacja jak w zapytaniach OSM
        except ValueError as exc:
            return err(str(exc))
        return jsonify(r.layer_objects(key, values, (request.args.get("q") or "").strip()[:100]))

    @app.get("/api/options")
    def get_options():
        return jsonify(describe_options())

    @app.post("/api/grid-preview")
    def post_grid_preview():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(grid_preview(body.get("area"), body.get("spacing")))
        except USER_ERRORS as exc:
            return err(str(exc))

    @app.get("/api/examples")
    def get_examples():
        out = []
        for p in sorted(EXAMPLES_DIR.glob("*.json")) if EXAMPLES_DIR.exists() else []:
            try:
                out.append({"file": p.name, "name": json.loads(p.read_text(encoding="utf-8")).get("name", p.stem)})
            except (OSError, json.JSONDecodeError):
                continue
        return jsonify(out)

    @app.get("/api/examples/<name>")
    def get_example(name):
        p = EXAMPLES_DIR / Path(name).name
        if p.suffix != ".json" or not p.exists():
            return err("Nie znaleziono przykładu.", 404)
        return Response(p.read_text(encoding="utf-8"), mimetype="application/json")

    # --- analiza --------------------------------------------------------
    @app.post("/api/analyses")
    def post_analysis():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return err("Oczekiwano konfiguracji analizy w formacie JSON.")
        try:
            analysis = Analysis(body, client=client)
        except USER_ERRORS as exc:
            return err(str(exc))
        job = Job(analysis)
        jobs.add(job)
        jobs.start(job)
        return jsonify(job.to_json()), 202

    @app.get("/api/analyses/<job_id>")
    def get_analysis(job_id):
        job = jobs.get(job_id)
        if not job:
            return err("Analiza nie istnieje lub wygasła – uruchom ją ponownie.", 404)
        return jsonify(job.to_json(include_result=request.args.get("result") == "1"))

    @app.post("/api/analyses/<job_id>/evaluate")
    def post_evaluate(job_id):
        job = jobs.get(job_id)
        if not job:
            return err("Analiza nie istnieje lub wygasła – uruchom ją ponownie.", 404)
        if job.status != "done":
            return err("Analiza nie jest jeszcze gotowa.", 409)
        body = request.get_json(silent=True) or {}
        try:
            dirs = body.get("directions") if isinstance(body.get("directions"), dict) else None
            ev = job.analysis.evaluate(body.get("method", "spotis"), body.get("weights") or {}, dirs)
        except USER_ERRORS as exc:
            return err(str(exc))
        job.remember(ev)
        return jsonify(ev)

    @app.get("/api/analyses/<job_id>/export.<fmt>")
    def export(job_id, fmt):
        job = jobs.get(job_id)
        if not job or job.status != "done":
            return err("Brak gotowej analizy do eksportu.", 404)
        a = job.analysis
        eval_id = request.args.get("eval")
        ev = job.evals.get(eval_id) if eval_id else None
        if eval_id and ev is None:
            return err("Ta wersja oceny wygasła – przelicz mapę i wyeksportuj ponownie.", 404)
        import unicodedata
        from urllib.parse import quote
        ascii_name = unicodedata.normalize("NFKD", a.name.replace("ł", "l").replace("Ł", "L"))
        slug = "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in ascii_name
                       if not unicodedata.combining(ch)).strip("_")[:60] or "heatflow"
        utf_name = quote(a.name[:80])

        def disposition(ext):
            return f"attachment; filename=\"{slug}.{ext}\"; filename*=UTF-8''{utf_name}.{ext}"
        if fmt == "geojson":
            feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                      "properties": props} for lon, lat, props in a.export_rows(ev)]
            meta = {
                "name": a.name, "goal": a.goal, "grid_spacing_m": a.grid.spacing,
                "osm_timestamp": max(a.osm_timestamps) if a.osm_timestamps else None,
                "data_source": a.data_source, "excluded_terrain": a.exclude, "n_excluded_points": a.n_excluded,
                "criteria": ev["criteria"] if ev else [c.to_json() for c in a.criteria],
                "method": ev["method"] if ev else None, "weights": ev["weights"] if ev else None,
                "directions": ev["directions"] if ev else None, "excluded": ev["excluded"] if ev else None,
                "score_lower_is_better": ev["method"]["lower_is_better"] if ev else None,
            }
            body = json.dumps({"type": "FeatureCollection", "heatflow": meta, "features": feats}, ensure_ascii=False)
            return Response(body, mimetype="application/geo+json",
                            headers={"Content-Disposition": disposition("geojson")})
        if fmt == "csv":
            buf = io.StringIO()
            rows = list(a.export_rows(ev))
            cols = ["lon", "lat"] + [k for k in rows[0][2].keys()]
            w = csv.writer(buf)
            w.writerow(cols)
            for lon, lat, props in rows:
                w.writerow([f"{lon:.6f}", f"{lat:.6f}"] + [props[k] for k in cols[2:]])
            return Response(buf.getvalue(), mimetype="text/csv",
                            headers={"Content-Disposition": disposition("csv")})
        return err("Nieobsługiwany format eksportu.", 404)

    # --- dane użytkownika -----------------------------------------------
    @app.get("/api/datasets")
    def get_datasets():
        return jsonify(storage.list_datasets())

    @app.post("/api/datasets")
    def post_dataset():
        f = request.files.get("file")
        if not f or not f.filename:
            return err("Nie wybrano pliku.")
        try:
            meta = storage.save_dataset(f.filename, f.read())
        except DatasetError as exc:
            return err(str(exc))
        return jsonify(meta), 201

    @app.delete("/api/datasets/<ds_id>")
    def delete_dataset(ds_id):
        return (jsonify({"ok": True}) if storage.delete_dataset(ds_id) else err("Nie znaleziono pliku.", 404))

    # --- wyszukiwarka miejsc -------------------------------------------
    @app.get("/api/geocode")
    def geocode():
        q = (request.args.get("q") or "").strip()[:200]
        if len(q) < 2:
            return jsonify([])
        region = active_region()
        if region is not None:
            return jsonify(region.search(q))
        key = q.lower()
        if key in geo_cache:
            return jsonify(geo_cache[key])
        # zasady publicznego Nominatim: maks. 1 zapytanie na sekundę dla całej aplikacji
        with geo_lock:
            wait = geo_state["last"] + 1.1 - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                r = http.get(NOMINATIM_URL, params={
                    "q": q, "format": "jsonv2", "limit": 6, "polygon_geojson": 1,
                    "polygon_threshold": 0.0005, "accept-language": "pl",
                }, timeout=15)
                r.raise_for_status()
            except requests.RequestException:
                return err("Wyszukiwarka miejsc (Nominatim) jest chwilowo niedostępna.", 502)
            finally:
                geo_state["last"] = time.monotonic()
        out = []
        for it in r.json():
            gj = it.get("geojson")
            out.append({
                "name": it.get("display_name"),
                "type": it.get("type"),
                "category": it.get("category"),
                "lat": float(it["lat"]), "lon": float(it["lon"]),
                "bbox": [float(it["boundingbox"][2]), float(it["boundingbox"][0]),
                         float(it["boundingbox"][3]), float(it["boundingbox"][1])],
                "boundary": gj if gj and gj.get("type") in ("Polygon", "MultiPolygon") else None,
            })
        geo_cache[key] = out
        while len(geo_cache) > 200:
            geo_cache.popitem(last=False)
        return jsonify(out)

    return app


def main():
    import argparse

    p = argparse.ArgumentParser(description="Serwer HeatFlow")
    p.add_argument("--host", default=os.environ.get("HEATFLOW_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("HEATFLOW_PORT", "5000")))
    args = p.parse_args()
    print(f"HeatFlow: http://{args.host}:{args.port}")
    create_app().run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
