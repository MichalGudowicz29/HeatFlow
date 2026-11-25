/* HeatFlow – interfejs WWW.
   Przepływ zgodny z sześcioetapowym procesem: problem → kryteria i wagi → obszar →
   punkty analizy → MCDA → mapa preferencji. */
(() => {
  "use strict";

  // ------------------------------------------------------------------ stałe
  // paleta kategorii: kolejne kolory wyraźnie różne w barwie i jasności; żaden nie jest zielenią akcji (#1f6b3a)
  const CRIT_COLORS = ["#2f6fb3", "#e08214", "#c2366b", "#2a9d9a", "#8a6d1f", "#4d4d4d", "#d4542b", "#5a8a2e",
    "#7fb0e0", "#b07a52", "#e2b42c", "#8c2f2f", "#5fbfa8", "#9a9a9a", "#1d3f6e"];
  const PALETTES = {
    // domyślna: sekwencyjna, im ciemniej, tym lepiej; słabe obszary są półprzezroczyste, więc mapa pod nimi zostaje czytelna
    intensity: { name: "Natężenie: jasne → ciemne", stops: ["#f7f0dc", "#f8d68e", "#f2a457", "#dd6431", "#b2321e", "#6e1611"], fade: true },
    heat: { name: "Chłodne → ciepłe", stops: ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"] },
    rdylgn: { name: "Czerwony → zielony", stops: ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"] },
    viridis: { name: "Viridis", stops: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"] },
    magma: { name: "Magma", stops: ["#000004", "#51127c", "#b73779", "#fc8961", "#fcfdbf"] },
  };
  const METHOD_LABEL = { distance: "Odległość", count: "Liczba w promieniu" };
  const STORAGE_KEY = "heatflow.draft.v2";

  // ------------------------------------------------------------------ stan
  const state = {
    catalog: null, options: null, datasets: [], region: null,
    name: "", goal: "",
    criteria: [],           // {uid, name, source, method, direction, metric, radius, color}
    weights: { mode: "manual", objective: "entropy", manual: {}, ranking: {}, scoring: {}, answers: {}, rancomInput: "pairs", rancomResult: null },
    area: null, areaLabel: "",
    spacing: "auto", maxDistance: 1500, method: "spotis", exclude: ["water"],
    display: { palette: "intensity", reverse: false, opacity: 0.85, smooth: true, scale: "value" },
    job: null, result: null, resultSig: null, resultCriteria: null, evaluation: null,
  };
  let uidSeq = 1;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 0) => Number(v).toLocaleString("pl-PL", { minimumFractionDigits: d, maximumFractionDigits: d });
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  async function api(path, opts = {}) {
    const init = { ...opts };
    if (opts.json !== undefined) {
      init.method = opts.method || "POST";
      init.headers = { "Content-Type": "application/json" };
      init.body = JSON.stringify(opts.json);
    }
    const r = await fetch(path, init);
    let data = null;
    try { data = await r.json(); } catch { /* brak treści */ }
    if (!r.ok) throw new Error((data && data.error) || `Błąd serwera (HTTP ${r.status}).`);
    return data;
  }

  // ------------------------------------------------------------------ mapa
  const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView([53.43, 14.55], 12);
  // podkład Esri Light Gray Canvas (bez klucza API) w dwóch warstwach: tło pod mapą ocen,
  // a warstwa samych napisów (Reference) nad nią – nazwy dzielnic i ulic zostają czytelne
  map.createPane("heatPane").style.zIndex = 390;      // pod obrysem obszaru (overlayPane 400)
  map.getPane("heatPane").classList.add("heat-pane");  // mix-blend-mode: multiply – ulice prześwitują przez kolor
  map.createPane("labelsPane").style.zIndex = 460;     // nad warstwą ocen i terenami wyłączonymi
  map.getPane("labelsPane").style.pointerEvents = "none";
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/";
  const ESRI_ATTR = 'Podkład © <a href="https://www.esri.com/">Esri</a>: Esri, HERE, Garmin, © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, and the GIS user community';
  L.tileLayer(ESRI + "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, maxNativeZoom: 16, attribution: ESRI_ATTR,
  }).addTo(map);
  L.tileLayer(ESRI + "World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}", {
    maxZoom: 19, maxNativeZoom: 16, pane: "labelsPane",
  }).addTo(map);
  map.attributionControl.setPrefix('<a href="https://leafletjs.com">Leaflet</a>');
  const regionLayer = L.geoJSON(null, { style: { color: "#5b6b6d", weight: 1.5, fill: false, opacity: 0.8 }, interactive: false }).addTo(map);
  const areaLayer = L.geoJSON(null, { style: { color: "#0a0a0a", weight: 1.6, dashArray: "5 4", fill: false, opacity: 0.85 } }).addTo(map);
  const previewRenderer = L.canvas({ padding: 0.3 });
  map.createPane("excludedPane").style.zIndex = 450;  // nad warstwą ocen, pod znacznikami
  const excludedLayer = L.geoJSON(null, {
    pane: "excludedPane", interactive: false,
    style: { fillColor: "#c9cecd", fillOpacity: 0.92, color: "#aeb5b4", weight: 0.6 },
  }).addTo(map);
  const featureLayers = {};
  let heatOverlay = null, heatCanvas = null, topLayer = L.layerGroup().addTo(map), hoverMarker = null;
  let drawHandler = null;

  // ------------------------------------------------------------------ kroki
  function openStep(n) {
    $$(".step").forEach((s) => {
      const open = s.dataset.step === String(n) ? !s.classList.contains("open") : false;
      s.classList.toggle("open", open);
      $(".step-head", s).setAttribute("aria-expanded", open);
    });
  }
  $$(".step-head").forEach((h) => h.addEventListener("click", () => openStep(h.closest(".step").dataset.step)));

  function updateSummaries() {
    const set = (n, txt, done) => {
      $(`[data-summary="${n}"]`).textContent = txt;
      $(`.step[data-step="${n}"]`).classList.toggle("done", !!done);
    };
    set(1, state.name || "", !!state.name);
    const n = state.criteria.length;
    set(2, n ? `${n} ${n === 1 ? "kryterium" : n < 5 ? "kryteria" : "kryteriów"}, wagi: ${weightModeLabel()}` : "", n > 0 && weightsValid());
    set(3, state.area ? state.areaLabel : "", !!state.area);
    set(4, gridSummary || "", !!state.area);
    const m = (state.options?.methods || []).find((x) => x.id === state.method);
    set(5, m ? m.name : "", true);
    renderChecklist();
    saveDraft();
  }

  function weightModeLabel() {
    return { manual: "ręczne", rancom: "RANCOM", objective: { entropy: "entropia", critic: "CRITIC", std: "odch. std." }[state.weights.objective], equal: "równe" }[state.weights.mode];
  }

  // ------------------------------------------------------------------ krok 1
  $("#name").addEventListener("input", (e) => { state.name = e.target.value; updateSummaries(); });
  $("#goal").addEventListener("input", (e) => { state.goal = e.target.value; saveDraft(); });

  async function loadExamples() {
    try {
      const list = await api("/api/examples");
      const box = $("#examples");
      box.innerHTML = "";
      list.forEach((ex) => {
        const b = document.createElement("button");
        b.type = "button"; b.className = "btn btn--secondary btn--sm";
        b.innerHTML = `<span class="ex-k">Przykład:</span> ${esc(ex.name)}`;
        b.addEventListener("click", async () => applyScenario(await api(`/api/examples/${encodeURIComponent(ex.file)}`)));
        box.appendChild(b);
      });
    } catch { /* przykłady są opcjonalne */ }
  }

  function fitArea() {
    const b = areaLayer.getBounds();
    if (!b.isValid()) return;
    const wide = window.innerWidth > 820;
    const right = wide && !$("#results").hidden && !$("#results").classList.contains("collapsed") ? $("#results").offsetWidth + 36 : 30;
    map.fitBounds(b, { paddingTopLeft: [30, wide ? 70 : 60], paddingBottomRight: [right, 30] });
  }

  // ------------------------------------------------------------------ krok 2: kryteria
  function nextColor() {
    const used = new Set(state.criteria.map((c) => c.color));
    return CRIT_COLORS.find((c) => !used.has(c)) || CRIT_COLORS[state.criteria.length % CRIT_COLORS.length];
  }

  function addCriterion(def) {
    const maxN = state.options?.limits?.max_criteria || 15;
    if (state.criteria.length >= maxN) { alert(`Maksymalnie ${maxN} kryteriów.`); return; }
    const c = {
      uid: "c" + (uidSeq++),
      name: def.name, source: def.source,
      method: def.method || "distance",
      direction: def.direction || (def.method === "count" ? "benefit" : "cost"),
      metric: def.metric || "euclidean",
      radius: def.radius || 500,
      maxDistance: def.maxDistance || null,  // null = limit z kroku „Punkty analizy”
      color: nextColor(),
      catalogId: def.catalogId || null,
      scope: def.scope || "all",
    };
    state.criteria.push(c);
    state.weights.manual[c.uid] = def.weight ?? 1;
    state.weights.ranking[c.uid] = def.rank ?? state.criteria.length;
    renderCriteria(); renderWeights(); renderCatalog(); onDataChanged();
  }

  function sourceLabel(src) {
    if (src.type === "dataset") {
      const ds = state.datasets.find((d) => d.id === src.dataset_id);
      return `Plik: ${ds ? ds.name : src.name || src.dataset_id}`;
    }
    const v = src.values && src.values.length ? src.values.join(" | ") : "dowolna wartość";
    const n = (src.refs || []).length;
    return `OSM: ${src.key} = ${v}` + (n ? `, tylko wybrane obiekty (${n})` : "");
  }

  function renderCriteria() {
    const box = $("#criteriaList");
    box.innerHTML = "";
    const walkOk = true;
    state.criteria.forEach((c) => {
      const el = document.createElement("div");
      el.className = "crit";
      el.style.setProperty("--c", c.color);
      el.innerHTML = `
        <div class="crit-top">
          <input class="crit-name" value="${esc(c.name)}" aria-label="Nazwa kryterium">
          <button class="icon-btn" type="button" title="Usuń kryterium" aria-label="Usuń kryterium ${esc(c.name)}">×</button>
        </div>
        <div class="crit-src">${esc(sourceLabel(c.source))}</div>
        <div class="crit-opts">
          <select data-k="method" aria-label="Sposób pomiaru">
            <option value="distance">${METHOD_LABEL.distance}</option>
            <option value="count">${METHOD_LABEL.count}</option>
          </select>
          ${c.method === "distance"
            ? `<select data-k="metric" aria-label="Rodzaj odległości">
                 <option value="euclidean">w linii prostej</option>
                 <option value="walking"${walkOk ? "" : " disabled"}>piesza (po sieci dróg)</option>
               </select>`
            : `<select data-k="radius" aria-label="Promień">
                 ${[100, 200, 300, 400, 500, 750, 1000, 1500, 2000].map((r) => `<option value="${r}">promień ${fmt(r)} m</option>`).join("")}
               </select>`}
          ${c.method === "distance" ? `<select data-k="limit" class="full" aria-label="Limit odległości">
              <option value="">limit jak w kroku 4 (${fmt(state.maxDistance)} m)</option>
              ${[500, 800, 1000, 1500, 2000, 3000, 5000].map((m) => `<option value="${m}">limit ${fmt(m)} m</option>`).join("")}
            </select>` : ""}
          <div class="dir full" role="radiogroup" aria-label="Kierunek preferencji">
            <button type="button" data-dir="benefit">${c.method === "distance" ? "Dalej = lepiej" : "Więcej = lepiej"}</button>
            <button type="button" data-dir="cost">${c.method === "distance" ? "Bliżej = lepiej" : "Mniej = lepiej"}</button>
          </div>
          ${c.source.type === "osm" && state.region ? `<label class="check full scope"><input type="checkbox" data-k="scope"${c.scope === "some" ? " checked" : ""}>
              Tylko wybrane obiekty</label>` : ""}
        </div>
        ${c.scope === "some" ? `<div class="objpick">
            <input type="search" placeholder="Szukaj po nazwie" aria-label="Szukaj obiektu: ${esc(c.name)}">
            <div class="objsel"></div>
            <div class="objlist" aria-live="polite"></div>
          </div>` : ""}`;
      $("[data-k=method]", el).value = c.method;
      if (c.method === "distance") $("[data-k=metric]", el).value = c.metric;
      else $("[data-k=radius]", el).value = String(c.radius);
      $$(".dir button", el).forEach((b) => {
        b.classList.toggle("active", b.dataset.dir === c.direction);
        b.setAttribute("aria-pressed", b.dataset.dir === c.direction);
        b.addEventListener("click", () => { c.direction = b.dataset.dir; renderCriteria(); onWeightsChanged(); });
      });
      $(".crit-name", el).addEventListener("input", (e) => { c.name = e.target.value; renderWeights(); updateSummaries(); });
      $(".icon-btn", el).addEventListener("click", () => {
        state.criteria = state.criteria.filter((x) => x !== c);
        normalizeRanks();
        renderCriteria(); renderWeights(); renderCatalog(); onDataChanged();
      });
      $("[data-k=method]", el).addEventListener("change", (e) => {
        c.method = e.target.value;
        c.direction = c.method === "count" ? "benefit" : "cost";
        renderCriteria(); onDataChanged();
      });
      const metric = $("[data-k=metric]", el);
      if (metric) metric.addEventListener("change", (e) => { c.metric = e.target.value; onDataChanged(); });
      const rad = $("[data-k=radius]", el);
      if (rad) rad.addEventListener("change", (e) => { c.radius = Number(e.target.value); onDataChanged(); });
      const lim = $("[data-k=limit]", el);
      if (lim) {
        if (c.maxDistance && ![...lim.options].some((o) => o.value === String(c.maxDistance))) lim.add(new Option(`limit ${fmt(c.maxDistance)} m`, String(c.maxDistance)));
        lim.value = c.maxDistance ? String(c.maxDistance) : "";
        lim.addEventListener("change", (e) => { c.maxDistance = e.target.value ? Number(e.target.value) : null; onDataChanged(); });
      }
      const scope = $("[data-k=scope]", el);
      if (scope) {
        scope.addEventListener("change", (e) => {
          c.scope = e.target.checked ? "some" : "all";
          if (c.scope === "all") { delete c.source.refs; delete c.source.ref_names; }
          renderCriteria(); onDataChanged();
        });
      }
      if (c.scope === "some") setupObjectPicker(c, el);
      box.appendChild(el);
    });
    updateSummaries();
  }

  // pozycje rankingu RANCOM po usunięciu kryterium: gęsta numeracja 1..k z zachowaniem remisów
  function normalizeRanks() {
    const vals = [...new Set(state.criteria.map((c) => Number(state.weights.ranking[c.uid] || 1)))].sort((a, b) => a - b);
    state.criteria.forEach((c) => { state.weights.ranking[c.uid] = vals.indexOf(Number(state.weights.ranking[c.uid] || 1)) + 1; });
  }

  // wybór konkretnych obiektów warstwy (np. jedna uczelnia zamiast wszystkich)
  function setupObjectPicker(c, el) {
    const input = $(".objpick input", el), list = $(".objlist", el), sel = $(".objsel", el);
    // refs i ref_names są równoległe; obiekty o tej samej nazwie dodajemy i usuwamy razem
    const renderSel = () => {
      const names = [...new Set(c.source.ref_names || [])];
      sel.innerHTML = names.length
        ? names.map((nm) => `<span class="chip">${esc(nm)}<button type="button" data-n="${esc(nm)}" aria-label="Usuń ${esc(nm)}">×</button></span>`).join("")
        : '<p class="hint">Zaznacz co najmniej jeden obiekt.</p>';
      $$("button", sel).forEach((b) => b.addEventListener("click", () => toggle(b.dataset.n, [], false)));
      $(".crit-src", el).textContent = sourceLabel(c.source);
    };
    const toggle = (name, refsOfName, on) => {
      let refs = c.source.refs || [], names = c.source.ref_names || [];
      const keep = names.map((nm) => nm !== name);
      refs = refs.filter((_, k) => keep[k]); names = names.filter((_, k) => keep[k]);
      if (on) refsOfName.forEach((r) => { refs.push(r); names.push(name); });
      c.source.refs = refs; c.source.ref_names = names;
      renderSel(); onDataChanged();
      $$(".objlist input", el).forEach((x) => { if (x.dataset.n === name) x.checked = on; });
    };
    const load = debounce(async () => {
      const qs = new URLSearchParams({ key: c.source.key, values: (c.source.values || []).join("|"), q: input.value.trim() });
      try {
        const res = await api(`/api/layer-objects?${qs}`);
        const chosen = new Set(c.source.ref_names || []);
        list.innerHTML = res.items.length ? res.items.map((it, k) =>
          `<label class="check"><input type="checkbox" data-k="${k}" data-n="${esc(it.name)}" ${chosen.has(it.name) ? "checked" : ""}> <span>${esc(it.name)}${
            it.refs.length > 1 ? ` <small>(${it.refs.length} obiekty)</small>` : ""}</span></label>`).join("")
          + (res.truncated ? '<p class="hint">Pokazano pierwsze wyniki – zawęź wyszukiwanie.</p>' : "")
          : '<p class="hint">Brak nazwanych obiektów pasujących do wyszukiwania.</p>';
        $$("input", list).forEach((x) => x.addEventListener("change", () => {
          const it = res.items[Number(x.dataset.k)];
          toggle(it.name, it.refs, x.checked);
        }));
      } catch (e) { list.innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
    }, 250);
    input.addEventListener("input", load);
    renderSel(); load();
  }

  // katalog
  $("#btnAddCriterion").addEventListener("click", () => {
    const p = $("#picker");
    p.hidden = !p.hidden;
    $("#btnAddCriterion").textContent = p.hidden ? "Dodaj kryterium" : "Zamknij wybór";
    if (!p.hidden) $("#catalogSearch").focus();
  });
  $$(".picker-tabs button").forEach((b) => b.addEventListener("click", () => {
    $$(".picker-tabs button").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    $$(".picker-pane").forEach((p) => { p.hidden = p.dataset.pane !== b.dataset.tab; });
  }));
  $("#catalogSearch").addEventListener("input", () => renderCatalog());

  function renderCatalog() {
    if (!state.catalog) return;
    const q = $("#catalogSearch").value.trim().toLowerCase();
    const used = new Set(state.criteria.map((c) => c.catalogId).filter(Boolean));
    const box = $("#catalogList");
    box.innerHTML = "";
    state.catalog.categories.forEach((cat) => {
      const items = state.catalog.layers.filter((l) => l.category === cat.id &&
        (!q || l.name.toLowerCase().includes(q) || l.osm.key.includes(q) || (l.osm.values || []).some((v) => v.includes(q))));
      if (!items.length) return;
      const h = document.createElement("h4");
      h.textContent = cat.name;
      box.appendChild(h);
      items.forEach((l) => {
        const b = document.createElement("button");
        b.type = "button";
        b.disabled = used.has(l.id);
        b.title = l.description || "";
        b.innerHTML = `<span>${esc(l.name)}</span><small>${esc(l.osm.key)}=${esc((l.osm.values || ["*"]).join("|"))}</small>`;
        b.addEventListener("click", () => addCriterion({
          name: l.name, catalogId: l.id, source: { type: "osm", key: l.osm.key, values: l.osm.values },
          method: l.default.method, direction: l.default.direction,
        }));
        box.appendChild(b);
      });
    });
    if (!box.children.length) box.innerHTML = '<p class="hint">Brak wyników. Dodaj obiekt jako własny tag OSM.</p>';
  }

  $("#btnAddCustom").addEventListener("click", () => {
    const key = $("#customKey").value.trim();
    const vals = $("#customValues").value.split("|").map((s) => s.trim()).filter((s) => s && s !== "*");
    if (!/^[\w:\- ]+$/u.test(key)) { alert("Podaj poprawny klucz tagu OSM, np. amenity."); return; }
    if (vals.some((v) => !/^[\w:\- ]+$/u.test(v))) { alert("Wartości mogą zawierać litery, cyfry, _, - i :. Rozdziel je znakiem |."); return; }
    addCriterion({
      name: $("#customName").value.trim() || `${key}=${vals.join("|") || "*"}`,
      source: { type: "osm", key, values: vals.length ? vals : null }, method: "distance",
    });
    $("#customKey").value = $("#customValues").value = $("#customName").value = "";
  });

  // własne dane
  async function loadDatasets() {
    try { state.datasets = await api("/api/datasets"); } catch { state.datasets = []; }
    renderDatasets();
  }
  function renderDatasets() {
    const box = $("#datasetList");
    box.innerHTML = state.datasets.length ? "" : '<p class="hint">Nie wgrano jeszcze żadnych plików.</p>';
    state.datasets.forEach((d) => {
      const row = document.createElement("div");
      row.className = "ds";
      row.innerHTML = `<span>${esc(d.name)}<br><small>${fmt(d.n_features)} obiektów · ${esc(d.geometry_types.join(", "))}</small></span>
        <button type="button" class="btn btn--secondary btn--sm" data-a="add">Dodaj</button>
        <button type="button" class="icon-btn" data-a="del" title="Usuń plik" aria-label="Usuń plik ${esc(d.name)}">×</button>`;
      $("[data-a=add]", row).addEventListener("click", () => addCriterion({
        name: d.name.replace(/\.[^.]+$/, ""), source: { type: "dataset", dataset_id: d.id, name: d.name }, method: "distance",
      }));
      $("[data-a=del]", row).addEventListener("click", async () => {
        if (!confirm(`Usunąć plik „${d.name}”?`)) return;
        await api(`/api/datasets/${d.id}`, { method: "DELETE" });
        loadDatasets();
      });
      box.appendChild(row);
    });
  }
  $("#datasetFile").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const meta = await api("/api/datasets", { method: "POST", body: fd });
      await loadDatasets();
      addCriterion({ name: meta.name.replace(/\.[^.]+$/, ""), source: { type: "dataset", dataset_id: meta.id, name: meta.name }, method: "distance" });
    } catch (err) { alert(err.message); }
    e.target.value = "";
  });

  // ------------------------------------------------------------------ krok 2: wagi
  $$("#weightMode button").forEach((b) => b.addEventListener("click", () => {
    const was = state.weights.mode;
    state.weights.mode = b.dataset.mode;
    renderWeights(); onWeightsChanged();
    // RANCOM: od razu rozpoczynamy serię pytań (jak RANCOM(object_names=...) w pymcdm)
    if (b.dataset.mode === "rancom" && was !== "rancom" && state.weights.rancomInput === "pairs"
        && state.criteria.length >= 2 && !rancomComplete()) openRancomWizard();
  }));
  $("#objectiveMethod").addEventListener("change", (e) => { state.weights.objective = e.target.value; renderWeights(); onWeightsChanged(); });
  $$("#rancomInput button").forEach((b) => b.addEventListener("click", () => {
    state.weights.rancomInput = b.dataset.input;
    renderWeights(); onWeightsChanged();
    if (b.dataset.input === "pairs" && state.criteria.length >= 2 && !rancomComplete()) openRancomWizard();
  }));
  $("#btnRancomStart").addEventListener("click", () => openRancomWizard(rancomComplete() ? 0 : null));

  // --- RANCOM: porównania parami → macierz MAC → wagi z pymcdm (backend)
  const pairKey = (a, b) => `${a.uid}|${b.uid}`;
  function rancomPairs() {
    const out = [], cs = state.criteria;
    for (let i = 0; i < cs.length; i++) for (let j = i + 1; j < cs.length; j++) out.push([i, j]);
    return out;
  }
  function answerFor(i, j) {
    const a = state.criteria[i], b = state.criteria[j], ans = state.weights.answers;
    if (ans[pairKey(a, b)] !== undefined) return ans[pairKey(a, b)];
    if (ans[pairKey(b, a)] !== undefined) return 1 - ans[pairKey(b, a)];
    return undefined;
  }
  function setAnswer(i, j, v) {
    const a = state.criteria[i], b = state.criteria[j];
    delete state.weights.answers[pairKey(b, a)];
    state.weights.answers[pairKey(a, b)] = v;
  }
  const rancomAnswered = () => rancomPairs().filter(([i, j]) => answerFor(i, j) !== undefined).length;
  const rancomComplete = () => rancomAnswered() === rancomPairs().length;
  function rancomMatrix() {
    const n = state.criteria.length;
    const m = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 0.5 : null)));
    for (const [i, j] of rancomPairs()) {
      const v = answerFor(i, j);
      if (v === undefined) return null;
      m[i][j] = v; m[j][i] = 1 - v;
    }
    return m;
  }
  function rancomSpec() {
    const inp = state.weights.rancomInput;
    if (inp === "ranking") return { mode: "rancom", ranking: state.criteria.map((c) => Number(state.weights.ranking[c.uid] || 1)) };
    if (inp === "scoring") return { mode: "rancom", scoring: state.criteria.map((c) => Number(state.weights.scoring[c.uid] ?? 50)) };
    const m = rancomMatrix();
    return m ? { mode: "rancom", matrix: m } : null;
  }
  const refreshRancom = debounce(async () => {
    if (state.weights.mode !== "rancom") return;
    const spec = rancomSpec();
    const sig = JSON.stringify(spec);
    if (!spec || state.criteria.length < 1) { state.weights.rancomResult = null; renderRancomInfo(); refreshWeightValues(); return; }
    if (state.weights.rancomResult && state.weights.rancomResult.sig === sig) { renderRancomInfo(); return; }
    try {
      const data = await api("/api/weights/rancom", { json: { n: state.criteria.length, ...spec } });
      if (JSON.stringify(rancomSpec()) !== sig) return; // w międzyczasie zmieniono odpowiedzi
      state.weights.rancomResult = { sig, ...data };
    } catch (e) {
      state.weights.rancomResult = { sig, error: e.message };
    }
    renderRancomInfo(); refreshWeightValues(); onWeightsChanged();
  }, 150);

  function renderRancomInfo() {
    const box = $("#rancomInfo");
    if (state.weights.mode !== "rancom") { box.innerHTML = ""; return; }
    const n = state.criteria.length, res = state.weights.rancomResult;
    const inp = state.weights.rancomInput;
    const btn = $("#btnRancomStart");
    btn.hidden = inp !== "pairs" || n < 2;
    const total = rancomPairs().length, done = rancomAnswered();
    btn.textContent = done === 0 ? "Rozpocznij porównania" : done < total ? `Dokończ porównania (${done} z ${total})` : "Popraw odpowiedzi";
    let html = "";
    if (inp === "pairs" && n >= 2 && done < total) {
      html += `<p class="hint">Odpowiedziano na ${done} z ${total} pytań. Wagi zostaną wyznaczone po porównaniu wszystkich par.</p>`;
    }
    if (res && res.error) html += `<div class="alert error">${esc(res.error)}</div>`;
    else if (res && res.matrix && n >= 2 && JSON.stringify(rancomSpec()) === res.sig) {
      const head = state.criteria.map((c) => `<th title="${esc(c.name)}"><span class="sw" style="--c:${c.color}"></span></th>`).join("");
      const rows = res.matrix.map((r, i) => `<tr><th title="${esc(state.criteria[i].name)}"><span class="sw" style="--c:${state.criteria[i].color}"></span></th>${
        r.map((v, j) => `<td class="${i === j ? "diag" : ""}">${v === 0.5 ? "½" : fmt(v)}</td>`).join("")}<td class="scw">${fmt(res.scw[i], 1)}</td></tr>`).join("");
      html += `<details class="mac"><summary>Macierz porównań MAC i sumy SCW</summary>
        <table><thead><tr><th></th>${head}<th>SCW</th></tr></thead><tbody>${rows}</tbody></table>
        <p class="hint">Wiersz = kryterium, 1 – ważniejsze od kryterium w kolumnie, ½ – równie ważne, 0 – mniej ważne. Waga = SCW / ΣSCW (pymcdm RANCOM).</p></details>`;
      if (res.inconsistent_triads > 0) {
        html += `<p class="hint">Niespójne trójki odpowiedzi: ${res.inconsistent_triads} z ${res.n_triads}. RANCOM uwzględnia je w wagach; jeśli to pomyłka, popraw odpowiedzi.</p>`;
      }
    }
    box.innerHTML = html;
  }

  // kreator pytań
  let wiz = null;
  function openRancomWizard(startAt = null) {
    const pairs = rancomPairs();
    if (pairs.length === 0) return;
    let k = startAt ?? pairs.findIndex(([i, j]) => answerFor(i, j) === undefined);
    if (k < 0) k = 0;
    wiz = { pairs, k, review: startAt === 0 };
    renderWizard();
    const d = $("#rancomDialog");
    if (!d.open) d.showModal();
  }
  function renderWizard() {
    const { pairs, k } = wiz;
    const [i, j] = pairs[k];
    const a = state.criteria[i], b = state.criteria[j], cur = answerFor(i, j);
    $("#wizProgress").textContent = `Pytanie ${k + 1} z ${pairs.length}`;
    $("#wizBar").style.width = `${(rancomAnswered() / pairs.length) * 100}%`;
    const card = (c, v, label) => `<button type="button" class="wiz-opt${cur === v ? " chosen" : ""}" data-v="${v}" style="--c:${c ? c.color : "var(--muted)"}">
        <span class="wiz-key">${label}</span><span class="wiz-name">${c ? esc(c.name) : "Równie ważne"}</span>
        ${c ? `<small>${esc(sourceLabel(c.source))}</small>` : "<small>oba kryteria mają taką samą ważność</small>"}</button>`;
    $("#wizOptions").innerHTML = card(a, 1, "1") + card(null, 0.5, "½") + card(b, 0, "0");
    $$("#wizOptions .wiz-opt").forEach((btn) => btn.addEventListener("click", () => answerWizard(Number(btn.dataset.v))));
    $("#wizBack").disabled = k === 0;
    const chosen = $("#wizOptions .wiz-opt.chosen");
    (chosen || $("#wizOptions .wiz-opt")).focus();
  }
  function answerWizard(v) {
    const [i, j] = wiz.pairs[wiz.k];
    setAnswer(i, j, v);
    state.weights.rancomResult = null;
    if (wiz.review) {
      // poprawianie odpowiedzi: przechodzimy kolejno przez wszystkie pytania
      if (wiz.k + 1 < wiz.pairs.length) { wiz.k++; renderWizard(); } else closeWizard();
    } else {
      const next = wiz.pairs.findIndex(([a, b]) => answerFor(a, b) === undefined);
      if (next < 0) closeWizard(); else { wiz.k = next; renderWizard(); }
    }
    renderRancomInfo(); refreshRancom(); saveDraft();
  }
  function closeWizard() {
    const d = $("#rancomDialog");
    if (d.open) d.close();
    wiz = null;
    renderWeights(); refreshRancom(); onWeightsChanged();
  }
  $("#wizBack").addEventListener("click", () => { if (wiz && wiz.k > 0) { wiz.k--; renderWizard(); } });
  $("#wizClose").addEventListener("click", closeWizard);
  $("#rancomDialog").addEventListener("cancel", (e) => { e.preventDefault(); closeWizard(); });
  $("#rancomDialog").addEventListener("keydown", (e) => {
    if (!wiz) return;
    if (e.key === "ArrowLeft" || e.key === "1") { e.preventDefault(); answerWizard(1); }
    else if (e.key === "ArrowDown" || e.key === "ArrowUp" || e.key === "=" ) { e.preventDefault(); answerWizard(0.5); }
    else if (e.key === "ArrowRight" || e.key === "0") { e.preventDefault(); answerWizard(0); }
  });

  function currentWeightsPreview() {
    const n = state.criteria.length;
    if (!n) return [];
    switch (state.weights.mode) {
      case "manual": {
        const v = state.criteria.map((c) => Math.max(0, Number(state.weights.manual[c.uid]) || 0));
        const s = v.reduce((a, b) => a + b, 0);
        return s > 0 ? v.map((x) => x / s) : v.map(() => 0);
      }
      case "rancom": {
        if (n === 1) return [1];
        const res = state.weights.rancomResult, spec = rancomSpec();
        if (res && res.weights && spec && res.sig === JSON.stringify(spec)) return res.weights;
        return state.criteria.map(() => null);
      }
      case "equal": return state.criteria.map(() => 1 / n);
      default: {
        const ev = state.evaluation;
        if (ev && !isStale() && ev.weights_mode === state.weights.objective) {
          return state.criteria.map((c) => ev.weights_input[c.uid] ?? null);
        }
        return state.criteria.map(() => null);
      }
    }
  }

  function weightsValid() {
    if (!state.criteria.length) return false;
    if (state.weights.mode === "manual") return currentWeightsPreview().some((w) => w > 0);
    if (state.weights.mode === "rancom") return state.criteria.length === 1 || !!rancomSpec();
    return true;
  }

  function renderWeights() {
    $$("#weightMode button").forEach((b) => {
      b.classList.toggle("active", b.dataset.mode === state.weights.mode);
      b.setAttribute("aria-checked", b.dataset.mode === state.weights.mode);
    });
    const mode = state.weights.mode, inp = state.weights.rancomInput;
    $("#objectiveField").hidden = mode !== "objective";
    $("#objectiveMethod").value = state.weights.objective;
    $("#rancomBox").hidden = mode !== "rancom";
    $$("#rancomInput button").forEach((b) => { b.classList.toggle("active", b.dataset.input === inp); b.setAttribute("aria-checked", b.dataset.input === inp); });
    $("#weightHint").textContent = {
      manual: "Wpisz istotność każdego kryterium. Wagi są normalizowane do sumy 100%.",
      rancom: {
        pairs: "RANCOM: odpowiedz na serię pytań, porównując ważność każdej pary kryteriów. Z odpowiedzi powstaje macierz porównań, z której pymcdm wyznacza wagi – także wtedy, gdy odpowiedzi są ze sobą sprzeczne.",
        ranking: "RANCOM: ustal ranking ważności (1 = najważniejsze, remisy dozwolone). pymcdm buduje z niego macierz porównań i wyznacza wagi.",
        scoring: "RANCOM: oceń ważność każdego kryterium w skali 0–100 (więcej = ważniejsze). pymcdm zamienia punktację na ranking i wyznacza wagi.",
      }[inp],
      objective: {
        entropy: "Wagi z danych: większą wagę dostają kryteria, których wartości są w obszarze rozłożone bardziej nierównomiernie (entropia Shannona).",
        critic: "Wagi z danych: zróżnicowanie kryterium i jego niezależność od pozostałych – im słabsza lub bardziej przeciwna korelacja, tym większa waga (CRITIC).",
        std: "Wagi z danych: proporcjonalne do odchylenia standardowego znormalizowanych wartości.",
      }[state.weights.objective] + " Zostaną policzone po pobraniu danych.",
      equal: "Każde kryterium ma taką samą wagę.",
    }[mode];
    const box = $("#weightsList");
    box.innerHTML = "";
    const preview = currentWeightsPreview();
    const n = state.criteria.length;
    state.criteria.forEach((c, i) => {
      const row = document.createElement("div");
      row.className = "wrow";
      row.style.setProperty("--c", c.color);
      let input = "<span></span>";
      if (mode === "manual") {
        input = `<input type="number" min="0" max="100" step="1" value="${esc(state.weights.manual[c.uid] ?? 1)}" aria-label="Waga: ${esc(c.name)}">`;
      } else if (mode === "rancom" && inp === "ranking") {
        input = `<select aria-label="Pozycja w rankingu: ${esc(c.name)}">${Array.from({ length: n }, (_, k) => `<option value="${k + 1}">${k + 1}. miejsce</option>`).join("")}</select>`;
      } else if (mode === "rancom" && inp === "scoring") {
        input = `<input type="number" min="0" max="100" step="1" value="${esc(state.weights.scoring[c.uid] ?? 50)}" aria-label="Punkty: ${esc(c.name)}">`;
      }
      const w = preview[i];
      row.innerHTML = `<span class="wname" title="${esc(c.name)}">${esc(c.name)}</span>${input}
        <span class="wval">${w == null ? "–" : fmt(w * 100, w * 100 < 10 ? 1 : 0) + "%"}</span>
        <div class="wbar"><span style="width:${w == null ? 0 : (w * 100).toFixed(1)}%"></span></div>`;
      const num = $("input", row);
      if (num && mode === "manual") num.addEventListener("input", (e) => { state.weights.manual[c.uid] = e.target.value === "" ? 0 : Number(e.target.value); refreshWeightValues(); onWeightsChanged(); });
      if (num && mode === "rancom") num.addEventListener("input", (e) => { state.weights.scoring[c.uid] = e.target.value === "" ? 0 : Number(e.target.value); refreshRancom(); });
      const sel = $("select", row);
      if (sel) {
        sel.value = String(Math.min(state.weights.ranking[c.uid] || 1, n));
        sel.addEventListener("change", (e) => { state.weights.ranking[c.uid] = Number(e.target.value); refreshRancom(); });
      }
      box.appendChild(row);
    });
    renderRancomInfo();
    if (mode === "rancom") refreshRancom();
    updateSummaries();
  }

  function refreshWeightValues() {
    const preview = currentWeightsPreview();
    $$("#weightsList .wrow").forEach((row, i) => {
      const w = preview[i];
      $(".wval", row).textContent = w == null ? "–" : fmt(w * 100, w * 100 < 10 ? 1 : 0) + "%";
      $(".wbar span", row).style.width = w == null ? "0" : (w * 100).toFixed(1) + "%";
    });
    updateSummaries();
  }

  function weightsSpec() {
    const m = state.weights.mode;
    if (m === "manual") return { mode: "manual", values: state.criteria.map((c) => Math.max(0, Number(state.weights.manual[c.uid]) || 0)) };
    if (m === "rancom") {
      if (state.criteria.length === 1) return { mode: "rancom", ranking: [1] };
      return rancomSpec() || { mode: "rancom", matrix: null };
    }
    if (m === "objective") return { mode: state.weights.objective };
    return { mode: "equal" };
  }

  // ------------------------------------------------------------------ krok 3: obszar
  const drawStyle = { color: "#1f4e8c", weight: 2, fillOpacity: 0.06 };
  function startDraw(kind) {
    if (drawHandler) drawHandler.disable();
    drawHandler = kind === "rect"
      ? new L.Draw.Rectangle(map, { shapeOptions: drawStyle, showArea: false })
      : new L.Draw.Polygon(map, { shapeOptions: drawStyle, showArea: false, allowIntersection: false });
    drawHandler.enable();
  }
  $("#btnDrawRect").addEventListener("click", () => startDraw("rect"));
  $("#btnDrawPoly").addEventListener("click", () => startDraw("poly"));
  $("#btnClearArea").addEventListener("click", () => setArea(null));
  map.on(L.Draw.Event.CREATED, (e) => {
    const g = e.layer.toGeoJSON().geometry;
    setArea(g, e.layerType === "rectangle" ? "Prostokąt narysowany na mapie" : "Wielokąt narysowany na mapie");
  });

  function setArea(geom, label = "", fit = true) {
    state.area = geom;
    state.areaLabel = label;
    areaLayer.clearLayers();
    if (geom) {
      areaLayer.addData(geom);
      if (fit) fitArea();
    }
    onDataChanged();
    refreshGrid();
  }

  let gridSummary = "";
  const refreshGrid = debounce(async () => {
    if (!state.area) {
      $("#areaInfo").textContent = "Nie wybrano obszaru.";
      $("#gridInfo").textContent = "Wybierz obszar, aby zobaczyć liczbę punktów.";
      gridSummary = ""; updateSummaries(); return;
    }
    try {
      const p = await api("/api/grid-preview", { json: { area: state.area, spacing: state.spacing } });
      if (p.clipped && p.area) {
        // obszar wychodził poza granicę miasta – pokazujemy i zapisujemy wersję przyciętą
        state.area = p.area;
        state.areaLabel = `${state.areaLabel.replace(/ \(przycięty.*\)$/, "")} (przycięty do granicy miasta)`;
        areaLayer.clearLayers(); areaLayer.addData(p.area);
        onDataChanged();
      }
      $("#areaInfo").innerHTML = `${esc(state.areaLabel)}<br>Powierzchnia <b>${fmt(p.area_km2, p.area_km2 < 10 ? 2 : 1)} km²</b>`;
      $("#gridInfo").innerHTML = `Siatka co <b>${fmt(p.spacing)} m</b>: <b>${fmt(p.n_points)}</b> punktów analizy.` +
        (state.exclude.length ? " Punkty na wyłączonych terenach zostaną pominięte." : "") +
        (state.spacing === "auto" ? " Rozstaw dobrany do wielkości obszaru." : "");
      gridSummary = `${fmt(p.n_points)} pkt co ${fmt(p.spacing)} m`;
      $("#runError").hidden = true;
    } catch (err) {
      $("#gridInfo").innerHTML = `<span class="alert error">${esc(err.message)}</span>`;
      gridSummary = "";
    }
    updateSummaries();
  }, 250);

  // wyszukiwarka
  const searchInput = $("#placeSearch");
  const resultsBox = $("#placeResults");
  async function doSearch() {
    const q = searchInput.value.trim();
    if (q.length < 2) { resultsBox.hidden = true; return; }
    resultsBox.innerHTML = '<li aria-disabled="true">Szukam…</li>';
    resultsBox.hidden = false;
    try {
      const items = await api(`/api/geocode?q=${encodeURIComponent(q)}`);
      resultsBox.innerHTML = "";
      if (!items.length) { resultsBox.innerHTML = '<li aria-disabled="true">Nie znaleziono miejsca.</li>'; }
      items.forEach((it) => {
        const li = document.createElement("li");
        const [main, ...rest] = it.name.split(", ");
        const where = [it.type, ...rest.slice(0, 2)].filter(Boolean).join(", ");
        li.innerHTML = `${esc(main)}<small>${esc(where)}</small>
          <div class="actions">${it.boundary ? '<button type="button" class="btn btn--secondary btn--sm" data-a="boundary">Użyj granicy jako obszaru</button>' : ""}
          <button type="button" class="btn btn--secondary btn--sm" data-a="bbox">Użyj prostokąta</button></div>`;
        li.addEventListener("click", (ev) => {
          const a = ev.target.closest("button")?.dataset.a;
          const b = it.bbox;
          if (a === "boundary") { setArea(it.boundary, `Granica: ${main}`); openStep(3); }
          else if (a === "bbox") {
            setArea({ type: "Polygon", coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]] }, `Obwiednia: ${main}`);
            openStep(3);
          } else { map.fitBounds([[b[1], b[0]], [b[3], b[2]]]); return; }
          resultsBox.hidden = true;
        });
        resultsBox.appendChild(li);
      });
      resultsBox.hidden = false;
    } catch (err) {
      resultsBox.innerHTML = `<li aria-disabled="true">${esc(err.message)}</li>`;
      resultsBox.hidden = false;
    }
  }
  // wyszukiwanie po Enterze – zasady Nominatim nie pozwalają na autouzupełnianie
  searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); doSearch(); }
    if (e.key === "Escape") resultsBox.hidden = true;
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) resultsBox.hidden = true; });

  // ------------------------------------------------------------------ krok 4 i 5
  $("#spacing").addEventListener("change", (e) => { state.spacing = e.target.value === "auto" ? "auto" : Number(e.target.value); onDataChanged(); refreshGrid(); });
  $("#maxDistance").addEventListener("change", (e) => { state.maxDistance = Number(e.target.value); renderCriteria(); onDataChanged(); });
  $("#method").addEventListener("change", (e) => { state.method = e.target.value; renderMethodHint(); onWeightsChanged(); updateSummaries(); });

  function renderExclusions() {
    const box = $("#excludeList");
    box.innerHTML = (state.options.exclusions || []).map((e) =>
      `<label class="check"><input type="checkbox" value="${e.id}"${state.exclude.includes(e.id) ? " checked" : ""}> ${esc(e.name)}</label>`).join("");
    $$("input", box).forEach((x) => x.addEventListener("change", () => {
      state.exclude = $$("input:checked", box).map((i) => i.value);
      onDataChanged(); refreshGrid();
    }));
  }

  function renderMethods() {
    const sel = $("#method");
    sel.innerHTML = state.options.methods.map((m) => `<option value="${m.id}">${esc(m.name)} – ${esc(m.score_label.toLowerCase())}</option>`).join("");
    sel.value = state.method;
    renderMethodHint();
  }
  function renderMethodHint() {
    const m = state.options.methods.find((x) => x.id === state.method);
    $("#methodHint").textContent = m ? m.description : "";
  }

  // ------------------------------------------------------------------ uruchomienie
  function renderChecklist() {
    const items = [
      ["Kryteria", state.criteria.length > 0 && state.criteria.every((c) => c.scope !== "some" || (c.source.refs || []).length > 0)],
      ["Wagi", weightsValid()],
      ["Obszar", !!state.area],
    ];
    $("#checklist").innerHTML = items.map(([t, ok]) => `<li class="${ok ? "ok" : ""}" title="${ok ? "Gotowe" : "Do uzupełnienia"}">${t}</li>`).join("");
    $("#btnRun").disabled = !items.every(([, ok]) => ok) || !!state.running;
  }

  function dataSignature() {
    return JSON.stringify({
      area: state.area, spacing: state.spacing, maxDistance: state.maxDistance, exclude: state.exclude,
      crit: state.criteria.map((c) => [c.uid, c.source, c.method, c.method === "distance" ? [c.metric, c.maxDistance] : c.radius]),
    });
  }
  const isStale = () => !!state.result && state.resultSig !== dataSignature();

  function buildConfig() {
    return {
      heatflow_scenario: 1,
      name: state.name || "Analiza HeatFlow", goal: state.goal,
      area: state.area, area_label: state.areaLabel,
      grid: { spacing: state.spacing }, max_distance: state.maxDistance, exclude: state.exclude,
      criteria: state.criteria.map((c) => ({
        id: c.uid, name: c.name, source: c.source, method: c.method, direction: c.direction,
        metric: c.metric, radius: c.radius, max_distance: c.maxDistance || state.maxDistance,
      })),
      method: state.method,
      weights: (() => {
        const spec = weightsSpec();
        if (spec.mode === "rancom" && state.weights.rancomInput === "pairs" && !spec.matrix) {
          // niedokończone porównania – zapisujemy udzielone odpowiedzi
          spec.answers = rancomPairs().filter(([i, j]) => answerFor(i, j) !== undefined).map(([i, j]) => [i, j, answerFor(i, j)]);
          delete spec.matrix;
        }
        return spec;
      })(),
      display: { ...state.display },
    };
  }

  $("#btnRun").addEventListener("click", runAnalysis);

  async function runAnalysis() {
    const err = $("#runError");
    err.hidden = true;
    const cfg = buildConfig();
    const sig = dataSignature();
    state.running = true;
    try {
      $("#btnRun").disabled = true;
      setProgress(0.01, "Uruchamianie analizy…");
      const job = await api("/api/analyses", { json: cfg });  // state.job zmieniamy dopiero po sukcesie
      let st = job;
      while (st.status === "queued" || st.status === "running") {
        await new Promise((r) => setTimeout(r, 900));
        st = await api(`/api/analyses/${job.id}`);
        setProgress(st.progress, st.message);
      }
      if (st.status === "error") throw new Error(st.error);
      const full = await api(`/api/analyses/${job.id}?result=1`);
      state.job = full;
      state.result = full.result;
      state.resultSig = sig;
      state.resultCriteria = state.criteria.map((c) => c.uid);
      setProgress(1, `Gotowe – dane pobrane i policzone w ${fmt(full.result.duration_s, 1)} s.`);
      $("#progress").classList.add("done");  // pasek znika, zostaje krótki status
      state.develop = true;                   // nowa mapa „wywołuje się” raz
      await evaluateNow(true);
    } catch (e) {
      err.textContent = state.result ? `${e.message} Poprzedni wynik pozostaje na mapie.` : e.message;
      err.hidden = false;
      $("#progress").hidden = true;
    } finally {
      state.running = false;
      renderChecklist();
    }
  }

  function setProgress(f, msg) {
    $("#progress").hidden = false;
    if (f < 1) $("#progress").classList.remove("done");
    $("#progressBar").style.width = `${Math.round(f * 100)}%`;
    $("#progressMsg").textContent = msg || "";
  }

  const evaluateSoon = debounce(() => evaluateNow(false), 250);
  let evalSeq = 0;

  function invalidateEvaluation(msg) {
    state.evaluation = null;
    state.evalPending = false;
    if (heatOverlay) { map.removeLayer(heatOverlay); heatOverlay = null; }
    topLayer.clearLayers();
    map.closePopup();
    $("#results").classList.add("no-eval");  // legenda, wagi i ranking należały do poprzedniej oceny
    $("#evalNote").textContent = msg;
    $("#evalNote").hidden = !msg;
    setExports(false);
  }

  async function evaluateNow(fit) {
    if (!state.result) return;
    if (isStale()) { renderResults(); return; }
    if (!weightsValid()) {
      const msg = state.weights.mode === "rancom" && state.weights.rancomInput === "pairs"
        ? `Dokończ porównania RANCOM (${rancomAnswered()} z ${rancomPairs().length}), aby zobaczyć mapę.`
        : "Ustaw dodatnią wagę co najmniej jednego kryterium, aby zobaczyć mapę.";
      invalidateEvaluation(msg);
      return;
    }
    const seq = ++evalSeq;
    const jobId = state.job.id;
    try {
      const body = {
        method: state.method, weights: weightsSpec(),
        directions: Object.fromEntries(state.criteria.map((c) => [c.uid, c.direction])),
      };
      const ev = await api(`/api/analyses/${jobId}/evaluate`, { json: body });
      if (seq !== evalSeq || !state.job || state.job.id !== jobId) return; // nowsza ocena w drodze
      state.evaluation = ev;
      state.evalPending = false;
      $("#results").classList.remove("no-eval");
      $("#evalNote").hidden = true;
      $("#runError").hidden = true;
    } catch (e) {
      if (seq !== evalSeq) return;
      invalidateEvaluation(e.message);
      return;
    }
    renderResults(fit);
    if (state.weights.mode === "objective") refreshWeightValues();
  }

  function onWeightsChanged() {
    updateSummaries();
    if (state.result && !isStale()) {
      state.evalPending = true;  // do czasu nowej oceny popupy i eksport są wstrzymane
      map.closePopup();
      setExports(false);
      evaluateSoon();
    }
  }

  function setExports(on) {
    const ev = state.evaluation;
    const ok = on && ev && state.job;
    // bez wyniku linki nie mają adresu (zamiast „#”), więc nie prowadzą donikąd
    const link = (sel, url) => { if (ok) $(sel).setAttribute("href", url); else $(sel).removeAttribute("href"); };
    link("#expGeojson", ok && `/api/analyses/${state.job.id}/export.geojson?eval=${ev.eval_id}`);
    link("#expCsv", ok && `/api/analyses/${state.job.id}/export.csv?eval=${ev.eval_id}`);
    $$(".exports > *").forEach((el) => { el.classList.toggle("disabled", !ok); el.setAttribute("aria-disabled", !ok); });
  }
  function onDataChanged() {
    updateSummaries();
    if (!state.result) return;
    const stale = isStale();
    $("#staleNote").hidden = !stale;
    if (stale) {
      state.evalPending = true;
      map.closePopup();
      setExports(false);
    } else {
      // powrót do ustawień, z którymi policzono dane – oceniamy ponownie (mogły się zmienić wagi/metoda)
      onWeightsChanged();
    }
  }

  // ------------------------------------------------------------------ wynik: siatka i kolory
  function gridIndex(res) {
    if (res._idx) return res._idx;
    const g = res.grid, n = g.nx * g.ny;
    const pos = new Int32Array(n).fill(-1);
    const lon = [], lat = [];
    let k = 0;
    for (let p = 0; p < n; p++) {
      if (g.mask.charCodeAt(p) === 49) {
        pos[p] = k++;
        lon.push(g.lon_min + (p % g.nx) * g.dlon);
        lat.push(g.lat_min + Math.floor(p / g.nx) * g.dlat);
      }
    }
    res._idx = { pos, lon, lat };
    return res._idx;
  }

  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function paletteFn() {
    let stops = PALETTES[state.display.palette].stops.map(hexToRgb);
    if (state.display.reverse) stops = stops.slice().reverse();
    return (t) => {
      t = Math.min(1, Math.max(0, t));
      const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), f = x - i;
      const a = stops[i], b = stops[i + 1];
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
    };
  }
  // 1 = najlepsza ocena w obszarze, 0 = najgorsza – niezależnie od kierunku metody
  function goodness(ev) {
    if (state.display.scale === "rank") {
      // percentyl pozycji = udział punktów ściśle gorszych (remisy dostają tę samą wartość,
      // grupa najgorszych remisów = 0). Porównywalny między metodami, ale ukrywa wielkość różnic.
      // z rang backendu (pełna precyzja): ranga r (metoda „min”) => r−1 lepszych; gorszych = n − r − (remisy − 1)
      const n = ev.ranks.length;
      if (n < 2) return ev.ranks.map(() => 1);
      const ties = new Map();
      ev.ranks.forEach((r) => ties.set(r, (ties.get(r) || 0) + 1));
      return ev.ranks.map((r) => (n - (r - 1) - ties.get(r)) / (n - 1));
    }
    const s = ev.scores, lo = ev.summary.min, hi = ev.summary.max, span = hi - lo || 1;
    return ev.method.lower_is_better ? s.map((v) => (hi - v) / span) : s.map((v) => (v - lo) / span);
  }

  const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  const invMercY = (y) => (Math.atan(Math.exp(y)) * 360) / Math.PI - 90;

  function drawHeatmap() {
    const res = state.result, ev = state.evaluation;
    if (heatOverlay) { map.removeLayer(heatOverlay); heatOverlay = null; }
    if (!res || !ev) return;
    const g = res.grid, idx = gridIndex(res), good = goodness(ev), col = paletteFn();
    // paleta „fade”: jasny koniec skali jest półprzezroczysty, ciemny (najlepszy) – pełny
    const fade = PALETTES[state.display.palette].fade;
    const alpha = (t) => {
      if (!fade) return 255;
      const p = state.display.reverse ? 1 - t : t;
      return Math.round(255 * (0.32 + 0.68 * Math.min(1, Math.max(0, p) / 0.8)));
    };
    const west = g.lon_min - g.dlon / 2, east = g.lon_min + (g.nx - 0.5) * g.dlon;
    const south = g.lat_min - g.dlat / 2, north = g.lat_min + (g.ny - 0.5) * g.dlat;
    const scale = Math.max(1, Math.min(8, Math.floor(1400 / Math.max(g.nx, g.ny))));
    const W = g.nx * scale, H = g.ny * scale;
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    const yTop = mercY(north), yBot = mercY(south);
    const val = (ix, iy) => {
      if (ix < 0 || iy < 0 || ix >= g.nx || iy >= g.ny) return NaN;
      const k = idx.pos[iy * g.nx + ix];
      return k < 0 ? NaN : good[k];
    };
    for (let py = 0; py < H; py++) {
      const lat = invMercY(yTop - ((py + 0.5) / H) * (yTop - yBot));
      const gy = (lat - g.lat_min) / g.dlat;
      for (let px = 0; px < W; px++) {
        const lon = west + ((px + 0.5) / W) * (east - west);
        const gx = (lon - g.lon_min) / g.dlon;
        const nx0 = Math.round(gx), ny0 = Math.round(gy);
        if (Number.isNaN(val(nx0, ny0))) continue;
        let t;
        if (!state.display.smooth) t = val(nx0, ny0);
        else {
          const x0 = Math.floor(gx), y0 = Math.floor(gy), fx = gx - x0, fy = gy - y0;
          const c = [[x0, y0, (1 - fx) * (1 - fy)], [x0 + 1, y0, fx * (1 - fy)], [x0, y0 + 1, (1 - fx) * fy], [x0 + 1, y0 + 1, fx * fy]];
          let sw = 0, sv = 0;
          for (const [cx, cy, w] of c) { const v = val(cx, cy); if (!Number.isNaN(v)) { sw += w; sv += w * v; } }
          t = sw > 0 ? sv / sw : val(nx0, ny0);
        }
        const rgb = col(t), o = (py * W + px) * 4;
        img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = alpha(t);
      }
    }
    // przycięcie do dokładnej granicy obszaru
    const out = document.createElement("canvas");
    out.width = W; out.height = H;
    const octx = out.getContext("2d");
    const tmp = document.createElement("canvas");
    tmp.width = W; tmp.height = H;
    tmp.getContext("2d").putImageData(img, 0, 0);
    octx.save();
    octx.beginPath();
    const toPx = ([lon, lat]) => [((lon - west) / (east - west)) * W, ((yTop - mercY(lat)) / (yTop - yBot)) * H];
    const polys = res.area.type === "Polygon" ? [res.area.coordinates] : res.area.coordinates;
    polys.forEach((rings) => rings.forEach((ring) => ring.forEach((pt, i) => {
      const [x, y] = toPx(pt);
      if (i === 0) octx.moveTo(x, y); else octx.lineTo(x, y);
    })));
    octx.clip("evenodd");
    octx.imageSmoothingEnabled = state.display.smooth;
    octx.drawImage(tmp, 0, 0);
    octx.restore();
    heatCanvas = out;
    heatOverlay = L.imageOverlay(out.toDataURL(), [[south, west], [north, east]], {
      opacity: state.display.opacity, interactive: false, pane: "heatPane",
      className: "heat-overlay" + (state.develop ? " develop" : ""),
    }).addTo(map);
    state.develop = false;
    areaLayer.bringToFront();
  }

  function drawLegend() {
    const ev = state.evaluation;
    if (!ev) return;
    const col = paletteFn();
    const stops = Array.from({ length: 11 }, (_, i) => { const c = col(i / 10); return `rgb(${c.map(Math.round).join(",")}) ${i * 10}%`; });
    $("#legendBar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;
    const better = ev.method.lower_is_better ? ev.summary.min : ev.summary.max;
    const worse = ev.method.lower_is_better ? ev.summary.max : ev.summary.min;
    const d = Math.abs(ev.summary.max - ev.summary.min) < 0.1 ? 4 : 3;
    // oś legendy zawsze biegnie „gorzej → lepiej”; liczby są drugorzędne, kierunek mówi słowo
    const rank = state.display.scale === "rank";
    $(".legend").classList.toggle("rank", rank);  // histogram percentyli jest płaski z definicji – chowamy go
    $("#results").style.setProperty("--best", `rgb(${col(1).map(Math.round).join(",")})`);
    if (rank) {
      $("#legendWorst").innerHTML = `<span class="dirw">← gorzej</span><b>${fmt(Math.max(...ev.ranks))}. miejsce</b>`;
      $("#legendBest").innerHTML = `<b>1. miejsce</b><span class="dirw">lepiej →</span>`;
      $("#legendLabel").innerHTML = `${esc(ev.method.name)}: pozycja w rankingu<small>kolor według percentyla</small>`;
    } else {
      $("#legendWorst").innerHTML = `<span class="dirw">← gorzej</span><b>${fmt(worse, d)}</b>`;
      $("#legendBest").innerHTML = `<b>${fmt(better, d)}</b><span class="dirw">lepiej →</span>`;
      $("#legendLabel").innerHTML = `${esc(ev.method.name)}: ${esc(ev.method.score_label.toLowerCase())}<small>${ev.method.lower_is_better ? "mniej = lepiej" : "więcej = lepiej"}</small>`;
    }
    $$("#scaleMode button").forEach((b) => {
      const on = b.dataset.scale === (state.display.scale || "value");
      b.classList.toggle("active", on); b.setAttribute("aria-checked", on);
    });
    // histogram rozkładu ocen na osi „gorzej → lepiej”
    const cv = $("#legendHist"), ctx = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1, Wc = cv.clientWidth || 300, Hc = 40;
    cv.width = Wc * dpr; cv.height = Hc * dpr; ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, Wc, Hc);
    const bins = 40, h = new Array(bins).fill(0);
    goodness(ev).forEach((t) => { h[Math.max(0, Math.min(bins - 1, Math.floor(t * bins)))]++; });
    const mx = Math.max(...h) || 1, bw = Wc / bins;
    h.forEach((v, i) => {
      const c = col((i + 0.5) / bins), bh = v ? Math.max(1.5, (v / mx) * (Hc - 4)) : 0;
      ctx.fillStyle = `rgb(${c.map(Math.round).join(",")})`;
      ctx.fillRect(i * bw + 0.5, Hc - bh, bw - 1, bh);
    });
    // linia bazowa, żeby jasne słupki nie znikały na bieli
    ctx.fillStyle = "rgba(10,10,10,.18)";
    ctx.fillRect(0, Hc - 1, Wc, 1);
    renderDisplaySummary();
  }

  // ------------------------------------------------------------------ wynik: panel
  function critById(id) { return state.criteria.find((c) => c.uid === id); }
  function resCrit(id) { return state.result.criteria.find((c) => c.id === id); }

  function renderResults(fit = false) {
    const res = state.result, ev = state.evaluation;
    const panel = $("#results");
    if (!res) { panel.hidden = true; return; }
    panel.hidden = false;
    $("#staleNote").hidden = !isStale();
    $("#resTitle").textContent = res.name;
    const g = res.grid;
    const osmDate = res.osm_timestamp ? new Date(res.osm_timestamp).toLocaleDateString("pl-PL") : null;
    const where = state.region ? state.region.name : "";
    $("#resKicker").textContent = ["Mapa preferencji", where, ev ? ev.method.name : null].filter(Boolean).join(" · ");
    $("#resMeta").innerHTML = `${fmt(g.n_points)} pkt co ${fmt(g.spacing)} m · ${fmt(res.area_km2, res.area_km2 < 10 ? 2 : 1)} km²` +
      (osmDate ? `<br>dane OSM z ${osmDate}` : "") + (res.duration_s != null ? ` · ${fmt(res.duration_s, 1)} s` : "");
    excludedLayer.clearLayers();
    const ex = res.excluded;
    if (ex && ex.geometry) excludedLayer.addData(ex.geometry);
    $("#excludedNote").hidden = !(ex && ex.n_points);
    if (ex && ex.n_points) {
      $("#excludedNote").textContent = `Wyłączone z oceny: ${ex.kinds.map((k) => k.name.toLowerCase()).join(", ")} (${fmt(ex.n_points)} pkt)`;
    }
    if (!ev) return;
    map.closePopup();
    drawHeatmap();
    drawLegend();
    // wagi faktycznie użyte
    const uw = $("#usedWeights");
    uw.innerHTML = res.criteria.map((rc) => {
      const c = critById(rc.id), w = ev.weights[rc.id] || 0;
      return w > 0 ? `<span title="${esc(c ? c.name : rc.name)}: ${fmt(w * 100, 1)}%" style="--c:${c ? c.color : "#999"};flex:${w.toFixed(4)} 1 0"></span>` : "";
    }).join("");
    $("#usedWeightsLegend").innerHTML = res.criteria.map((rc) => {
      const c = critById(rc.id), w = ev.weights[rc.id] || 0;
      return `<span class="n${w > 0 ? "" : " off"}" style="--c:${c ? c.color : "#999"}"><span>${esc(rc.name)}</span></span><b class="${w > 0 ? "" : "off"}">${fmt(w * 100, w > 0 && w < 0.1 ? 1 : 0)}%</b>`;
    }).join("");
    const warn = [...res.warnings];
    ev.excluded.forEach((id) => {
      const rc = resCrit(id);
      if (rc && !res.warnings.some((w) => w.includes(`„${rc.name}”`))) warn.push(`„${rc.name}” nie różnicuje lokalizacji w tym obszarze i zostało pominięte w ocenie.`);
    });
    // uwagi zwinięte w jedno „Uwagi do danych (n)”, żeby nie spychały rankingu
    $("#warnings").innerHTML = warn.map((w) => `<p class="alert warn">${esc(w)}</p>`).join("");
    $("#warnCount").textContent = warn.length;
    $("#warnCount").setAttribute("aria-label", `${warn.length} ${warn.length === 1 ? "uwaga" : warn.length < 5 ? "uwagi" : "uwag"}`);
    $("#warnBox").hidden = !warn.length;
    renderTop(ev);
    renderLayerToggles();
    setExports(!isStale());
    if (fit) fitArea();
  }

  function renderTop(ev) {
    const idx = gridIndex(state.result);
    const order = ev.ranks.map((r, i) => [r, i]).sort((a, b) => a[0] - b[0]).slice(0, 10);
    const d = Math.abs(ev.summary.max - ev.summary.min) < 0.1 ? 4 : 3;
    // wynik jest główną liczbą wiersza, współrzędne – drugorzędne
    $("#topList").innerHTML = order.map(([r, i], k) =>
      `<li class="${k === 0 ? "first" : ""}"><button type="button" class="top-row" data-i="${i}" aria-label="Miejsce ${r}: wynik ${fmt(ev.scores[i], d)}, ${fmt(idx.lat[i], 5)}, ${fmt(idx.lon[i], 5)}">
        <span class="rk">${r}</span>
        <span class="xy">${fmt(idx.lat[i], 5)}, ${fmt(idx.lon[i], 5)}</span><span class="sc">${fmt(ev.scores[i], d)}</span></button></li>`).join("");
    $$("#topList .top-row").forEach((b) => b.addEventListener("click", () => {
      const i = Number(b.dataset.i);
      // punkt w środku widocznej części mapy (z lewej strony panelu wyników); popup po zakończeniu ruchu,
      // inaczej automatyczne przesunięcie nie zadziała i popup schowa się pod panelem
      const z = Math.max(map.getZoom(), 15);
      const panelOpen = window.innerWidth > 820 && !$("#results").hidden && !$("#results").classList.contains("collapsed");
      const shift = panelOpen ? ($("#results").offsetWidth + 24) / 2 : 0;
      const pt = map.project([idx.lat[i], idx.lon[i]], z).add([shift, -120]);
      map.once("moveend", () => openPointPopup(i));
      map.setView(map.unproject(pt, z), z);
    }));
    topLayer.clearLayers();
    topMarkers = {};
    if ($("#showTop").checked) {
      // lepsze pozycje rysowane wyżej, gdy znaczniki się nakładają
      order.slice().reverse().forEach(([r, i]) => {
        const first = r === order[0][0];
        const sz = first ? 30 : 24;
        topMarkers[i] = L.marker([idx.lat[i], idx.lon[i]], {
          icon: L.divIcon({ className: "", html: `<div class="top-marker${first ? " first" : ""}">${r}</div>`, iconSize: [sz, sz], iconAnchor: [sz / 2, sz / 2] }),
          title: `Miejsce ${r}`, alt: `Miejsce ${r}`, keyboard: true, zIndexOffset: (1000 - r) * 10, riseOnHover: true,
        }).on("click", () => openPointPopup(i)).addTo(topLayer);
      });
    }
    markSelected(selectedPoint);
  }

  // zaznaczenie jednego miejsca: ten sam stan w wierszu rankingu i na znaczniku
  let topMarkers = {}, selectedPoint = -1;
  function markSelected(i) {
    selectedPoint = i;
    $$("#topList .top-row").forEach((b) => b.setAttribute("aria-current", Number(b.dataset.i) === i ? "true" : "false"));
    Object.entries(topMarkers).forEach(([k, m]) => {
      const el = m.getElement && m.getElement();
      const on = Number(k) === i;
      if (el) el.firstElementChild?.classList.toggle("sel", on);
      m.setZIndexOffset(on ? 100000 : (1000 - Number(el?.textContent || 0)) * 10);
    });
  }
  $("#showTop").addEventListener("change", () => state.evaluation && renderTop(state.evaluation));

  function fmtVal(rc, v) {
    if (rc.method === "count") return `${fmt(v)} szt.`;
    const capped = v >= rc.max_distance - 1e-6;
    return `${capped ? "≥ " : ""}${fmt(v)} m`;
  }

  function renderLayerToggles() {
    const res = state.result, box = $("#layerToggles");
    box.innerHTML = "";
    res.criteria.forEach((rc) => {
      const c = critById(rc.id);
      const pts = res.previews[rc.id] || [];
      const lab = document.createElement("label");
      lab.className = "check";
      const on = !!(featureLayers[rc.id] && map.hasLayer(featureLayers[rc.id]));
      lab.innerHTML = `<input type="checkbox" ${on ? "checked" : ""} ${pts.length ? "" : "disabled"}>
        <span class="sw" style="--c:${c ? c.color : "#777"}"></span>${esc(rc.name)} <small>(${fmt(rc.stats.n_features)}${
          pts.length < rc.stats.n_features ? `, na mapie ${fmt(pts.length)}` : ""})</small>`;
      $("input", lab).addEventListener("change", (e) => {
        if (featureLayers[rc.id]) { map.removeLayer(featureLayers[rc.id]); delete featureLayers[rc.id]; }
        if (e.target.checked) {
          featureLayers[rc.id] = L.layerGroup(pts.map(([lon, lat]) => L.circleMarker([lat, lon], {
            renderer: previewRenderer, radius: 3.5, weight: 1, color: "#fff", fillColor: c ? c.color : "#333", fillOpacity: 0.95,
          }))).addTo(map);
        }
      });
      box.appendChild(lab);
    });
  }

  // ------------------------------------------------------------------ klik na mapie
  function nodeAt(latlng) {
    const res = state.result;
    if (!res) return -1;
    const g = res.grid;
    const ix = Math.round((latlng.lng - g.lon_min) / g.dlon), iy = Math.round((latlng.lat - g.lat_min) / g.dlat);
    if (ix < 0 || iy < 0 || ix >= g.nx || iy >= g.ny) return -1;
    return gridIndex(res).pos[iy * g.nx + ix];
  }

  function openPointPopup(i) {
    const res = state.result, ev = state.evaluation, idx = gridIndex(res);
    if (!ev || state.evalPending || isStale()) return;
    const d = Math.abs(ev.summary.max - ev.summary.min) < 0.1 ? 4 : 3;
    const usedPos = Object.fromEntries(ev.used.map((id, k) => [id, k]));
    const rows = res.criteria.map((rc, j) => {
      const c = critById(rc.id), w = ev.weights[rc.id] || 0;
      const k = usedPos[rc.id];
      let contrib = "";
      if (ev.contributions && k !== undefined) {
        const v = ev.contributions[i][k];
        const frac = w > 0 ? Math.min(1, Math.max(0, v / w)) : 0;  // strata względna: 0 = najlepsza wartość, 1 = najgorsza
        // pełna ścieżka = najgorsza wartość w obszarze; wypełnienie od lewej
        contrib = `<td class="loss"><span class="track" role="img" aria-label="Strata ${fmt(frac * 100, 0)}%" title="${fmt(v, 4)}"><span style="--c:${c ? c.color : "#777"};width:${(frac * 100).toFixed(1)}%"></span></span></td>`;
      } else if (ev.contributions) contrib = "<td></td>";
      return `<tr><td><span class="sw" style="--c:${c ? c.color : "#777"}"></span>${esc(rc.name)}</td><td class="n">${fmtVal(rc, res.values[j][i])}</td><td class="n">${fmt(w * 100, 0)}%</td>${contrib}</tr>`;
    }).join("");
    const html = `<div class="pp">
      <h4>Miejsce ${ev.ranks[i]} <span class="of">z ${fmt(ev.ranks.length)}</span></h4>
      <div class="meta">${esc(ev.method.name)} <b>${fmt(ev.scores[i], d)}</b> (${ev.method.lower_is_better ? "mniej = lepiej" : "więcej = lepiej"})<br>${fmt(idx.lat[i], 5)}, ${fmt(idx.lon[i], 5)}</div>
      <table><thead><tr><th>Kryterium</th><th class="n">Wartość</th><th class="n">Waga</th>${ev.contributions ? '<th class="loss">Strata</th>' : ""}</tr></thead><tbody>${rows}</tbody></table>
      ${ev.contributions ? '<div class="note">Strata: jak daleko punkt jest od najlepszej wartości kryterium w obszarze (pełny pasek = najgorsza wartość). Suma ważonych strat to wynik SPOTIS.</div>' : ""}
    </div>`;
    const wide = window.innerWidth > 820, panelOpen = !$("#results").hidden && !$("#results").classList.contains("collapsed");
    L.popup({ maxWidth: 380, minWidth: 300, autoPanPaddingTopLeft: [20, 70], autoPanPaddingBottomRight: [wide && panelOpen ? $("#results").offsetWidth + 36 : 20, 20] }).setLatLng([idx.lat[i], idx.lon[i]]).setContent(html).openOn(map);
    if (hoverMarker) map.removeLayer(hoverMarker);
    hoverMarker = L.circleMarker([idx.lat[i], idx.lon[i]], { radius: 7, color: "#1f6b3a", weight: 2.5, fill: false }).addTo(map);
    markSelected(i);
  }
  map.on("click", (e) => {
    if (drawHandler && drawHandler.enabled()) return;
    const i = nodeAt(e.latlng);
    if (i >= 0 && state.evaluation && !isStale()) openPointPopup(i);
  });
  map.on("popupclose", () => {
    if (hoverMarker) { map.removeLayer(hoverMarker); hoverMarker = null; }
    markSelected(-1);
  });

  // ------------------------------------------------------------------ ustawienia wyświetlania
  function renderPalettes() {
    $("#palette").innerHTML = Object.entries(PALETTES).map(([k, p]) => `<option value="${k}">${p.name}</option>`).join("");
    $("#palette").value = state.display.palette;
    $("#paletteReverse").checked = state.display.reverse;
    $("#smooth").checked = state.display.smooth;
    $("#opacity").value = state.display.opacity;
    $$("#scaleMode button").forEach((b) => {
      const on = b.dataset.scale === (state.display.scale || "value");
      b.classList.toggle("active", on); b.setAttribute("aria-checked", on);
    });
    renderDisplaySummary();
  }
  function renderDisplaySummary() {
    const p = PALETTES[state.display.palette];
    $("#displaySummary").textContent = p ? p.name.split(":")[0] + (state.display.reverse ? ", odwrócona" : "") : "";
  }
  $("#palette").addEventListener("change", (e) => { state.display.palette = e.target.value; redraw(); });
  $("#paletteReverse").addEventListener("change", (e) => { state.display.reverse = e.target.checked; redraw(); });
  $("#smooth").addEventListener("change", (e) => { state.display.smooth = e.target.checked; redraw(); });
  $$("#scaleMode button").forEach((b) => b.addEventListener("click", () => { state.display.scale = b.dataset.scale; redraw(); renderPalettes(); }));
  $("#opacity").addEventListener("input", (e) => { state.display.opacity = Number(e.target.value); if (heatOverlay) heatOverlay.setOpacity(state.display.opacity); saveDraft(); });
  function redraw() {
    saveDraft(); renderDisplaySummary();
    if (state.evaluation) { drawHeatmap(); drawLegend(); renderTop(state.evaluation); }
  }

  $("#btnCollapse").addEventListener("click", (e) => {
    const p = $("#results");
    p.classList.toggle("collapsed");
    const collapsed = p.classList.contains("collapsed");
    const btn = e.currentTarget;
    btn.setAttribute("aria-label", collapsed ? "Rozwiń panel wyników" : "Zwiń panel wyników");
    btn.setAttribute("aria-expanded", !collapsed);
  });

  $("#expPng").addEventListener("click", () => {
    if (!heatCanvas || !state.evaluation || isStale()) return;
    const a = document.createElement("a");
    a.href = heatCanvas.toDataURL("image/png");
    a.download = `${(state.result.name || "heatflow").replace(/[^\p{L}\p{N}]+/gu, "_")}_warstwa.png`;
    a.click();
  });

  // ------------------------------------------------------------------ scenariusze
  $("#btnSaveScenario").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(buildConfig(), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(state.name || "scenariusz").replace(/[^\p{L}\p{N}]+/gu, "_")}.heatflow.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $("#btnLoadScenario").addEventListener("click", () => $("#scenarioFile").click());
  $("#scenarioFile").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try { applyScenario(JSON.parse(await f.text())); } catch (err) { alert(`Nie udało się wczytać scenariusza: ${err.message}`); }
    e.target.value = "";
  });

  function applyScenario(s, fromDraft = false) {
    if (!s || typeof s !== "object" || !Array.isArray(s.criteria)) throw new Error("To nie jest plik scenariusza HeatFlow.");
    state.name = s.name || ""; state.goal = s.goal || "";
    $("#name").value = state.name; $("#goal").value = state.goal;
    state.criteria = []; state.weights.manual = {}; state.weights.ranking = {};
    state.weights.scoring = {}; state.weights.answers = {}; state.weights.rancomResult = null;
    const w = s.weights || { mode: "equal" };
    s.criteria.forEach((c, i) => addCriterion({
      name: c.name, source: c.source, method: c.method, direction: c.direction, metric: c.metric, radius: c.radius,
      maxDistance: c.max_distance && Number(c.max_distance) !== Number(s.max_distance || 1500) ? Number(c.max_distance) : null,
      catalogId: c.catalog_id || null, scope: c.source && c.source.refs && c.source.refs.length ? "some" : "all",
      weight: w.mode === "manual" && w.values ? w.values[i] : 1,
      rank: w.mode === "rancom" && w.ranking ? w.ranking[i] : i + 1,
    }));
    if (w.mode === "manual") state.weights.mode = "manual";
    else if (w.mode === "rancom") {
      state.weights.mode = "rancom";
      const cs = state.criteria;
      if (Array.isArray(w.matrix) || Array.isArray(w.answers)) {
        state.weights.rancomInput = "pairs";
        if (Array.isArray(w.matrix)) {
          for (let i = 0; i < cs.length; i++) for (let j = i + 1; j < cs.length; j++) {
            const v = w.matrix[i] && w.matrix[i][j];
            if (v === 0 || v === 0.5 || v === 1) setAnswer(i, j, v);
          }
        }
        (w.answers || []).forEach(([i, j, v]) => { if (cs[i] && cs[j] && [0, 0.5, 1].includes(v)) setAnswer(i, j, v); });
      } else if (Array.isArray(w.scoring)) {
        state.weights.rancomInput = "scoring";
        cs.forEach((c, i) => { state.weights.scoring[c.uid] = Number(w.scoring[i]); });
      } else state.weights.rancomInput = "ranking";
    }
    else if (["entropy", "critic", "std"].includes(w.mode)) { state.weights.mode = "objective"; state.weights.objective = w.mode; }
    else state.weights.mode = "equal";
    // wagi ręczne w procentach dla czytelności
    if (state.weights.mode === "manual") {
      const tot = state.criteria.reduce((a, c) => a + (Number(state.weights.manual[c.uid]) || 0), 0) || 1;
      if (tot <= 1.0001) state.criteria.forEach((c) => { state.weights.manual[c.uid] = Math.round((state.weights.manual[c.uid] / tot) * 1000) / 10; });
    }
    state.spacing = s.grid && s.grid.spacing && s.grid.spacing !== "auto" ? Number(s.grid.spacing) : "auto";
    const sp = $("#spacing");
    if (state.spacing !== "auto" && ![...sp.options].some((o) => o.value === String(state.spacing))) sp.add(new Option(String(state.spacing), String(state.spacing)));
    sp.value = String(state.spacing);
    state.maxDistance = Number(s.max_distance) || 1500;
    state.exclude = Array.isArray(s.exclude) ? s.exclude.filter((e) => (state.options.exclusions || []).some((x) => x.id === e)) : ["water"];
    renderExclusions();
    const md = $("#maxDistance");
    if (![...md.options].some((o) => o.value === String(state.maxDistance))) md.add(new Option(`${fmt(state.maxDistance)} m`, String(state.maxDistance)));
    md.value = String(state.maxDistance);
    state.method = s.method && state.options.methods.some((m) => m.id === s.method) ? s.method : "spotis";
    $("#method").value = state.method; renderMethodHint();
    if (s.display) { state.display = { ...state.display, ...s.display }; renderPalettes(); }
    let area = s.area;
    if (area && area.bbox && !area.type) {
      const [w0, s0, e0, n0] = area.bbox;
      area = { type: "Polygon", coordinates: [[[w0, s0], [e0, s0], [e0, n0], [w0, n0], [w0, s0]]] };
    }
    setArea(area || null, s.area_label || (area ? "Obszar ze scenariusza" : ""), !fromDraft);
    renderCriteria(); renderWeights();
    if (!fromDraft) {
      openStep(2);
      // panel od początku: zwinięty krok 1 i otwarty krok 2, a nie miejsce, w którym był przycisk przykładu
      $("#panel").scrollTop = 0;
    }
  }

  function saveDraft() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(buildConfig())); } catch { /* tryb prywatny */ }
  }
  function loadDraft() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) applyScenario(JSON.parse(raw), true);
    } catch { /* uszkodzony szkic – pomijamy */ }
  }

  function setupRegion() {
    const r = state.region;
    if (!r) {
      $("#dataSource").textContent = "Dane: OpenStreetMap przez Overpass API (online).";
      return;
    }
    regionLayer.addData(r.boundary);
    map.fitBounds(regionLayer.getBounds(), { padding: [20, 20] });
    const date = r.osm_timestamp ? new Date(r.osm_timestamp).toLocaleDateString("pl-PL") : "?";
    $("#dataSource").textContent = `Dane: lokalny wyciąg OpenStreetMap dla miasta ${r.name}, stan na ${date}. Analiza nie korzysta z zewnętrznego API.`;
    $("#areaHint").textContent = `Analiza obejmuje miasto ${r.name}. Wybierz całe miasto, znajdź osiedle albo narysuj obszar – zostanie przycięty do granicy miasta.`;
    $("#btnCity").hidden = false;
    $("#btnCity").textContent = `Cały ${r.name}`;
    $("#placeSearch").placeholder = "Szukaj osiedla, ulicy lub obiektu";
  }
  $("#btnCity").addEventListener("click", () => { if (state.region) setArea(state.region.boundary, `Granica miasta ${state.region.name}`); });

  // ------------------------------------------------------------------ start
  async function init() {
    try {
      const [catalog, options] = await Promise.all([api("/api/catalog"), api("/api/options")]);
      state.catalog = catalog; state.options = options;
    } catch (e) {
      $("#runError").textContent = `Nie można połączyć się z serwerem HeatFlow: ${e.message}`;
      $("#runError").hidden = false;
      return;
    }
    try { state.region = await api("/api/region"); } catch { state.region = null; }
    setupRegion();
    renderMethods(); renderPalettes(); renderCatalog(); renderExclusions();
    await loadDatasets();
    loadExamples();
    loadDraft();
    renderCriteria(); renderWeights(); updateSummaries();
  }
  init();
})();
