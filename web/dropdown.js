/* HeatFlow – lista rozwijana w stylu aplikacji zamiast systemowego menu <select>.
   Natywny <select> zostaje przyciskiem: trzyma wartość, styl, etykiety i zdarzenie „change”, więc app.js
   nic nie musi wiedzieć. Blokujemy tylko systemowe okienko i pokazujemy własną listę (role=listbox).
   Na ekranach dotykowych zostaje natywny wybierak – na telefonie jest wygodniejszy. */
(() => {
  if (matchMedia("(pointer: coarse)").matches) return;

  let pop = null;   // <div role=listbox>
  let sel = null;   // otwarty <select>
  let items = [];   // [{el, opt}]
  let active = -1;
  let typed = "", typedAt = 0;
  let uid = 0;

  const enhanced = (t) => t instanceof HTMLSelectElement && !t.multiple && t.size <= 1;
  const check = '<svg class="dd-check" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  function open(s) {
    close();
    if (s.disabled || !s.options.length) return;
    sel = s;
    if (!s.id) s.id = `dd-sel-${++uid}`;
    pop = document.createElement("div");
    pop.className = "dd-pop";
    pop.id = `${s.id}-list`;
    pop.setAttribute("role", "listbox");
    const label = s.getAttribute("aria-label") || s.closest("label")?.querySelector("span")?.textContent || "";
    if (label) pop.setAttribute("aria-label", label);
    const cs = getComputedStyle(s);
    pop.style.fontFamily = cs.fontFamily;
    pop.style.fontSize = cs.fontSize;

    items = [...s.options].filter((o) => !o.hidden).map((opt, i) => {
      const el = document.createElement("div");
      el.className = "dd-opt";
      el.id = `${pop.id}-${i}`;
      el.setAttribute("role", "option");
      el.setAttribute("aria-selected", String(opt.selected));
      if (opt.disabled) el.setAttribute("aria-disabled", "true");
      el.innerHTML = check;
      const txt = document.createElement("span");
      txt.textContent = opt.label || opt.text;
      el.appendChild(txt);
      pop.appendChild(el);
      return { el, opt };
    });

    pop.addEventListener("mousedown", (e) => e.preventDefault()); // fokus zostaje na <select>
    pop.addEventListener("click", (e) => {
      const i = items.findIndex((it) => it.el.contains(e.target));
      if (i >= 0) choose(i);
    });
    pop.addEventListener("mousemove", (e) => {
      const i = items.findIndex((it) => it.el.contains(e.target));
      if (i >= 0 && i !== active) setActive(i, false);
    });

    document.body.appendChild(pop);
    place();
    s.setAttribute("aria-expanded", "true");
    s.setAttribute("aria-controls", pop.id);
    setActive(Math.max(0, items.findIndex((it) => it.opt.selected)), true);
  }

  function place() {
    const r = sel.getBoundingClientRect();
    const gap = 4, margin = 8;
    const below = innerHeight - r.bottom - gap - margin;
    const above = r.top - gap - margin;
    const want = Math.min(pop.scrollHeight, 320);
    const up = below < want && above > below;
    const maxH = Math.max(120, Math.min(320, up ? above : below));
    pop.style.maxHeight = `${maxH}px`;
    pop.style.minWidth = `${r.width}px`;
    pop.style.maxWidth = `${Math.max(r.width, 360)}px`;
    const w = pop.offsetWidth;
    pop.style.left = `${Math.max(margin, Math.min(r.left, innerWidth - w - margin))}px`;
    pop.style.top = up ? `${r.top - gap - Math.min(pop.offsetHeight, maxH)}px` : `${r.bottom + gap}px`;
    pop.dataset.side = up ? "top" : "bottom";
  }

  function setActive(i, center) {
    if (!items[i]) return;
    items[active]?.el.classList.remove("active");
    active = i;
    const el = items[i].el;
    el.classList.add("active");
    sel.setAttribute("aria-activedescendant", el.id);
    if (center) el.scrollIntoView({ block: "center" });
    else el.scrollIntoView({ block: "nearest" });
  }

  function move(step) {
    for (let i = active + step; i >= 0 && i < items.length; i += step) {
      if (!items[i].opt.disabled) return setActive(i, false);
    }
  }

  function edge(first) {
    const order = first ? items.map((_, i) => i) : items.map((_, i) => items.length - 1 - i);
    const i = order.find((k) => !items[k].opt.disabled);
    if (i != null) setActive(i, false);
  }

  function choose(i) {
    const it = items[i];
    if (!it || it.opt.disabled) return;
    const s = sel;
    const changed = s.value !== it.opt.value || !it.opt.selected;
    close();
    s.focus({ preventScroll: true });
    if (changed) {
      it.opt.selected = true;
      s.dispatchEvent(new Event("input", { bubbles: true }));
      s.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function close() {
    if (!pop) return;
    pop.remove();
    sel.setAttribute("aria-expanded", "false");
    sel.removeAttribute("aria-activedescendant");
    sel.removeAttribute("aria-controls");
    pop = null; sel = null; items = []; active = -1;
  }

  function typeahead(ch) {
    const now = Date.now();
    typed = now - typedAt > 700 ? ch : typed + ch;
    typedAt = now;
    const q = typed.toLocaleLowerCase("pl");
    const n = items.length;
    const start = typed.length === 1 ? active + 1 : active;
    for (let k = 0; k < n; k++) {
      const i = (start + k) % n;
      const it = items[i];
      if (!it.opt.disabled && (it.opt.label || it.opt.text).toLocaleLowerCase("pl").startsWith(q)) return setActive(i, false);
    }
  }

  // Klik w <select>: zamiast systemowego menu – nasza lista (drugi klik zamyka).
  document.addEventListener("mousedown", (e) => {
    const t = e.target;
    if (pop && !pop.contains(t) && t !== sel) close();
    if (!enhanced(t) || e.button !== 0) return;
    e.preventDefault();
    if (t.disabled) return;
    t.focus({ preventScroll: true });
    if (sel === t) close(); else open(t);
  }, true);

  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (!enhanced(t)) return;
    if (!pop || sel !== t) {
      const openKey = e.key === " " || e.key === "Enter" || e.key === "ArrowDown" || e.key === "ArrowUp" || (e.altKey && e.key === "ArrowDown");
      if (openKey && !e.metaKey && !e.ctrlKey) { e.preventDefault(); open(t); }
      return;
    }
    switch (e.key) {
      case "ArrowDown": e.preventDefault(); move(1); break;
      case "ArrowUp": e.preventDefault(); move(-1); break;
      case "Home": case "PageUp": e.preventDefault(); edge(true); break;
      case "End": case "PageDown": e.preventDefault(); edge(false); break;
      case "Enter": case " ": e.preventDefault(); choose(active); break;
      case "Escape": e.preventDefault(); e.stopPropagation(); close(); break;
      case "Tab": close(); break;
      default:
        if (e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey) { e.preventDefault(); typeahead(e.key); }
    }
  }, true);

  // Lista jest przyklejona do pola: przy przewinięciu panelu, zmianie rozmiaru lub utracie fokusu – zamknij.
  document.addEventListener("scroll", (e) => { if (pop && !pop.contains(e.target)) close(); }, true);
  addEventListener("resize", close);
  addEventListener("blur", close);
  document.addEventListener("focusout", (e) => { if (pop && e.target === sel) setTimeout(() => { if (sel && document.activeElement !== sel) close(); }); });
})();
