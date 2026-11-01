// Page logic shared by index.html and paper.html: loads data.json, fills the numbers in the text, draws the figures,
// and switches between interactive plots and the static publication figures (PDF/SVG).
// Each page defines S = { layout(extra), col, cfg } before loading this file.
(function () {
  const { P, fmt } = window.Figs;
  const $ = id => document.getElementById(id);
  const sel1 = q => document.querySelector(q);
  const get = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
  const FMT = {
    f0: x => fmt(x, 0), f1: x => fmt(x, 1), f2: x => fmt(x, 2), f3: x => fmt(x, 3), f4: x => fmt(x, 4),
    nm: x => fmt(x * 1e3, 1), nm0: x => fmt(x * 1e3, 0), pct: x => fmt(100 * x, 1) + "%", pct2: x => fmt(100 * x, 2) + "%",
    e1: x => { const e = Math.floor(Math.log10(Math.abs(x))); return `${(x / 10 ** e).toFixed(1)}×10<sup>${e}</sup>`; },
  };
  const stat = (v, l) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`;

  // Static figures: plot id -> function giving the figure name for the current controls (null keeps the plot)
  const on = id => (sel1(`#${id} .on`) || {}).dataset || {};
  const STATIC = {
    "plot-A1": () => ({ T: "A_barrier_T", err: "A_barrier_err", conv: "A_barrier_conv" })[on("A1-c").c || "T"],
    "plot-A2": () => "A_cylinder",
    "plot-A3": () => "A_purcell_" + (on("A3-c").c || "radial"),
    "plot-B1": () => "B_map_" + (on("B1-c").c || "nanobeam"),
    "plot-B2": () => "B_zoom_bowtie",
    "plot-B3": () => "B_sweep",
    "plot-B4": () => ({ spread: "B_ensemble", linear: "B_linear" })[on("B4-c").c || "spread"],
    "plot-C1": () => "C_coupling",
    "plot-C2": () => "C_fourier",
    "plot-C3": () => "C_eels_" + (on("C3-c").c || "nanobeam"),
    "plot-D1": () => "D_regime",
    "plot-D2": () => "D_dynamics",
    "plot-D3": () => "D_anticrossing",
  };
  const FIGS = ["A_barrier_T", "A_barrier_err", "A_barrier_conv", "A_cylinder", "A_purcell_radial", "A_purcell_azimuthal",
    "B_map_nanobeam", "B_map_bowtie", "B_zoom_bowtie", "B_sweep", "B_ensemble", "B_linear",
    "C_coupling", "C_fourier", "C_eels_nanobeam", "C_eels_bowtie", "D_regime", "D_dynamics", "D_anticrossing"];
  if ($("figlist")) $("figlist").innerHTML = FIGS.map(f => `<a href="figures/${f}.pdf" target="_blank" rel="noopener">${f}</a>`).join("");
  Object.keys(STATIC).forEach(id => {
    if (!$(id)) return;
    const fig = document.createElement("figure"); fig.className = "rootfig"; fig.id = id + "-root";
    fig.innerHTML = `<a target="_blank" rel="noopener"><img alt="figure"></a><figcaption>Publication figure (matplotlib) · click for the PDF</figcaption>`;
    $(id).after(fig);
  });
  let staticMode = false;
  function updateFigures() {
    Object.entries(STATIC).forEach(([id, f]) => {
      const plot = $(id), fig = $(id + "-root"); if (!plot || !fig) return;
      const name = staticMode ? f() : null;
      plot.classList.toggle("has-root", !!name); fig.classList.toggle("has", !!name);
      if (name) { const img = fig.querySelector("img"); if (!img.src.endsWith(name + ".svg")) { img.src = `figures/${name}.svg`; img.alt = name; fig.querySelector("a").href = `figures/${name}.pdf`; } }
    });
    if (!staticMode && window.Plotly) document.querySelectorAll(".plot").forEach(p => { if (p.data) Plotly.Plots.resize(p); });
  }
  function setView(v) {
    staticMode = v === "root"; document.body.classList.toggle("root-mode", staticMode);
    document.querySelectorAll("#view button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
    try { localStorage.setItem("view", v); } catch (e) {}
    updateFigures();
  }
  document.querySelectorAll("#view button").forEach(b => b.onclick = () => setView(b.dataset.v));

  function seg(id, cb) {
    const el = $(id); if (!el) return;
    el.querySelectorAll("button").forEach(b => b.onclick = () => {
      el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      cb(b.dataset.c); if (staticMode) updateFigures();
    });
  }

  fetch("data.json").then(r => r.json()).then(D => {
    document.querySelectorAll("[data-k]").forEach(el => {
      const v = get(D, el.dataset.k);
      if (v === undefined) { el.textContent = "?"; return; }
      el.innerHTML = el.dataset.f ? FMT[el.dataset.f](v) : v;
    });
    if ($("kpis")) $("kpis").innerHTML = (D.S.kpis || []).map(([v, l]) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join("");
    ["A", "B", "C", "D"].forEach(m => { if ($("stats-" + m)) $("stats-" + m).innerHTML = (D.S["stats" + m] || []).map(([v, l]) => stat(v, l)).join(""); });
    ["cmp-table", "design-table", "g-table"].forEach(t => { if ($(t)) $(t).innerHTML += (D.S[t] || []).map(r => "<tr>" + r.map((c, i) => `<td${i ? ' class="num"' : ""}>${c}</td>`).join("") + "</tr>").join(""); });

    const state = { A1: "T", A3: "radial", B1: "nanobeam", B4: "spread", C1: "nanobeam", C3: "nanobeam" };
    const draw = {
      A1: () => P.barrier(S, D, state.A1), A2: () => P.cylinder(S, D), A3: () => P.purcell2d(S, D, state.A3),
      B1: () => P.map(S, D, state.B1), B2: () => P.zoom(S, D), B3: () => P.sweep(S, D), B4: () => P.ensemble(S, D, state.B4),
      C1: () => P.coupling(S, D, state.C1), C2: () => P.fourier(S, D), C3: () => P.eels(S, D, state.C3),
      D1: () => P.regime(S, D), D2: () => P.dynamics(S, D), D3: () => P.anticross(S, D),
    };
    Object.keys(state).forEach(k => seg(`${k}-c`, v => { state[k] = v; draw[k](); }));
    window.redraw = () => Object.values(draw).forEach(f => { try { f(); } catch (e) { console.error(e); } });
    window.redraw();
    let v0 = "live";
    try { v0 = new URLSearchParams(location.search).get("view") || localStorage.getItem("view") || "live"; } catch (e) {}
    setView(v0 === "root" ? "root" : "live");
    if (window.renderMathInElement) renderMathInElement(document.body, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
  });
  window.addEventListener("load", () => {
    if (window.renderMathInElement) renderMathInElement(document.body, { delimiters: [{ left: "$", right: "$", display: false }], throwOnError: false });
  });

  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.05 });
  document.querySelectorAll(".reveal").forEach(el => io.observe(el));

  // Hero (Web page only): a silicon nanobeam with a bowtie at its centre, the cavity field breathing in time,
  // and an electron passing along the beam with its field flattened by the Lorentz contraction.
  const cv = $("wave");
  if (cv) {
    const ctx = cv.getContext("2d"), reduce = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
    const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    function frame(now) {
      const dpr = devicePixelRatio, w = cv.width = cv.clientWidth * dpr, h = cv.height = cv.clientHeight * dpr;
      ctx.clearRect(0, 0, w, h);
      const narrow = cv.clientWidth < 700, x0b = narrow ? 0 : w * .6, cx = narrow ? w * .5 : w * .8, cy = h * (narrow ? .5 : .42);
      const a = Math.min(w, 1400 * dpr) * (narrow ? .07 : .036), W = a * 1.5, t = reduce ? 0.6 : (now - t0) / 1000;
      ctx.save(); ctx.beginPath(); ctx.rect(x0b, 0, w - x0b, h); ctx.clip();
      // field envelope: Gaussian along the beam, standing wave with period 2a, oscillating in time
      const env = x => Math.exp(-(((x - cx) / (3.2 * a)) ** 2));
      ctx.fillStyle = css("--surface-2"); ctx.fillRect(x0b, cy - W / 2, w - x0b, W);
      for (let x = x0b; x < w; x += 2 * dpr) {
        const s = env(x) * Math.cos(Math.PI * (x - cx) / a) * Math.cos(2.2 * t);
        ctx.fillStyle = s > 0 ? css("--accent") : css("--accent-2"); ctx.globalAlpha = Math.min(.75, Math.abs(s) * .8);
        ctx.fillRect(x, cy - W / 2, 2 * dpr, W);
      }
      ctx.globalAlpha = 1; ctx.fillStyle = css("--bg");
      // holes (period a, tapered towards the centre), skipping the centre where the bowtie sits
      for (let i = 1; i < 30; i++) {
        const x0 = i < 5 ? (0.36 + 0.92 * (i - 1)) * a : (4.05 + (i - 5)) * a;
        const r = a * (i < 5 ? 0.24 + 0.02 * i : 0.33);
        [1, -1].forEach(sg => { const x = cx + sg * x0; if (x > -a && x < w + a && i > 1) { ctx.beginPath(); ctx.arc(x, cy, r, 0, 2 * Math.PI); ctx.fill(); } });
      }
      // bowtie: two air triangles pointing at the centre
      [1, -1].forEach(sg => { ctx.beginPath(); ctx.moveTo(cx + sg * .05 * a, cy - .03 * a); ctx.lineTo(cx + sg * .6 * a, cy - .25 * a); ctx.lineTo(cx + sg * .6 * a, cy + .25 * a); ctx.lineTo(cx + sg * .05 * a, cy + .03 * a); ctx.closePath(); ctx.fill(); });
      // electron passing above the beam
      const period = 3.2, ph = reduce ? .45 : (t % period) / period, ex = x0b - 0.1 * (w - x0b) + ph * 1.2 * (w - x0b), ey = cy - W / 2 - .55 * a;
      const g = ctx.createRadialGradient(ex, ey, 0, ex, ey, 1.4 * a);
      g.addColorStop(0, css("--accent-3")); g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.save(); ctx.translate(ex, ey); ctx.scale(.35, 1); ctx.translate(-ex, -ey);
      ctx.globalAlpha = .45; ctx.fillStyle = g; ctx.beginPath(); ctx.arc(ex, ey, 1.4 * a, 0, 2 * Math.PI); ctx.fill(); ctx.restore();
      ctx.globalAlpha = 1; ctx.fillStyle = css("--accent-3"); ctx.beginPath(); ctx.arc(ex, ey, 3.5 * dpr, 0, 2 * Math.PI); ctx.fill();
      ctx.strokeStyle = css("--accent-3"); ctx.globalAlpha = .35; ctx.setLineDash([4 * dpr, 6 * dpr]); ctx.beginPath(); ctx.moveTo(x0b, ey); ctx.lineTo(w, ey); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1; ctx.restore();
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
})();
