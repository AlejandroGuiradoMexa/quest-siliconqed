// Interactive figures shared by index.html (Web) and paper.html (Paper).
// Each page passes its own style: S.layout(extra) -> Plotly layout, S.col -> named colours, S.cfg -> Plotly config.
// Every number drawn comes from data.json, built by scripts/build_site.py from results/*.json.
(function () {
  const fmt = (x, d = 2) => (x === null || x === undefined || isNaN(x)) ? "–" : Number(x).toFixed(d);
  const re = a => a.map(z => z[0]), im = a => a.map(z => z[1]);
  const line = (x, y, name, color, extra = {}) => Object.assign({ x, y, name, mode: "lines", line: { color, width: 1.8 } }, extra);
  const dash = (x, y, name, color, d = "dash", extra = {}) => line(x, y, name, color, Object.assign({ line: { color, width: 1.6, dash: d } }, extra));
  const dots = (x, y, name, color, symbol = "circle-open", extra = {}) =>
    Object.assign({ x, y, name, mode: "markers", marker: { color, symbol, size: 7, line: { width: 1.3, color } } }, extra);
  const P = {};

  // ---------------------------------------------------------------- A
  P.barrier = (S, D, mode) => {
    const b = D.A.barrier, c = S.col, w = b.w;
    // the review's convention: T = E(L/2)/E_in(-L/2) = t exp(i w L)
    const rot = (z, wi) => [z[0] * Math.cos(wi) - z[1] * Math.sin(wi), z[0] * Math.sin(wi) + z[1] * Math.cos(wi)];
    if (mode === "T") {
      const Te = b.T_exact.map((z, i) => rot(z, w[i])), T3 = b.T_qnm3.map((z, i) => rot(z, w[i]));
      const sel = w.map((x, i) => x > 2.0 ? i : -1).filter(i => i >= 0);
      Plotly.react("plot-A1", [line(w, re(Te), "Re T, exact", c.c2), line(w, im(Te), "Im T, exact", c.c1),
        dash(sel.map(i => w[i]), sel.map(i => T3[i][0]), "Re T, three QNMs", c.c2), dash(sel.map(i => w[i]), sel.map(i => T3[i][1]), "Im T, three QNMs", c.c1)],
        S.layout({ xaxis: { title: { text: "frequency, ωL/c" }, range: [0, 6] }, yaxis: { title: { text: "transmission T" }, range: [-1.05, 1.3] } }), S.cfg);
    } else if (mode === "err") {
      Plotly.react("plot-A1", [line(w, b.err3, "three QNMs (n = 3, 4, 5)", c.c1), line(w, b.err1, "one QNM (n = 4)", c.c4)],
        S.layout({ xaxis: { title: { text: "frequency, ωL/c" }, range: [2, 6] }, yaxis: { title: { text: "relative error of T" }, type: "log", range: [-2.5, 0.5] } }), S.cfg);
    } else {
      const cv = b.conv, x = cv.map(r => Math.log10(r[0])), xx = [0, 3.4];
      const [st, it] = b.published.fit_total, [ss, is] = b.published.fit_scat;
      Plotly.react("plot-A1", [dots(x, cv.map(r => Math.log10(r[1])), "total field", c.c1, "square-open"),
        dots(x, cv.map(r => Math.log10(r[2])), "scattered field", c.c2),
        dash(xx, xx.map(v => st * v + it), "fits of [1], Fig. 22", c.muted, "dot"), dash(xx, xx.map(v => ss * v + is), "", c.muted, "dot", { showlegend: false })],
        S.layout({ xaxis: { title: { text: "log₁₀(2N+1), number of QNMs" }, range: [0, 3.4] }, yaxis: { title: { text: "log₁₀ relative error, ω₀ = 4c/L" } } }), S.cfg);
    }
  };
  P.cylinder = (S, D) => {
    const cy = D.A.cylinder, c = S.col, cols = [c.c1, c.c2, c.c3, c.c4, c.data], sym = ["circle-open", "square-open", "diamond-open", "triangle-up-open", "x"];
    const orders = cy.pconv.map(r => r.order);
    Plotly.react("plot-A2", cy.modes.map((m, j) => Object.assign(line(orders, cy.pconv.map(r => r.err[j]), `m = ${m.m}, Q = ${fmt(m.Q, 0)}`, cols[j]),
      { mode: "lines+markers", marker: { symbol: sym[j], size: 7, color: cols[j] } })),
      S.layout({ xaxis: { title: { text: "polynomial order of the elements" }, dtick: 1 }, yaxis: { title: { text: "|k̃_FEM − k̃_exact| / |k̃|" }, type: "log" } }), S.cfg);
  };
  P.purcell2d = (S, D, pol) => {
    const p = D.A.purcell2d, c = S.col;
    Plotly.react("plot-A3", [line(p.k, p[pol].F_direct, "direct FEM solution", c.data), dash(p.k, p[pol].F_qnm, "QNM pair", c.c2)],
      S.layout({ xaxis: { title: { text: "k (µm⁻¹)" } }, yaxis: { title: { text: "2D Purcell factor" } } }), S.cfg);
  };

  // ---------------------------------------------------------------- B
  function outline(d, xr, color) {
    const shapes = [];
    d.xs.forEach((x, i) => {
      if (d.w !== null && i === 0) return;
      const r = d.rs[i];
      [1, -1].forEach(s => { if (Math.abs(x) - r < xr) shapes.push({ type: "circle", x0: s * x - r, x1: s * x + r, y0: -r, y1: r, line: { color, width: 1 } }); });
    });
    (d.outline || []).forEach(P => shapes.push({ type: "path", path: "M " + P.map(p => p.join(",")).join(" L ") + " Z", line: { color, width: 1 } }));
    [1, -1].forEach(s => shapes.push({ type: "line", x0: -xr, x1: xr, y0: s * d.W / 2, y1: s * d.W / 2, line: { color, width: 1 } }));
    return shapes;
  }
  P.map = (S, D, name) => {
    const d = D.B.designs[name], m = d.map;
    Plotly.react("plot-B1", [{ type: "heatmap", x: m.x, y: m.y, z: m.I, colorscale: "Magma", zmax: name === "nanobeam" ? 1 : 0.15, zmin: 0,
      showscale: false, hovertemplate: "x %{x:.3f} µm<br>y %{y:.3f} µm<br>|E|² %{z:.3f}<extra></extra>" }],
      S.layout({ shapes: outline(d, m.x[m.x.length - 1], "rgba(255,255,255,.75)"), xaxis: { title: { text: "x (µm)" }, range: [m.x[0], m.x[m.x.length - 1]] },
        yaxis: { title: { text: "y (µm)" }, scaleanchor: "x", range: [m.y[0], m.y[m.y.length - 1]] }, showlegend: false }), S.cfg);
  };
  P.zoom = (S, D) => {
    const d = D.B.designs.bowtie, z = d.zoom, k = 1e3;
    const shapes = d.outline.map(P => ({ type: "path", path: "M " + P.map(p => `${p[0] * k},${p[1] * k}`).join(" L ") + " Z", line: { color: "rgba(255,255,255,.8)", width: 1 } }));
    Plotly.react("plot-B2", [{ type: "heatmap", x: z.x.map(v => v * k), y: z.y.map(v => v * k), z: z.I, colorscale: "Magma", showscale: false,
      hovertemplate: "x %{x:.1f} nm<br>y %{y:.1f} nm<br>|E|² %{z:.3f}<extra></extra>" }],
      S.layout({ shapes, xaxis: { title: { text: "x (nm)" } }, yaxis: { title: { text: "y (nm)" }, scaleanchor: "x" }, showlegend: false }), S.cfg);
  };
  P.sweep = (S, D) => {
    const s = D.B.sweep, c = S.col, w = s.rows.map(r => r.w * 1e3);
    const lg = v => Math.log10(v), Ar = s.rows.map(r => r.A / s.nanobeam_A), Qs = s.rows.map(r => r.Q);
    Plotly.react("plot-B3", [Object.assign(line(w, Ar, "mode area A / A_nanobeam", c.c1), { mode: "lines+markers" }),
      Object.assign(dash(w, Qs, "in-plane Q (right axis)", c.c2), { mode: "lines+markers", yaxis: "y2" })],
      S.layout({ xaxis: { title: { text: "bridge width w (nm)" }, type: "log" },
        yaxis: { title: { text: "A / A_nanobeam" }, type: "log", range: [lg(Math.min(...Ar)) - 0.15, lg(Math.max(...Ar)) + 0.35] },
        yaxis2: { title: { text: "in-plane Q" }, type: "log", overlaying: "y", side: "right", showgrid: false, range: [lg(Math.min(...Qs)) - 0.15, lg(Math.max(...Qs)) + 0.35] },
        legend: { x: 0.02, xanchor: "left", y: 0.98, yanchor: "top" }, margin: { l: 62, r: 64, t: 10, b: 46 } }), S.cfg);
  };
  P.ensemble = (S, D, mode) => {
    const e = D.B.ensemble, c = S.col, tr = [];
    if (mode === "spread") {
      [["nanobeam", c.c1], ["bowtie", c.c2]].forEach(([n, col]) => {
        const rr = e.rows[n], sg = rr.map(r => r.sigma * 1e3);
        tr.push(dots(sg, rr.map(r => r.dlam_std * 1e3), `${n}, ensemble`, col));
        tr.push(line(sg, rr.map(r => r.dlam_std_pred * 1e3), `${n}, first order`, col));
      });
      Plotly.react("plot-B4", tr, S.layout({ xaxis: { title: { text: "fabrication error σ (nm)" } }, yaxis: { title: { text: "std of λ (nm)" } } }), S.cfg);
    } else {
      let lo = 0, hi = 0;
      [["nanobeam", c.c1], ["bowtie", c.c2]].forEach(([n, col]) => {
        const r = e.rows[n][e.rows[n].length - 1], l0 = D.B.designs[n].lam;
        const x = r.lam_lin.map(v => (v - l0) * 1e3), y = r.lam.map(v => (v - l0) * 1e3);
        lo = Math.min(lo, ...x, ...y); hi = Math.max(hi, ...x, ...y);
        tr.push(dots(x, y, n, col));
      });
      tr.push(dash([lo, hi], [lo, hi], "y = x", c.muted, "dot"));
      Plotly.react("plot-B4", tr, S.layout({ xaxis: { title: { text: "Δλ, first-order theory (nm)" } }, yaxis: { title: { text: "Δλ, re-simulated (nm)" } } }), S.cfg);
    }
  };

  // ---------------------------------------------------------------- C
  P.coupling = (S, D, name) => {
    const c = S.col, T = Array.from({ length: 157 }, (_, i) => 10 + i * 2.5), cols = { "0.02": c.c1, "0.05": c.c2, "0.10": c.c3 };
    Plotly.react("plot-C1", Object.entries(D.C[name].b).map(([b, r]) => line(T, r.C2, `b = ${fmt(b * 1e3, 0)} nm`, cols[b])),
      S.layout({ xaxis: { title: { text: "electron energy (keV)" }, range: [10, 400] }, yaxis: { title: { text: "|C(v)|² (arb.)" }, type: "log" } }), S.cfg);
  };
  P.fourier = (S, D) => {
    const c = S.col, tr = [];
    [["nanobeam", c.c1], ["bowtie", c.c2]].forEach(([n, col]) => {
      const r = D.C[n].b["0.05"], a = D.C[n].a, mx = Math.max(...r.F2);
      tr.push(line(r.q.map(q => q * a / Math.PI), r.F2.map(v => v / mx), n, col));
    });
    Plotly.react("plot-C2", tr, S.layout({ xaxis: { title: { text: "q a / π (a: mirror period)" }, range: [0, 4] }, yaxis: { title: { text: "spatial spectrum of f_x (norm.)" } },
      shapes: [{ type: "line", x0: 1, x1: 1, yref: "paper", y0: 0, y1: 1, line: { color: c.muted, dash: "dot", width: 1 } }] }), S.cfg);
  };
  P.eels = (S, D, name) => {
    const e = D.C[name].eels, c = S.col, k0 = D.C[name].k, x = e.k.map(k => (k - k0[0]) / (-2 * k0[1]));
    Plotly.react("plot-C3", [line(x, e.direct, "direct FEM (scattered field)", c.data), dash(x, e.qnm, "single QNM", c.c2)],
      S.layout({ xaxis: { title: { text: "(k − Re k̃) / (−2 Im k̃)" } }, yaxis: { title: { text: "energy-loss probability (arb.)" } } }), S.cfg);
  };

  // ---------------------------------------------------------------- D
  P.regime = (S, D) => {
    const m = D.D.map, c = S.col, G = 2 * Math.PI * 1e9, cols = [c.c4, c.c1, c.c2];
    const tr = [line(m.Q, m.boundary_g.map(v => v / G), "4g = κ + γ + 2γ*", c.data), dash(m.Q, m.boundary_g_nodeph.map(v => v / G), "4g = κ + γ", c.muted, "dot")];
    D.D.rows.filter(r => Math.abs(r.QE - m.qe) < 1e-9).forEach((r, i) => tr.push(dash([m.Q[0], m.Q[m.Q.length - 1]], [r.g / G, r.g / G], r.label, cols[i])));
    Plotly.react("plot-D1", tr, S.layout({ xaxis: { title: { text: "cavity Q" }, type: "log" }, yaxis: { title: { text: "g/2π (GHz)" }, type: "log" },
      shapes: [{ type: "line", x0: D.D.kim.Q, x1: D.D.kim.Q, yref: "paper", y0: 0, y1: 1, line: { color: c.muted, width: 1 } }] }), S.cfg);
  };
  P.dynamics = (S, D) => {
    const d = D.D.dynamics, c = S.col, t = d.t.map(v => v * 1e9), cols = [c.c4, c.c1, c.c2], tr = [];
    Object.entries(d.Q).forEach(([q, r], i) => {
      tr.push(line(t, r.Pe, `Q = ${Number(q).toLocaleString("en")}`, cols[i]));
      tr.push(dash(t, r.Pe_nodeph, "", cols[i], "dot", { showlegend: false }));
    });
    tr.push(line(t, t.map(x => Math.exp(-x / (D.D.kim.tau0 * 1e9))), "no cavity", c.muted));
    Plotly.react("plot-D2", tr, S.layout({ xaxis: { title: { text: "time (ns)" } }, yaxis: { title: { text: "P_e(t)" }, range: [0, 1.02] } }), S.cfg);
  };
  P.anticross = (S, D) => {
    const a = D.D.anticrossing, c = S.col;
    const z = a.w.map((_, j) => a.det.map((_, i) => Math.log10(a.S[i][j] + 1e-4)));
    Plotly.react("plot-D3", [{ type: "heatmap", x: a.det, y: a.w, z, colorscale: "Viridis", showscale: false, hovertemplate: "Δ/g %{x:.2f}<br>(ω−ω_c)/g %{y:.2f}<extra></extra>" },
      dash(a.det, a.cm_re.map(v => v[0]), "coupled-mode Re ω", "#ffffff", "dot"), dash(a.det, a.cm_re.map(v => v[1]), "", "#ffffff", "dot", { showlegend: false })],
      S.layout({ xaxis: { title: { text: "detuning (ω_e − ω_c)/g" } }, yaxis: { title: { text: "(ω − ω_c)/g" } }, showlegend: false }), S.cfg);
  };

  window.Figs = { P, fmt };
})();
