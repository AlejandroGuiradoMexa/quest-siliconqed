"""Publication figures (matplotlib, Latin Modern fonts, journal style) for modules A-D.

Reads results/*.json and writes PDF (fonts embedded) and SVG files to results/figures/.
    .venv/bin/python scripts/plots.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
RES = BASE / "results"
FIG = RES / "figures"
for f in (BASE / "site" / "fonts").glob("lmroman10-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "Latin Modern Roman", "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "legend.frameon": False, "figure.figsize": (3.4, 2.55),
    "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
C = {"data": "black", "fit": "#1f5fa8", "paper": "#c0392b", "aux": "#2e8b57", "grey": "#888888", "orange": "#d4820a"}


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def cx(a):
    a = np.asarray(a, float)
    return a[..., 0] + 1j * a[..., 1]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)


# ---------------------------------------------------------------- A
def module_a(a):
    b = a["barrier"]
    w, te, t3 = np.array(b["w"]), cx(b["T_exact"]), cx(b["T_qnm3"])
    # same convention as the review: T = E(L/2) / E_in(-L/2) = t exp(i w L)
    Te, T3 = te * np.exp(1j * w), t3 * np.exp(1j * w)
    fig, ax = plt.subplots()
    ax.plot(w, Te.real, color=C["paper"], label="Re $T$, exact")
    ax.plot(w, Te.imag, color=C["fit"], label="Im $T$, exact")
    sel = (w > 2.0) & (w < 6)
    ax.plot(w[sel], T3.real[sel], "--", color=C["paper"], lw=1, label="three QNMs ($n = 3, 4, 5$)")
    ax.plot(w[sel], T3.imag[sel], "--", color=C["fit"], lw=1)
    ax.set(xlabel=r"frequency, $\omega L/c$", ylabel="transmission $T$", xlim=(0, 6), ylim=(-1.05, 1.3))
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    save(fig, "A_barrier_T")

    fig, ax = plt.subplots()
    ax.semilogy(w, b["err3"], color=C["fit"], label="three QNMs")
    ax.semilogy(w, b["err1"], color=C["orange"], label="one QNM ($n = 4$)")
    ax.set(xlabel=r"frequency, $\omega L/c$", ylabel=r"relative error $|T_{\rm QNM} - T|/|T|$", xlim=(2, 6), ylim=(3e-3, 3))
    ax.legend()
    save(fig, "A_barrier_err")

    cv = np.array(b["conv"])
    fig, ax = plt.subplots()
    x = np.log10(cv[:, 0])
    ax.plot(x, np.log10(cv[:, 1]), "s", ms=3.5, mfc="none", color=C["fit"], label="total-field expansion")
    ax.plot(x, np.log10(cv[:, 2]), "o", ms=3.5, mfc="none", color=C["paper"], label="scattered-field expansion")
    xx = np.linspace(0, 3.3, 10)
    for (s, i), col, ls in ((b["published"]["fit_total"], C["fit"], ":"), (b["published"]["fit_scat"], C["paper"], ":")):
        ax.plot(xx, s * xx + i, ls, color=col, lw=1)
    ax.plot([], [], ":", color=C["grey"], label="fits published in [1], Fig. 22")
    ax.set(xlabel=r"number of QNMs, $\log_{10}(2N+1)$", ylabel=r"$\log_{10}$ relative error at $\omega_0 = 4c/L$", xlim=(0, 3.4))
    ax.legend(fontsize=6.5)
    save(fig, "A_barrier_conv")

    cy = a["cylinder"]
    fig, ax = plt.subplots()
    orders = [r["order"] for r in cy["pconv"]]
    marks = "osd^v"
    for j, m in enumerate(cy["modes"]):
        k = cx(m["k"])
        ax.semilogy(orders, [r["err"][j] for r in cy["pconv"]], marker=marks[j], ms=3.5, mfc="none",
                    label=rf"$m = {m['m']}$, $Q = {m['Q']:.0f}$")
    ax.set(xlabel="polynomial order of the elements", ylabel=r"$|\tilde k_{\rm FEM} - \tilde k_{\rm exact}| / |\tilde k|$", xticks=orders)
    ax.minorticks_off()
    ax.legend(fontsize=6.5)
    save(fig, "A_cylinder")

    p = a["purcell2d"]
    k, kq = np.array(p["k"]), cx(p["kq"])
    for lab in ("radial", "azimuthal"):
        fig, ax = plt.subplots()
        ax.plot(k, p[lab]["F_direct"], color=C["data"], label="direct FEM solution")
        ax.plot(k, p[lab]["F_qnm"], "--", color=C["paper"], label="QNM pair")
        ax.set(xlabel=r"$k$ (µm$^{-1}$)", ylabel="2D Purcell factor", xlim=(k[0], k[-1]))
        ax.legend()
        save(fig, f"A_purcell_{lab}")


# ---------------------------------------------------------------- B
def _outline(ax, d, xr):
    for sg in (1, -1):
        for i, (x, r) in enumerate(zip(d["xs"], d["rs"])):
            if d["w"] is not None and i == 0:
                continue
            if abs(x) - r < xr:
                ax.add_patch(Circle((sg * x, 0), r, fill=False, lw=0.5, color="white"))
    if d.get("outline"):
        for P in d["outline"]:
            ax.add_patch(Polygon(np.array(P), closed=True, fill=False, lw=0.5, color="white"))
    ax.axhline(d["W"] / 2, color="white", lw=0.5)
    ax.axhline(-d["W"] / 2, color="white", lw=0.5)


def module_b(b):
    for name in ("nanobeam", "bowtie"):
        d = b["designs"][name]
        m = d["map"]
        fig, ax = plt.subplots(figsize=(3.4, 1.25))
        ax.imshow(np.array(m["I"]), extent=[m["x"][0], m["x"][-1], m["y"][0], m["y"][-1]], origin="lower",
                  cmap="magma", aspect="equal", vmax=1 if name == "nanobeam" else 0.15)
        _outline(ax, d, m["x"][-1])
        ax.set(xlabel=r"$x$ (µm)", ylabel=r"$y$ (µm)", xlim=(m["x"][0], m["x"][-1]), ylim=(m["y"][0], m["y"][-1]))
        ax.minorticks_off()
        save(fig, f"B_map_{name}")
    d = b["designs"]["bowtie"]
    z = d["zoom"]
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.imshow(np.array(z["I"]), extent=[z["x"][0] * 1e3, z["x"][-1] * 1e3, z["y"][0] * 1e3, z["y"][-1] * 1e3],
              origin="lower", cmap="magma", vmax=1)
    for P in d["outline"]:
        ax.add_patch(Polygon(np.array(P) * 1e3, closed=True, fill=False, lw=0.6, color="white"))
    ax.set(xlabel=r"$x$ (nm)", ylabel=r"$y$ (nm)")
    ax.minorticks_off()
    save(fig, "B_zoom_bowtie")

    rows = b["sweep"]["rows"]
    w = np.array([r["w"] for r in rows]) * 1e3
    fig, ax = plt.subplots()
    ax.loglog(w, [r["A"] / b["sweep"]["nanobeam_A"] for r in rows], "o-", ms=3.5, mfc="none", color=C["fit"],
              label=r"mode area $A/A_{\rm nanobeam}$")
    ax2 = ax.twinx()
    ax2.loglog(w, [r["Q"] for r in rows], "s--", ms=3.5, mfc="none", color=C["paper"], label="in-plane $Q$")
    ax.set(xlabel="bridge width $w$ (nm)", ylabel=r"$A / A_{\rm nanobeam}$")
    ax2.set_ylabel("in-plane $Q$ (2D)", color=C["paper"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    save(fig, "B_sweep")

    e = b["ensemble"]
    fig, ax = plt.subplots()
    for name, col in (("nanobeam", C["fit"]), ("bowtie", C["paper"])):
        rr = e["rows"][name]
        sg = np.array([r["sigma"] for r in rr]) * 1e3
        ax.plot(sg, np.array([r["dlam_std"] for r in rr]) * 1e3, "o", ms=4, mfc="none", color=col, label=f"{name}, ensemble")
        ax.plot(sg, np.array([r["dlam_std_pred"] for r in rr]) * 1e3, "-", color=col, lw=1, label=f"{name}, first order")
    ax.set(xlabel=r"fabrication error $\sigma$ (nm)", ylabel=r"spread of $\lambda$, std (nm)")
    ax.legend(fontsize=6.5)
    save(fig, "B_ensemble")

    fig, ax = plt.subplots()
    for name, col in (("nanobeam", C["fit"]), ("bowtie", C["paper"])):
        r = e["rows"][name][-1]
        ax.plot((np.array(r["lam_lin"]) - b["designs"][name]["lam"]) * 1e3, (np.array(r["lam"]) - b["designs"][name]["lam"]) * 1e3,
                "o", ms=3, mfc="none", color=col, label=name)
    lim = ax.get_xlim()
    ax.plot(lim, lim, ":", color=C["grey"], lw=1)
    ax.set(xlabel=r"$\Delta\lambda$, first-order theory (nm)", ylabel=r"$\Delta\lambda$, re-simulated (nm)")
    ax.legend(title=rf"$\sigma$ = {e['sigmas'][-1] * 1e3:.0f} nm", title_fontsize=7)
    save(fig, "B_linear")


# ---------------------------------------------------------------- C
def module_c(c):
    fig, ax = plt.subplots()
    T = np.linspace(10, 400, 157)
    for name, col in (("nanobeam", C["fit"]), ("bowtie", C["paper"])):
        for bkey, ls in (("0.02", "-"), ("0.05", "--"), ("0.10", ":")):
            y = np.array(c[name]["b"][bkey]["C2"])
            ax.semilogy(T, y, ls, color=col, lw=1, label=f"{name}, $b$ = {float(bkey) * 1e3:.0f} nm")
    ax.set(xlabel="electron energy (keV)", ylabel=r"$|C(v)|^2$ (arb.)", xlim=(10, 400))
    ax.legend(fontsize=6, ncol=1)
    save(fig, "C_coupling")

    fig, ax = plt.subplots()
    for name, col in (("nanobeam", C["fit"]), ("bowtie", C["paper"])):
        r = c[name]["b"]["0.05"]
        q = np.array(r["q"]) * c[name]["a"] / np.pi
        F = np.array(r["F2"])
        ax.plot(q, F / F.max(), color=col, label=name)
    ax.axvline(1, color=C["grey"], ls=":", lw=1)
    ax.set(xlabel=r"$q a / \pi$ (mirror period $a$)", ylabel="spatial spectrum of $f_x$ (norm.)", xlim=(0, 4))
    ax.legend()
    save(fig, "C_fourier")

    for name in ("nanobeam", "bowtie"):
        e = c[name]["eels"]
        k = np.array(e["k"])
        k0 = cx(c[name]["k"])
        x = (k - k0.real) / (-2 * k0.imag)
        fig, ax = plt.subplots()
        ax.plot(x, e["direct"], color=C["data"], label="direct FEM (scattered field)")
        ax.plot(x, e["qnm"], "--", color=C["paper"], label="single QNM")
        ax.set(xlabel=r"$(k - {\rm Re}\,\tilde k)/(-2\,{\rm Im}\,\tilde k)$", ylabel="energy-loss probability (arb.)")
        ax.legend(title=f"{e['T']:.0f} keV, $b$ = {e['b'] * 1e3:.0f} nm", title_fontsize=7, loc="upper left", fontsize=6.5)
        save(fig, f"C_eels_{name}")


# ---------------------------------------------------------------- D
def module_d(d):
    m = d["map"]
    Q, kap = np.array(m["Q"]), np.array(m["kappa"])
    fig, ax = plt.subplots()
    GHz = 2 * np.pi * 1e9
    ax.loglog(Q, np.array(m["boundary_g"]) / GHz, color=C["data"], label=r"$4g = \kappa + \gamma + 2\gamma^*$")
    ax.loglog(Q, np.array(m["boundary_g_nodeph"]) / GHz, ":", color=C["grey"], label=r"$4g = \kappa + \gamma$ (no dephasing)")
    cols = [C["orange"], C["fit"], C["paper"]]
    labs = []
    for r in d["rows"]:
        if abs(r["QE"] - m["qe"]) > 1e-9:
            continue
        labs.append(r)
    for r, col in zip(labs, cols):
        ax.axhline(r["g"] / GHz, color=col, lw=1, ls="--", label=r["label"])
    ax.axvline(d["kim"]["Q"], color=C["grey"], lw=0.8)
    ax.set(xlabel="cavity $Q$", ylabel=r"$g/2\pi$ (GHz)", xlim=(Q[0], Q[-1]), ylim=(0.3, 80))
    ax.text(2e6, 20, "strong", fontsize=7)
    ax.text(1.2e3, 0.45, "weak (Purcell)", fontsize=7)
    ax.legend(fontsize=5.6, loc="lower right")
    save(fig, "D_regime")

    dy = d["dynamics"]
    t = np.array(dy["t"]) * 1e9
    fig, ax = plt.subplots()
    for (q, r), col in zip(dy["Q"].items(), cols):
        ax.plot(t, r["Pe"], color=col, label=f"$Q$ = {float(q):,.0f}")
        ax.plot(t, r["Pe_nodeph"], ":", color=col, lw=1)
    ax.plot(t, np.exp(-t / (d["kim"]["tau0"] * 1e9)), color=C["grey"], lw=1, label="no cavity")
    ax.set(xlabel="time (ns)", ylabel=r"$P_e(t)$", xlim=(0, t[-1]), ylim=(0, 1.02))
    ax.legend(fontsize=6.5, title="bowtie; dotted: no dephasing", title_fontsize=6.5)
    save(fig, "D_dynamics")

    ac = d["anticrossing"]
    fig, ax = plt.subplots()
    S = np.array(ac["S"])
    ax.imshow(np.log10(S.T + 1e-4), extent=[ac["det"][0], ac["det"][-1], ac["w"][0], ac["w"][-1]], origin="lower",
              aspect="auto", cmap="viridis")
    cm = np.array(ac["cm_re"])
    ax.plot(ac["det"], cm[:, 0], ":", color="white", lw=0.9)
    ax.plot(ac["det"], cm[:, 1], ":", color="white", lw=0.9)
    ax.set(xlabel=r"detuning $(\omega_e - \omega_c)/g$", ylabel=r"$(\omega - \omega_c)/g$")
    ax.minorticks_off()
    save(fig, "D_anticrossing")


def main():
    for name, fn in (("moduleA", module_a), ("moduleB", module_b), ("moduleC", module_c), ("moduleD", module_d)):
        d = load(name)
        if d:
            fn(d)
    print(f"{len(list(FIG.glob('*.pdf')))} figures in {FIG}")


if __name__ == "__main__":
    main()
