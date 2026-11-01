"""Bundle results/*.json into site/data.json and copy the static figures (PDF + SVG) to site/figures/.

The page reads only data.json: every number shown on it comes from the module scripts.
"""
import json
import math
import re
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SITE = ROOT / "site"
NTESTS = sum(1 for f in (ROOT / "tests").glob("test_*.py") for line in f.read_text().splitlines() if line.startswith("def test_"))


def load(name):
    path = RES / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def e1(x):
    e = math.floor(math.log10(abs(x)))
    return f"{x / 10**e:.1f}×10<sup>{e}</sup>"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def summary(d):
    """Numbers quoted in the text, derived from the module results (kept here so the HTML has no typed numbers)."""
    A, B, C, D = (d.get(k) for k in "ABCD")
    s = {"ntests": NTESTS}
    b = A["barrier"]
    conv = b["conv"]
    last = conv[-1]
    st, it = b["published"]["fit_total"]
    ss, is_ = b["published"]["fit_scat"]
    n_last = last[0]
    s["a_conv_n"] = f"{n_last:.0f}"
    s["a_conv_total"] = e1(last[1])
    s["a_conv_total_pub"] = e1(10 ** (st * math.log10(n_last) + it))
    s["a_conv_scat"] = e1(last[2])
    s["a_conv_scat_pub"] = e1(10 ** (ss * math.log10(n_last) + is_))
    s["a_fit_total"] = f"{b['fit_total'][0]:.3f} x + {b['fit_total'][1]:.2f}"
    s["a_fit_scat"] = f"{b['fit_scat'][0]:.3f} x + {b['fit_scat'][1]:.2f}"
    cy = A["cylinder"]
    best = cy["pconv"][-1]
    s["a_cyl_err"] = e1(max(best["err"]))
    s["a_cyl_order"] = best["order"]
    s["a_cyl_ndof"] = f"{best['ndof']:,}"
    p = A["purcell2d"]
    s["a_pur_err"] = pct(max(p["radial"]["peak_err"], p["azimuthal"]["peak_err"]))
    s["statsA"] = [[pct(b["min_err3"]), "three QNMs, minimum error of T (review: ≈ 2%)"],
                   [pct(b["min_err1"]), "one QNM, minimum error (review: ≈ 7%)"],
                   [e1(max(best["err"])), f"cylinder QNMs, FEM vs exact (order {best['order']})"],
                   [pct(max(p["radial"]["peak_err"], p["azimuthal"]["peak_err"])), "2D Purcell peak, QNM pair vs direct"]]
    nb, bt = B["designs"]["nanobeam"], B["designs"]["bowtie"]
    kim = B["kim"]
    s["b_area_ratio"] = f"{nb['A'] / bt['A']:.1f}"
    s["b_v_err"] = pct(abs(nb["V3"] / kim["V"] - 1), 0)
    s["design-table"] = [[lab, f"{x['scale']:.3f}", f"{x['lam'] * 1e3:.1f}", f"{x['Q']:,.0f}", f"{x['A'] * 1e6:,.0f}",
                          f"{x['A_rel']:.3f}", f"{x['V3']:.3f}"]
                         for lab, x in (("nanobeam", nb), ("bowtie, w = 10 nm", bt))]
    pb = [abs(complex(*x["dk_dr_all"]) / complex(*x["dk_dr_fd"]) - 1) for x in (nb, bt)]
    s["b_pert_err"] = pct(max(pb))
    s["b_dw_err"] = pct(abs(complex(*bt["dk_dw"]) / complex(*bt["dk_dw_fd"]) - 1))
    # dlambda/dw in nm per nm: dk/dw (1/um per um) -> dlam = -lam^2/(2pi) dk
    s["b_dlam_dw"] = f"{-bt['lam'] ** 2 / (2 * math.pi) * bt['dk_dw'][0]:.2f}"
    s["b_dlam_dr"] = f"{-nb['lam'] ** 2 / (2 * math.pi) * nb['dk_dr_all'][0]:.2f}"
    mc = B["mesh_check"]
    s["b_mesh_A"] = pct(max(abs(r["A"] / mc[0]["A"] - 1) for r in mc), 2)
    s["b_mesh_Q"] = pct(max(abs(r["Q"] / mc[0]["Q"] - 1) for r in mc), 2)
    s["b_mesh_ndof"] = f"{mc[-1]['ndof']:,}"
    ens = B["ensemble"]["rows"]
    top = {k: v[-1] for k, v in ens.items()}
    s["b_ens_sigma"] = f"{top['nanobeam']['sigma'] * 1e3:.0f}"
    s["b_ens_n"] = f"{top['nanobeam']['n']}"
    s["b_ens"] = {k: dict(std=f"{v['dlam_std'] * 1e3:.2f}", pred=f"{v['dlam_std_pred'] * 1e3:.2f}",
                          mean=f"{v['dlam_mean'] * 1e3:+.2f}", Qmed=f"{v['Q_median']:,.0f}") for k, v in top.items()}
    s["statsB"] = [[f"{nb['V3']:.3f}", f"nanobeam V in (λ/n)³ (FDTD of [2]: {kim['V']})"],
                   [f"{s['b_area_ratio']}×", "smaller mode area with a 10 nm bowtie"],
                   [s["b_pert_err"], "perturbation theory vs re-simulation, dk/dr"],
                   [f"{s['b_ens']['bowtie']['std']} nm", f"std of λ, bowtie, σ = {s['b_ens_sigma']} nm"]]
    s["cmp-table"] = [
        ["1D barrier, three QNMs, minimum error of T", pct(b["min_err3"]), "≈ 2% [1, Fig. 21]"],
        [f"1D barrier, total-field error, {n_last:.0f} QNMs", e1(last[1]), s["a_conv_total_pub"] + " (fit of [1, Fig. 22])"],
        [f"1D barrier, scattered-field error, {n_last:.0f} QNMs", e1(last[2]), s["a_conv_scat_pub"] + " (fit of [1, Fig. 22])"],
        ["Cylinder QNMs, FEM vs exact roots", e1(max(best["err"])), "exact"],
        ["Nanobeam mode volume, (λ/n)³", f"{nb['V3']:.3f}", f"{kim['V']} (FDTD) [2]"],
        ["Purcell factor from Q = 4600 and V", f"{D['rows'][0]['F']:.0f} (V of [2])", f"{kim['F_max']} [2]"],
        ["Hole-radius sensitivity, perturbation vs re-simulation", s["b_pert_err"], "re-simulation"],
    ]
    if C:
        def tpk(n, b="0.05"):
            return C[n]["b"][b]["T_peak"]
        s["c_T_nb"] = f"{tpk('nanobeam'):.0f}"
        s["c_T_bt"] = f"{tpk('bowtie'):.0f}"
        s["c_T_pi_a"] = f"{C['nanobeam']['T_pi_over_a_mirror']:.0f}" if C["nanobeam"]["T_pi_over_a_mirror"] else "–"
        s["c_eels_err"] = {n: pct(C[n]["eels"]["peak_err"], 2) for n in ("nanobeam", "bowtie")}
        T = [10 + 2.5 * i for i in range(157)]
        i300 = T.index(300.0)
        s["c_tail"] = {n: e1(C[n]["b"]["0.05"]["C2"][i300] / max(C[n]["b"]["0.05"]["C2"])) for n in ("nanobeam", "bowtie")}
        s["c_qpeak"] = {n: f"{C[n]['b']['0.05']['q_peak'] * C[n]['a'] / math.pi:.3f}" for n in ("nanobeam", "bowtie")}
        s["statsC"] = [[f"{tpk('nanobeam'):.0f} keV", "best electron energy, nanobeam (b = 50 nm)"],
                       [f"{tpk('bowtie'):.0f} keV", "best electron energy, bowtie"],
                       [s["c_eels_err"]["nanobeam"], "EELS peak, single QNM vs direct (nanobeam)"],
                       [s["c_eels_err"]["bowtie"], "EELS peak, single QNM vs direct (bowtie)"]]
        s["cmp-table"].append(["Electron energy loss at resonance, single QNM vs direct", f"{s['c_eels_err']['nanobeam']} / {s['c_eels_err']['bowtie']}", "direct FEM"])
    if D:
        GHz = 2 * math.pi * 1e9
        rows = [r for r in D["rows"] if abs(r["QE"] - 0.10) < 1e-9]
        s["g-table"] = [[r["label"], f"{r['V']:.3f}", f"{r['g'] / GHz:.2f}", f"{r['kappa'] / GHz:.1f}", f"{r['F']:,.0f}",
                         f"{r['Q_strong']:,.0f}" if r["Q_strong"] else "–"] for r in rows]
        s["d_F_meas"] = f"{D['F_meas_eps1']:.1f}"
        s["d_eps_min"] = pct(D["eps_min"])
        s["d_mu"] = {f"qe{round(float(k) * 100)}": f"{v:.2f}" for k, v in D["mu"].items()}
        s["d_gamma_star_GHz"] = f"{D['gamma_star'] / GHz:.1f}"
        bt_row = next(r for r in rows if r["label"].startswith("bowtie"))
        s["d_Q_strong_bt"] = f"{bt_row['Q_strong']:,.0f}" if bt_row["Q_strong"] else "–"
        ac = D["anticrossing"]
        s["d_split"] = " / ".join(f"{x:+.2f}" for x in ac["peaks0"])
        s["d_rates"] = {q: dict(me=f"{r['rate'] / 1e9:.1f}", purcell=f"{r['purcell_rate'] / 1e9:.1f}") for q, r in D["dynamics"]["Q"].items()}
        s["d_cm"] = " / ".join(f"{x[0]:+.2f}" for x in ac["cm0"])
        s["statsD"] = [[s["d_F_meas"], "F_P from the lifetimes of [2], ε = 1 (paper: ≥ 31)"],
                       [s["d_eps_min"], "smallest quantum efficiency with F_P ≤ 1340"],
                       [s["d_Q_strong_bt"], "Q for strong coupling, bowtie, QE = 10%"],
                       [s["d_gamma_star_GHz"] + " GHz", "γ*/2π from the 0.091 nm linewidth of [2]"]]
        s["cmp-table"].append(["F<sub>P</sub> from the measured lifetimes (ε = 1)", s["d_F_meas"], "≥ 31 [2]"])
    s["kpis"] = [[e1(last[1]), f"QNM expansion error, {n_last:.0f} modes (fit of [1]: {s['a_conv_total_pub']})"],
                 [f"{nb['V3']:.2f} (λ/n)³", f"nanobeam mode volume (FDTD of [2]: {kim['V']})"],
                 [f"{s['b_area_ratio']}×", "smaller mode area with a 10 nm silicon bowtie"],
                 [f"{s.get('c_T_nb', '–')} keV", "electron energy that couples best to the nanobeam mode"]]
    return s


def main():
    data = {}
    for key in "ABCD":
        d = load(f"module{key}")
        if d is not None:
            data[key] = d
    if "B" in data:   # per-hole derivatives and raw ensemble draws are not plotted
        for x in data["B"]["designs"].values():
            x.pop("a_r", None)
            x.pop("a_x", None)
    if "C" in data:
        for x in ("nanobeam", "bowtie"):
            for r in data["C"][x]["b"].values():
                r.pop("fx_x", None)
                r.pop("fx", None)
    data["S"] = summary(data)
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    figs = SITE / "figures"
    if figs.exists():
        shutil.rmtree(figs)
    figs.mkdir()
    n = 0
    for f in sorted((RES / "figures").glob("*.pdf")):
        shutil.copy(f, figs / f.name)
        svg = f.with_suffix(".svg").read_text()
        svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)   # no DTD in the published SVG
        (figs / f.with_suffix(".svg").name).write_text(svg)
        n += 1
    print(f"site/figures: {n} figures (PDF + SVG)")
    print(f"site/data.json: {(SITE / 'data.json').stat().st_size / 1024:.0f} kB, modules: {sorted(data)}")


if __name__ == "__main__":
    main()
