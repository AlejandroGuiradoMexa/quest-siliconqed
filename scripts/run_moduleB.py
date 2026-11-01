"""Module B: a silicon nanobeam cavity for the G centre (1278 nm), with and without a central bowtie.
B1  the nanobeam of Kim et al. (Nano Lett. 25, 4347 (2025)) in the 2D effective-index model, tuned to 1278 nm
B2  bowtie bridge-width sweep: wavelength, Q (in-plane), mode area; mesh check
B3  first-order QNM perturbation theory vs re-simulation (hole radii, one hole position, bridge width)
B4  random fabrication errors (hole radii and positions): ensemble vs the linear prediction from B3
Run with OMP_NUM_THREADS=1 (the ensemble uses a process pool)."""
import os
import time
from multiprocessing import Pool

import numpy as np

from common import save
from sqed import cavity as Cv, fem

fem.ngs.SetNumThreads(2)
MB = Cv.membrane()
EPS = MB["n_eff"] ** 2
S0 = 1.278 / 1.4293           # first guess of the lateral scale (2D model of the unscaled design: 1429 nm)
BOW = dict(hb=0.170, xb=0.210)  # trapezoid base height and position (unscaled um), replaces the innermost hole pair


def bowtie(w):
    return None if w is None else dict(w=w, t=w, **BOW)


def solve(scale, w=None, perturb=None, order=3, h_bridge=None, lam_guess=1.278):
    kw = {} if h_bridge is None else dict(h_bridge=h_bridge)
    S, info = Cv.build(scale, bowtie=bowtie(w), perturb=perturb, order=order, **kw)
    k, e, Ey = Cv.cavity_mode(S, lam_guess)
    return S, info, k, e, Ey


def tune(w):
    s = S0
    for _ in range(3):
        S, info, k, e, Ey = solve(s, w)
        lam = 2 * np.pi / k.real
        if abs(lam - 1.278) < 2e-4:
            break
        s *= 1.278 / lam
    return s, S, info, k, e, Ey


def field_map(S, e, xr, yr, nx, ny):
    xs, ys = np.linspace(-xr, xr, nx), np.linspace(-yr, yr, ny)
    X, Y = np.meshgrid(xs, ys)
    E = S.eval(e, X.ravel(), Y.ravel())
    I = (abs(E[:, 0]) ** 2 + abs(E[:, 1]) ** 2).reshape(ny, nx)
    return dict(x=xs, y=ys, I=np.round(I / I.max(), 4))


def ensemble_job(args):
    scale, w, sigma, seed, nh = args
    rng = np.random.default_rng(seed)
    pert = dict(dr=rng.normal(0, sigma, 2 * nh), dx=rng.normal(0, sigma, 2 * nh))
    try:
        _, _, k, _, _ = solve(scale, w, perturb=pert)
        return [sigma, seed, k.real, k.imag] + list(pert["dr"]) + list(pert["dx"])
    except Exception as ex:   # noqa: BLE001
        print("failed", sigma, seed, ex, flush=True)
        return None


def main():
    out = dict(membrane=MB)
    t0 = time.time()
    # ------------------------------------------------------------ B1
    designs = {}
    for name, w in (("nanobeam", None), ("bowtie", 0.010)):
        s, S, info, k, e, Ey = tune(w)
        mn = Cv.mode_numbers(k, Ey, EPS)
        V3 = mn["A"] * MB["h_eff"] / (mn["lam"] / MB["n_si"]) ** 3
        d = dict(scale=s, w=w, ndof=S.ndof, lam=mn["lam"], Q=mn["Q"], k=k, A=mn["A"],
                 A_rel=mn["A"] / (mn["lam"] / (2 * MB["n_eff"])) ** 2, V3=V3,
                 xs=info["xs"], rs=info["rs"], W=info["W"], Xd=info["Xd"], Yd=info["Yd"],
                 map=field_map(S, e, 2.2, 0.6, 221, 61))
        if w is not None:
            d["zoom"] = field_map(S, e, 0.25, 0.25, 101, 101)
            d["outline"] = [Cv.bowtie_outline(w, s * BOW["xb"], s * BOW["hb"], w, sg) for sg in (1, -1)]
        # perturbation theory (B3) for this design
        nh = len(info["xs"])
        # per-hole derivatives (the innermost pair is replaced by the bowtie, so it has none there)
        real = [not (w is not None and i == 0) for i in range(len(info["xs"]))]
        a_r = np.array([Cv.dk_hole(S, e, k, sg * x, r, dr=1.0, eps_si=EPS) if ok else 0j for sg in (1, -1)
                        for x, r, ok in zip(info["xs"], info["rs"], real)])
        a_x = np.array([Cv.dk_hole(S, e, k, sg * x, r, dx=1.0, eps_si=EPS) if ok else 0j for sg in (1, -1)
                        for x, r, ok in zip(info["xs"], info["rs"], real)])
        d["dk_dr_all"] = a_r.sum()
        d["a_r"], d["a_x"] = a_r, a_x
        dd = 0.002
        fd = []
        for sgn in (1, -1):
            pert = dict(dr=np.full(2 * nh, sgn * dd), dx=np.zeros(2 * nh))
            fd.append(solve(s, w, perturb=pert)[2])
        d["dk_dr_fd"] = (fd[0] - fd[1]) / (2 * dd)
        if w is not None:
            d["dk_dw"] = Cv.dk_bowtie_shift(S, e, k, w, s * BOW["xb"], s * BOW["hb"], w, dw=1.0, eps_si=EPS)
            fw = []
            for sgn in (1, -1):
                bw = dict(w=w + sgn * dd, t=w, hb=BOW["hb"], xb=BOW["xb"] + sgn * dd / 2 / s)
                S2, _ = Cv.build(s, bowtie=bw)
                fw.append(Cv.cavity_mode(S2, 1.278)[0])
            d["dk_dw_fd"] = (fw[0] - fw[1]) / (2 * dd)
        designs[name] = d
        print(name, "scale %.4f lam %.4f Q %.0f A %.4g V3 %.3f dk/dr %s fd %s" % (s, d["lam"], d["Q"], d["A"], V3,
              np.round(d["dk_dr_all"], 3), np.round(d["dk_dr_fd"], 3)), f"{time.time() - t0:.0f}s", flush=True)
    out["designs"] = designs
    out["kim"] = dict(V=0.26, Q_meas=4600, lam_cav=1.27723, F_max=1340)

    # ------------------------------------------------------------ B2
    s = designs["nanobeam"]["scale"]
    sweep = []
    for w in (0.100, 0.080, 0.060, 0.040, 0.030, 0.020, 0.015, 0.010, 0.008, 0.006):
        S, info, k, e, Ey = solve(s, w)
        mn = Cv.mode_numbers(k, Ey, EPS)
        sweep.append(dict(w=w, lam=mn["lam"], Q=mn["Q"], A=mn["A"], ndof=S.ndof,
                          A_rel=mn["A"] / (mn["lam"] / (2 * MB["n_eff"])) ** 2))
        print("w %.3f lam %.4f Q %.0f A %.4g" % (w, mn["lam"], mn["Q"], mn["A"]), flush=True)
    out["sweep"] = dict(scale=s, rows=sweep, nanobeam_A=designs["nanobeam"]["A"])
    # mesh / order check at the design point
    chk = []
    sb = designs["bowtie"]["scale"]
    for order, hbr in ((3, None), (3, 0.010 / 6), (4, None), (4, 0.010 / 6)):
        S, info, k, e, Ey = solve(sb, 0.010, order=order, h_bridge=hbr)
        mn = Cv.mode_numbers(k, Ey, EPS)
        chk.append(dict(order=order, h_bridge=hbr or 0.010 / 3, ndof=S.ndof, lam=mn["lam"], Q=mn["Q"], A=mn["A"]))
        print("check", chk[-1], flush=True)
    out["mesh_check"] = chk

    # ------------------------------------------------------------ B4
    sigmas = (0.0005, 0.001, 0.002, 0.003)
    nsamp = 24
    jobs = []
    for name in ("nanobeam", "bowtie"):
        d = designs[name]
        nh = len(d["xs"])
        jobs += [(d["scale"], d["w"], sg, 1000 * i + int(sg * 1e4), nh) for sg in sigmas for i in range(nsamp)]
    with Pool(5) as pool:
        res = pool.map(ensemble_job, jobs)
    ens = {}
    n_per = len(sigmas) * nsamp
    for j, name in enumerate(("nanobeam", "bowtie")):
        d = designs[name]
        rows = [r for r in res[j * n_per:(j + 1) * n_per] if r is not None]
        nh2 = 2 * len(d["xs"])
        k0 = d["k"]
        per = []
        for sg in sigmas:
            R = np.array([r for r in rows if r[0] == sg])
            kk = R[:, 2] + 1j * R[:, 3]
            dr, dx = R[:, 4:4 + nh2], R[:, 4 + nh2:4 + 2 * nh2]
            lin = k0 + dr @ d["a_r"] + dx @ d["a_x"]
            lam, lam_lin = 2 * np.pi / kk.real, 2 * np.pi / lin.real
            Q, Q_lin = kk.real / (-2 * kk.imag), lin.real / (-2 * lin.imag)
            pred_std = sg * np.sqrt(np.sum(d["a_r"].real ** 2) + np.sum(d["a_x"].real ** 2))
            per.append(dict(sigma=sg, n=len(R), lam=lam, Q=Q, lam_lin=lam_lin, Q_lin=Q_lin,
                            dlam_mean=np.mean(lam - d["lam"]), dlam_std=np.std(lam, ddof=1),
                            dlam_std_pred=pred_std * d["lam"] ** 2 / (2 * np.pi),
                            Q_median=np.median(Q), Q_lin_median=np.median(Q_lin)))
            print(name, "sigma", sg, "std lam %.3g pred %.3g  Qmed %.0f lin %.0f" % (
                per[-1]["dlam_std"], per[-1]["dlam_std_pred"], per[-1]["Q_median"], per[-1]["Q_lin_median"]), flush=True)
        ens[name] = per
    out["ensemble"] = dict(sigmas=sigmas, nsamp=nsamp, rows=ens)
    out["time_s"] = time.time() - t0
    save("moduleB", out)


if __name__ == "__main__":
    main()
