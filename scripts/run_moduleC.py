"""Module C: a fast electron passing the cavity (aloof, in the membrane plane, parallel to the beam).
C1  coupling of the cavity QNM to the electron versus electron energy: |int f_x(x, y_e) exp(-i w x / v) dx|^2
C2  energy-loss spectrum near the resonance: direct scattered-field FEM solution vs the single-QNM approximation
Needs results/moduleB.json (the tuned designs)."""
import time

import numpy as np

from common import load, save
from sqed import cavity as Cv, electron as El, fem

fem.ngs.SetNumThreads(4)
B = load("moduleB")
MB = Cv.membrane()
EPS = MB["n_eff"] ** 2
BOW = dict(hb=0.170, xb=0.210)
out = {}
t0 = time.time()
T = np.linspace(10, 400, 157)
for name in ("nanobeam", "bowtie"):
    d = B["designs"][name]
    w = d["w"]
    S, info = Cv.build(d["scale"], bowtie=None if w is None else dict(w=w, t=w, **BOW))
    k, e, Ey = Cv.cavity_mode(S, 1.278)
    X = info["Xd"]
    res = dict(k=k, b={}, a=d["scale"] * 0.350, a_center=d["scale"] * 0.250)
    # C1: coupling versus energy, three impact parameters (distance from the beam edge)
    for b in (0.02, 0.05, 0.10):
        ye = info["W"] / 2 + b
        xs = np.linspace(-X, X, 6001)
        fx = S.eval(e, xs, np.full_like(xs, ye))[:, 0]
        bet, _ = El.beta_gamma(T)
        q = k.real / bet
        C = np.array([np.trapezoid(fx * np.exp(-1j * qi * xs), xs) for qi in q])
        # spatial spectrum of f_x along the line
        qq = np.linspace(0, 4 * np.pi / res["a"], 800)
        F = np.array([np.trapezoid(fx * np.exp(-1j * qi * xs), xs) for qi in qq])
        i = np.argmax(abs(C))
        res["b"][f"{b:.2f}"] = dict(ye=ye, C2=abs(C) ** 2, T_peak=T[i], q=qq, F2=abs(F) ** 2,
                                    q_peak=qq[np.argmax(abs(F))], fx_x=xs[::10], fx=fx[::10])
        print(name, "b", b, "T_peak %.0f keV" % T[i], flush=True)
    # phase matching estimates from the lattice: q = pi / a  ->  v = w a / pi
    for lab, a in (("mirror", res["a"]), ("center", res["a_center"])):
        beta = k.real * a / np.pi
        res[f"T_pi_over_a_{lab}"] = float(El.T_from_beta(beta)) if beta < 1 else None
    # C2: EELS near the resonance at the coupling peak, b = 50 nm
    b = 0.05
    ye = info["W"] / 2 + b
    Tp = res["b"]["0.05"]["T_peak"]
    gam = -2 * k.imag
    ks = np.linspace(k.real - 6 * gam, k.real + 6 * gam, 49)
    Gd, Gq = [], []
    for kk in ks:
        f = El.scattered_rhs(S, kk, Tp, ye)
        es = S.solve(kk, f)
        Gd.append(El.loss(S, es, kk, Tp, ye, X))
        eq = e * (e @ f) / (k**2 - kk**2)
        Gq.append(El.loss(S, eq, kk, Tp, ye, X))
    Gd, Gq = np.array(Gd), np.array(Gq)
    i = np.argmax(Gd)
    res["eels"] = dict(T=Tp, b=b, k=ks, direct=Gd, qnm=Gq, peak_err=abs(Gq[i] - Gd[i]) / Gd[i],
                       min_direct=Gd.min())
    print(name, "EELS peak direct %.4g qnm %.4g err %.3f min %.3g" % (Gd[i], Gq[i], res["eels"]["peak_err"], Gd.min()),
          f"{time.time() - t0:.0f}s", flush=True)
    out[name] = res
save("moduleC", out)
