"""Module A: quasinormal modes and their expansions.
A1  dielectric barrier in 1D (analytic), as in Kristensen et al., Adv. Opt. Photon. 12, 612 (2020), Figs. 1, 21, 22.
A2  2D FEM QNM solver with PML against the exact QNMs of a dielectric cylinder.
A3  2D Purcell factor near the cylinder: direct FEM solution vs the QNM (pair) approximation."""
import time

import numpy as np
from ngsolve import CF

from common import save
from sqed import barrier1d as B, cylinder as Cy, fem, geometry as G

out = {}

# ---------------------------------------------------------------- A1
w = np.linspace(0.02, 6, 600)
te = B.transmission_exact(w)
t3 = B.transmission_qnm_total(w, [3, 4, 5])
t1 = B.transmission_qnm_total(w, [4])
x = np.linspace(-3, 3, 601)
f4 = B.qnm_field(4, x) / np.sqrt(B.qnm_norm(4))
Ns = np.unique(np.round(np.logspace(0, 3, 16)).astype(int))
conv = []
w0 = np.array([4.0])
t0 = B.transmission_exact(w0)[0]
for N in Ns:
    m = range(-N, N + 1)
    conv.append([2 * N + 1, abs(B.transmission_qnm_total(w0, m)[0] - t0) / abs(t0),
                 abs(B.transmission_qnm(w0, m)[0] - t0) / abs(t0)])
conv = np.array(conv)
sel = conv[:, 0] >= 201
fit_t = np.polyfit(np.log10(conv[sel, 0]), np.log10(conv[sel, 1]), 1)
fit_s = np.polyfit(np.log10(conv[sel, 0]), np.log10(conv[sel, 2]), 1)
e3 = abs(t3 - te) / abs(te)
e1 = abs(t1 - te) / abs(te)
out["barrier"] = dict(
    w=w, T_exact=te, T_qnm3=t3, err3=e3, err1=e1,
    min_err3=float(e3.min()), w_min3=float(w[e3.argmin()]), min_err1=float(e1.min()), w_min1=float(w[e1.argmin()]),
    qnm=[B.qnm_freq(m) for m in range(-7, 8)], x=x, f4=f4,
    conv=conv, fit_total=fit_t, fit_scat=fit_s,
    published=dict(min_err3=0.02, min_err1=0.07, fit_total=[-2.00, 0.56], fit_scat=[-1.00, 0.21]))
print("A1", out["barrier"]["min_err3"], out["barrier"]["min_err1"], fit_t, fit_s)

# ---------------------------------------------------------------- A2
n, R = 3.48, 0.5
guesses = [(0, 3.129 - 0.153j), (2, 2.789 - 0.070j), (4, 4.2464 - 0.0043j), (4, 6.1418 - 0.0754j), (6, 5.6192 - 0.0002j)]
exact = [(m, Cy.qnm(m, n, R, g)) for m, g in guesses]
mesh = G.cylinder(R=R, h_in=0.05, h_out=0.12)
pconv = []
for order in (2, 3, 4, 5):
    t = time.time()
    S = fem.Maxwell2D(mesh, {"si": n**2}, order=order)
    row = dict(order=order, ndof=S.ndof, k=[], err=[])
    for m, k in exact:
        q = S.qnms(k, nev=6)
        row["k"].append(q[0][0])
        row["err"].append(abs(q[0][0] - k) / abs(k))
    row["time"] = time.time() - t
    pconv.append(row)
    print("A2 order", order, S.ndof, ["%.1e" % e for e in row["err"]])
out["cylinder"] = dict(n=n, R=R, modes=[dict(m=m, k=k, Q=k.real / (-2 * k.imag)) for m, k in exact], pconv=pconv,
                       pml=dict(Rd=2.0, Rp=4.0, alpha=2.0))

# ---------------------------------------------------------------- A3
kq = exact[2][1]
src = (0.42, 0.0)
a = 0.004
mesh = G.cylinder(R=R, h_in=0.05, h_out=0.15, src=src, src_r=a / 2)
S = fem.Maxwell2D(mesh, {"si": n**2, "src": n**2}, order=4)
Sref = fem.Maxwell2D(mesh, {"si": n**2, "src": n**2, "air": n**2, "pml": n**2}, order=4)
q = S.qnms(kq, nev=6)
pair = [p for p in q if abs(p[0] - kq) < 1e-5 * abs(kq)]
ks = np.linspace(kq.real - 0.04, kq.real + 0.04, 41)
area = np.pi * (a / 2) ** 2
pur = {}
for lab, pol in (("radial", (1, 0)), ("azimuthal", (0, 1))):
    f = S.rhs(CF(pol), "src") / area
    Fd, Fq = [], []
    for k in ks:
        pr = np.imag(np.conj(f) @ Sref.solve(k, f))
        Fd.append(np.imag(np.conj(f) @ S.solve(k, f)) / pr)
        eq = sum(p[1] * (p[1] @ f) / (p[0] ** 2 - k**2) for p in pair)
        Fq.append(np.imag(np.conj(f) @ eq) / pr)
    Fd, Fq = np.array(Fd), np.array(Fq)
    i = np.argmax(Fd)
    pur[lab] = dict(F_direct=Fd, F_qnm=Fq, peak_direct=Fd[i], peak_qnm=Fq[i], peak_err=abs(Fq[i] - Fd[i]) / Fd[i])
    print("A3", lab, Fd[i], Fq[i])
out["purcell2d"] = dict(k=ks, kq=kq, pair=[p[0] for p in pair], src=src, src_d=a, ndof=S.ndof, **pur)
save("moduleA", out)
