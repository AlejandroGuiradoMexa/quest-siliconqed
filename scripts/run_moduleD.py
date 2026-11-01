"""Module D: a G centre in the cavity. Dipole moment from the measured lifetime, coupling strength from the mode
volumes of module B, Purcell factor, Lindblad master equation (Jaynes-Cummings with dephasing) and the coupled-mode
(single-QNM) eigenfrequencies. Emitter data: Kim et al., Nano Lett. 25, 4347 (2025) and refs. therein."""
import numpy as np

from common import load, save
from sqed import cavity as Cv, jc
from sqed.materials import C0

B = load("moduleB")
MB = Cv.membrane()
lam = 1.278e-6
n = MB["n_si"]
w0 = 2 * np.pi * C0 / lam
# published emitter and cavity numbers (Kim et al. 2025)
KIM = dict(tau0=5.97e-9, tau_on=0.97e-9, tau_off_res=4.27e-9, DW=0.15, QE=(0.01, 0.10), Q=4600, V=0.26,
           F_max=1340, F_lower=31, linewidth_nm=0.091)
g0 = 1 / KIM["tau0"]
gamma_star = 0.5 * 2 * np.pi * C0 * KIM["linewidth_nm"] * 1e-9 / lam**2      # FWHM = 2 gamma*  (upper bound)
out = dict(kim=KIM, gamma0=g0, gamma_star=gamma_star, w0=w0, n=n)

# D1: dipole moment from the zero-phonon-line radiative rate DW * QE * gamma0
DEBYE = 3.33564e-30
out["mu"] = {f"{qe:.2f}": jc.dipole_from_rate(KIM["DW"] * qe * g0, n, lam) / DEBYE for qe in KIM["QE"]}

# D2: coupling strengths for the mode volumes
vols = {"Kim (FDTD)": KIM["V"], "nanobeam (this 2D model)": B["designs"]["nanobeam"]["V3"],
        "bowtie, w = 10 nm (this 2D model)": B["designs"]["bowtie"]["V3"]}
kappa = w0 / KIM["Q"]
rows = []
for lab, V in vols.items():
    for qe in KIM["QE"]:
        gz = KIM["DW"] * qe * g0
        mu = jc.dipole_from_rate(gz, n, lam)
        g = jc.coupling_g(mu, V * (lam / n) ** 3, n, lam)
        F = jc.purcell_formula(KIM["Q"], V)
        kap_tot = kappa + g0 + 2 * gamma_star
        rate_me = jc.emitter_decay(0.0, g, kappa, g0, gamma_star, np.array([0.0]))[1]
        rate_me0 = jc.emitter_decay(0.0, g, kappa, g0, 0.0, np.array([0.0]))[1]
        rows.append(dict(label=lab, V=V, QE=qe, g=g, kappa=kappa, F=F, F_zpl_rate=F * gz,
                         rate_me=rate_me, rate_me_nodeph=rate_me0, ratio_g_kappa=g / kappa,
                         Q_strong=None))
out["rows"] = rows

# D3: interpretation of the measured lifetimes with Eq. (1) of Kim et al.: F_P = tau0 (1/tau_on - 1/tau_off) / (DW eps)
F_meas = KIM["tau0"] * (1 / KIM["tau_on"] - 1 / KIM["tau_off_res"]) / KIM["DW"]
out["F_meas_eps1"] = F_meas
out["eps_min"] = F_meas / KIM["F_max"]          # smallest quantum efficiency compatible with F_P <= F_max

# D4: regime map: g versus Q for each volume (QE = 10 %), strong-coupling boundary 4 g > kappa + gamma + 2 gamma*
Qs = np.logspace(3, 7, 200)
qe = 0.10
mu = jc.dipole_from_rate(KIM["DW"] * qe * g0, n, lam)
Vs = np.logspace(-3, 0, 200)
gV = jc.coupling_g(mu, Vs * (lam / n) ** 3, n, lam)
out["map"] = dict(Q=Qs, V=Vs, g=gV, kappa=w0 / Qs, boundary_g=(w0 / Qs + g0 + 2 * gamma_star) / 4,
                  boundary_g_nodeph=(w0 / Qs + g0) / 4, qe=qe)
for r in rows:
    gr = jc.coupling_g(jc.dipole_from_rate(KIM["DW"] * r["QE"] * g0, n, lam), r["V"] * (lam / n) ** 3, n, lam)
    # 4 g = w0/Q + g0 + 2 gamma*  ->  Q = w0 / (4 g - g0 - 2 gamma*)
    den = 4 * gr - g0 - 2 * gamma_star
    r["Q_strong"] = w0 / den if den > 0 else None

# D5: master-equation dynamics and spectra for the bowtie design (QE = 10 %)
Vb = B["designs"]["bowtie"]["V3"]
gb = jc.coupling_g(mu, Vb * (lam / n) ** 3, n, lam)
t = np.linspace(0, 3e-9, 400)
dyn = {}
for Q in (4600, 1e5, 1e6):
    kap = w0 / Q
    Pe, rate = jc.emitter_decay(0.0, gb, kap, g0, gamma_star, t)
    Pe0, rate0 = jc.emitter_decay(0.0, gb, kap, g0, 0.0, t)
    dyn[f"{int(Q)}"] = dict(Pe=Pe, rate=rate, Pe_nodeph=Pe0, rate_nodeph=rate0,
                            purcell_rate=g0 + 4 * gb**2 / (kap + g0 + 2 * gamma_star))
out["dynamics"] = dict(t=t, g=gb, V=Vb, Q=dyn)
# anticrossing for a strong-coupling case (Q = 1e6), spectrum vs detuning
Q = 1e6
kap = w0 / Q
dets = np.linspace(-6, 6, 61) * gb
ws = np.linspace(-8, 8, 321) * gb
Smap = np.array([jc.cavity_spectrum(dd, gb, kap, g0, gamma_star, ws) for dd in dets])
cm = np.array([jc.coupled_modes(0.0, kap, dd, g0 + 2 * gamma_star, gb) for dd in dets])
out["anticrossing"] = dict(Q=Q, det=dets / gb, w=ws / gb, S=np.round(Smap / Smap.max(), 5), cm_re=cm.real / gb,
                           cm_im=cm.imag / gb, g=gb, kappa=kap, gamma_star=gamma_star)
# Liouvillian spectrum peaks vs coupled-mode Re(eigenvalues) at zero detuning
S0 = jc.cavity_spectrum(0.0, gb, kap, g0, gamma_star, ws)
pk = ws[1:-1][(S0[1:-1] > S0[:-2]) & (S0[1:-1] > S0[2:])]
out["anticrossing"]["peaks0"] = pk / gb
out["anticrossing"]["cm0"] = jc.coupled_modes(0.0, kap, 0.0, g0 + 2 * gamma_star, gb) / gb
for r in rows:
    print(r["label"], "QE", r["QE"], "g/2pi %.3g GHz" % (r["g"] / 2 / np.pi / 1e9), "F %.0f" % r["F"],
          "Q_strong %s" % (None if r["Q_strong"] is None else "%.3g" % r["Q_strong"]))
print("F_meas(eps=1) %.1f  eps_min %.3f  mu(D) %s" % (F_meas, out["eps_min"], out["mu"]))
save("moduleD", out)
