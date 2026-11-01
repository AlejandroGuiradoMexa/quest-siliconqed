"""Helpers for the nanobeam cavity: find the cavity QNM, mode area, and first-order QNM perturbation theory
for boundary displacements (holes and bowtie tips)."""
import numpy as np

from . import fem, geometry as G, materials as Mt

LAM_G = 1.278          # G-centre zero-phonon line (um), Kim et al. 2025 measure 1277.25 nm
T_SLAB = 0.220         # SOI device layer (um)


def membrane(lam=LAM_G, t=T_SLAB):
    nsi = Mt.n_si(lam)
    ne, _, heff = Mt.slab_profile(nsi, 1.0, t, lam, np.array([0.0]))
    return dict(n_si=nsi, n_eff=ne, h_eff=heff, lam=lam, t=t)


def build(scale, bowtie=None, n_mirror=6, order=3, perturb=None, **kw):
    mb = membrane()
    mesh, info = G.nanobeam(scale=scale, n_mirror=n_mirror, bowtie=bowtie, perturb=perturb, **kw)
    S = fem.Maxwell2D(mesh, {"si": mb["n_eff"] ** 2, "src": mb["n_eff"] ** 2}, order=order)
    return S, info


def cavity_mode(S, lam_guess, nev=10, Qmin=300):
    """The QNM with the largest |E_y| at the centre among the modes near lam_guess (normalised, e^T M e = 1)."""
    best = None
    for k, e, fr in S.qnms(2 * np.pi / lam_guess - 1e-4j, nev=nev, pml_max=0.05):
        Q = k.real / (-2 * k.imag)
        Ey = S.eval(e, 0.0, 0.0)[0, 1]
        if Q > Qmin and (best is None or abs(Ey) > abs(best[2])):
            best = (k, e, Ey)
    return best


def mode_numbers(k, Ey0, eps0):
    """Wavelength, Q, 2D mode area A = 1/Re(eps f_y^2) (um^2) and its value in (lam/2n)^2."""
    lam = 2 * np.pi / k.real
    A = 1 / np.real(eps0 * Ey0**2)
    return dict(k=k, lam=lam, Q=k.real / (-2 * k.imag), A=A, AQ=1 / (eps0 * Ey0**2))


def boundary_dk(S, e, k, pts, n_in2out, h, dl, eps_in, eps_out, dist=4e-4):
    """First-order QNM perturbation theory for a moving dielectric boundary (normalised mode, e^T M e = 1):
        delta k = -(k/2) int h [ (eps_out - eps_in) E_par.E_par - (1/eps_out - 1/eps_in) D_perp.D_perp ] dl,
    where h > 0 means the 'in' medium is replaced by the 'out' medium over a thickness h (unconjugated products,
    as appropriate for QNMs). Fields are sampled at a distance `dist` inside the 'in' medium."""
    x = pts[:, 0] - dist * n_in2out[:, 0]
    y = pts[:, 1] - dist * n_in2out[:, 1]
    E = S.eval(e, x, y)
    t = np.stack([-n_in2out[:, 1], n_in2out[:, 0]], 1)
    Epar = E[:, 0] * t[:, 0] + E[:, 1] * t[:, 1]
    Dperp = eps_in * (E[:, 0] * n_in2out[:, 0] + E[:, 1] * n_in2out[:, 1])
    I = np.sum(h * dl * ((eps_out - eps_in) * Epar**2 - (1 / eps_out - 1 / eps_in) * Dperp**2))
    return -0.5 * k * I


def dk_hole(S, e, k, xc, r, dr=0.0, dx=0.0, eps_si=None, npts=720):
    """Change of k when the hole (centre (xc, 0), radius r) grows by dr and moves by dx along x."""
    phi = (np.arange(npts) + 0.5) * 2 * np.pi / npts
    radial = np.stack([np.cos(phi), np.sin(phi)], 1)
    pts = np.stack([xc + r * np.cos(phi), r * np.sin(phi)], 1)
    h = dr + dx * np.cos(phi)                  # silicon replaced by air
    return boundary_dk(S, e, k, pts, -radial, h, r * 2 * np.pi / npts, eps_si, 1.0)


def bowtie_outline(w, xb, hb, t, sgn):
    return np.array([(sgn * w / 2, -t / 2), (sgn * xb, -hb / 2), (sgn * xb, hb / 2), (sgn * w / 2, t / 2)])


def dk_bowtie_shift(S, e, k, w, xb, hb, t, dw=1.0, eps_si=None, npts=400):
    """Change of k when both air trapezoids move rigidly outward by dw/2 (the bridge widens by dw)."""
    tot = 0
    for sgn in (1, -1):
        P = bowtie_outline(w, xb, hb, t, sgn)
        for i in range(4):
            a, b = P[i], P[(i + 1) % 4]
            L = np.linalg.norm(b - a)
            n = int(max(20, npts * L / 0.4))
            s = (np.arange(n) + 0.5) / n
            pts = a + np.outer(s, b - a)
            tvec = (b - a) / L
            nrm = np.array([tvec[1], -tvec[0]])
            c = P.mean(0)                       # make the normal point out of the air trapezoid
            if np.dot(nrm, (a + b) / 2 - c) < 0:
                nrm = -nrm
            h = (dw / 2) * np.dot(nrm, [sgn, 0.0])   # >0: boundary moves into silicon (air replaces silicon)
            # sampled on the silicon side: 'in' = silicon, normal from silicon into air = -nrm
            tot += boundary_dk(S, e, k, pts, np.tile(-nrm, (n, 1)), h, L / n, eps_si, 1.0)
    return tot
