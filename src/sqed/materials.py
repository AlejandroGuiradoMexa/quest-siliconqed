"""Silicon refractive index (Green 2008) and the TE0 mode of a silicon membrane (effective index)."""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

DATA = Path(__file__).resolve().parents[2] / "data"

# physical constants (SI)
C0 = 299792458.0
HBAR = 1.054571817e-34
EPS0 = 8.8541878128e-12


def load_green2008():
    """Tabulated n, k of crystalline Si at 300 K (Green 2008, refractiveindex.info, CC0). Wavelength in um."""
    rows = []
    for line in (DATA / "Si_Green-2008.yml").read_text().splitlines():
        p = line.split()
        if len(p) == 3:
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                pass
    return np.array(rows)


def n_si(lam_um):
    d = load_green2008()
    return float(np.interp(lam_um, d[:, 0], d[:, 1]))


def slab_te0(n_core, n_clad, t, lam):
    """Effective index of the fundamental TE mode (E parallel to the slab) of a symmetric slab.

    t and lam in the same unit. Solves kappa tan(kappa t/2) = gamma.
    """
    k0 = 2 * np.pi / lam

    def f(ne):
        kap = k0 * np.sqrt(n_core**2 - ne**2)
        gam = k0 * np.sqrt(ne**2 - n_clad**2)
        return kap * np.tan(kap * t / 2) - gam

    # fundamental mode: kappa t/2 < pi/2
    lo = np.sqrt(max(n_core**2 - (np.pi / (k0 * t)) ** 2, n_clad**2)) + 1e-9
    return brentq(f, lo, n_core - 1e-9)


def slab_profile(n_core, n_clad, t, lam, z):
    """TE0 field profile E(z) (E(0) = 1) and its effective thickness int n^2|E|^2 dz / (n_core^2 |E(0)|^2)."""
    ne = slab_te0(n_core, n_clad, t, lam)
    k0 = 2 * np.pi / lam
    kap = k0 * np.sqrt(n_core**2 - ne**2)
    gam = k0 * np.sqrt(ne**2 - n_clad**2)
    inside = np.abs(z) <= t / 2
    E = np.where(inside, np.cos(kap * z), np.cos(kap * t / 2) * np.exp(-gam * (np.abs(z) - t / 2)))
    # analytic integral of n^2 E^2
    core = n_core**2 * (t / 2 + np.sin(kap * t) / (2 * kap))
    clad = n_clad**2 * np.cos(kap * t / 2) ** 2 / gam
    h_eff = (core + clad) / n_core**2
    return ne, E, h_eff
