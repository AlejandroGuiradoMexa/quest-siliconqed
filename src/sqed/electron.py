"""A fast electron passing a 2D structure on a straight line y = y_e along x (aloof trajectory).

In the 2D model the electron is a line charge (uniform along z), the standard 2D analogue of an electron beam.
Units: lengths in um, k = w/c in 1/um, eps0 = mu0 = c = 1, unit line charge. Time dependence exp(-i w t).
The field of the moving charge in vacuum (Lorenz gauge, q = w/v, kappa = q/gamma):
    phi = exp(i q x - kappa |y - y_e|) / (2 kappa beta),  Ex = -i q phi / gamma^2,  Ey = kappa sgn(y - y_e) phi.
"""
import numpy as np
import ngsolve as ngs

ME_KEV = 510.99895


def beta_gamma(T_keV):
    g = 1 + np.asarray(T_keV, float) / ME_KEV
    return np.sqrt(1 - 1 / g**2), g


def T_from_beta(beta):
    return ME_KEV * (1 / np.sqrt(1 - np.asarray(beta) ** 2) - 1)


def incident_field_cf(k, T_keV, ye):
    """NGSolve CoefficientFunction of the vacuum field of the electron (valid for y != y_e)."""
    b, g = beta_gamma(T_keV)
    q = k / b
    kap = q / g
    dy = ngs.y - ye
    phi = ngs.exp(1j * q * ngs.x - kap * ngs.IfPos(dy, dy, -dy)) / (2 * kap * b)
    sgn = ngs.IfPos(dy, 1.0, -1.0)
    return ngs.CF((-1j * q * phi / g**2, kap * sgn * phi))


def scattered_rhs(solver, k, T_keV, ye, regions=("si", "src"), eps_b=1.0):
    """Right-hand side k^2 int (eps - eps_b) E0 . v dA of the scattered-field problem."""
    E0 = incident_field_cf(k, T_keV, ye)
    f = 0
    for r in regions:
        de = solver.eps_dict[r] - eps_b
        f = f + solver.rhs(k**2 * de * E0, r)
    return f


def line_integral(solver, e, k, T_keV, ye, X, n=4001):
    """int_{-X}^{X} E_x(x, y_e) exp(-i q x) dx for a solution vector (or GridFunction) e."""
    b, _ = beta_gamma(T_keV)
    q = k / b
    xs = np.linspace(-X, X, n)
    Ex = solver.eval(e, xs, np.full_like(xs, ye))[:, 0]
    return np.trapezoid(Ex * np.exp(-1j * q * xs), xs)


def loss(solver, e, k, T_keV, ye, X, n=4001):
    """Energy-loss spectral density of the line charge (arbitrary units, per unit length along z):
    Gamma ~ -Re int dx exp(-i q x) E_ind,x(x, y_e) / k. Positive for a passive structure."""
    return -np.real(line_integral(solver, e, k, T_keV, ye, X, n)) / k
