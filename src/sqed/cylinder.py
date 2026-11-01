"""Exact TE quasinormal modes (in-plane E, Hz polarisation) of a dielectric cylinder in air.
Continuity of Hz and (1/eps) dHz/dr gives  J_m'(n k R) / (n J_m(n k R)) = H_m'(k R) / H_m(k R)."""
import numpy as np
from scipy.optimize import root
from scipy.special import jv, jvp, hankel1, h1vp


def char_eq(k, m, n, R):
    return jvp(m, n * k * R) / (n * jv(m, n * k * R)) - h1vp(m, k * R) / hankel1(m, k * R)


def qnm(m, n, R, k_guess):
    """Complex k (Im k < 0) of the TE_m QNM nearest to k_guess (Newton/hybrid root on Re, Im)."""
    f = lambda z: [char_eq(z[0] + 1j * z[1], m, n, R).real, char_eq(z[0] + 1j * z[1], m, n, R).imag]
    s = root(f, [k_guess.real, k_guess.imag], tol=1e-14)
    k = s.x[0] + 1j * s.x[1]
    assert abs(char_eq(k, m, n, R)) < 1e-9, "root not converged"
    return k
