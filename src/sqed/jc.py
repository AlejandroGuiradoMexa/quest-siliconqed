"""A two-level emitter (colour centre) coupled to one cavity mode: coupling strength from the mode volume, Purcell
factor, Lindblad master equation in the one-excitation space {|g,0>, |e,0>, |g,1>}, and the coupled-mode
(single-QNM) eigenfrequencies. Frequencies in rad/s; time dependence exp(-i w t)."""
import numpy as np

from .materials import C0, HBAR, EPS0


def dipole_from_rate(gamma_rad, n, lam):
    """Transition dipole moment (C m) from a radiative rate gamma_rad (1/s) in a medium of index n:
    gamma = n w^3 mu^2 / (3 pi eps0 hbar c^3)."""
    w = 2 * np.pi * C0 / lam
    return np.sqrt(3 * np.pi * EPS0 * HBAR * C0**3 * gamma_rad / (n * w**3))


def coupling_g(mu, V, n, lam):
    """g (rad/s) for a dipole aligned with the field maximum of a mode with volume V (m^3) in index n:
    hbar g = mu sqrt(hbar w / (2 eps0 n^2 V))."""
    w = 2 * np.pi * C0 / lam
    return mu / HBAR * np.sqrt(HBAR * w / (2 * EPS0 * n**2 * V))


def purcell_formula(Q, V_over_lamn3):
    """F = (3 / 4 pi^2) Q / V, with V in units of (lambda/n)^3."""
    return 3 / (4 * np.pi**2) * Q / V_over_lamn3


# ------------------------------------------------------------------ master equation
# basis 0 = |g,0>, 1 = |e,0>, 2 = |g,1>
def liouvillian(delta, g, kappa, gamma, gamma_dephase):
    """Superoperator L (9 x 9, row-major vec of rho) in the frame of the cavity: H = delta |e><e| + g (|e,0><g,1| + h.c.).
    Collapse operators: sqrt(kappa) a, sqrt(gamma) sigma_-, sqrt(2 gamma*) |e><e| (coherence decay gamma*)."""
    H = np.zeros((3, 3), complex)
    H[1, 1] = delta
    H[1, 2] = H[2, 1] = g
    a = np.zeros((3, 3)); a[0, 2] = 1
    sm = np.zeros((3, 3)); sm[0, 1] = 1
    pe = np.zeros((3, 3)); pe[1, 1] = 1
    cs = [np.sqrt(kappa) * a, np.sqrt(gamma) * sm, np.sqrt(2 * gamma_dephase) * pe]
    I = np.eye(3)
    # vec(A rho B) = (A kron B^T) vec(rho) for row-major vectorisation
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for c in cs:
        cd = c.conj().T
        L += np.kron(c, c.conj()) - 0.5 * np.kron(cd @ c, I) - 0.5 * np.kron(I, (cd @ c).T)
    return L


def emitter_decay(delta, g, kappa, gamma, gamma_dephase, t):
    """Population of |e,0> versus time for the initial state |e,0>, and the effective decay rate 1 / int P_e dt."""
    L = liouvillian(delta, g, kappa, gamma, gamma_dephase)
    rho0 = np.zeros((3, 3), complex); rho0[1, 1] = 1
    w, V = np.linalg.eig(L)
    c = np.linalg.solve(V, rho0.ravel())
    Pe = np.real((V[4, :] * c) @ np.exp(np.outer(w, t)))
    X = np.linalg.solve(L[np.ix_(range(1, 9), range(1, 9))], -rho0.ravel()[1:])   # int rho dt, excluding |g0><g0|
    tau = np.real(X[3])                                                               # element (1,1) -> index 4 -> 3 here
    return Pe, 1 / tau


def cavity_spectrum(delta, g, kappa, gamma, gamma_dephase, w):
    """Spectrum of the photons leaking from the cavity after the emitter starts in |e,0> (quantum regression):
    S(w) = Re Tr[a^dag (-(L + i w))^-1 a X], X = int rho dt. w is measured from the cavity frequency."""
    L = liouvillian(delta, g, kappa, gamma, gamma_dephase)
    rho0 = np.zeros(9, complex); rho0[4] = 1
    # X: solve on the 8-dim space excluding |g0><g0| (L restricted there is invertible)
    idx = list(range(1, 9))
    X = np.zeros(9, complex)
    X[idx] = np.linalg.solve(L[np.ix_(idx, idx)], -rho0[idx])
    X = X.reshape(3, 3)
    a = np.zeros((3, 3)); a[0, 2] = 1
    aX = (a @ X).ravel()
    # a X lives in the coherence block {rho_(g0,e0), rho_(g0,g1)} (vec indices 1, 2), which L maps into itself;
    # there L + i w is invertible for all real w (the steady state |g0><g0| is outside the block)
    blk = [1, 2]
    Lb = L[np.ix_(blk, blk)]
    out = []
    for wi in np.atleast_1d(w):
        Y = np.linalg.solve(Lb + 1j * wi * np.eye(2), -aX[blk])
        out.append(np.real(Y[1]))      # Tr[a^dag Y] = Y_(g0,g1)
    return np.array(out)


def coupled_modes(w_c, kappa, w_e, gamma_e, g):
    """Eigenfrequencies of the non-Hermitian coupled-mode matrix [[w_c - i kappa/2, g], [g, w_e - i gamma_e/2]]
    (the single-QNM description of the coupled system; gamma_e = full width of the emitter coherence)."""
    M = np.array([[w_c - 0.5j * kappa, g], [g, w_e - 0.5j * gamma_e]])
    return np.sort_complex(np.linalg.eigvals(M))
