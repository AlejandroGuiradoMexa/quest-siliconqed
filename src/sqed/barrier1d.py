"""Quasinormal modes of a dielectric barrier in one dimension (Kristensen et al., Adv. Opt. Photon. 12, 612 (2020),
Sec. 1.3.1, 2.1, 3.2.3, 5.1.3). Units: L = c = 1. Time dependence exp(-i w t)."""
import numpy as np

NR, NB, L = np.pi, 1.0, 1.0


def qnm_freq(m, nr=NR, nb=NB):
    """Eq. (5): complex QNM frequency w_m L/c."""
    return (2 * np.pi * m + 1j * np.log((nr - nb) ** 2 / (nr + nb) ** 2)) / (2 * nr) / L


def qnm_field(m, x, nr=NR, nb=NB):
    """Eq. (2)-(3): unnormalised electric-field QNM."""
    k = qnm_freq(m, nr, nb)
    A = np.exp(-1j * (nr + nb) * k * L / 2) + np.exp(1j * (nr - nb) * k * L / 2 + 1j * m * np.pi)
    x = np.asarray(x, float)
    return np.where(x < -L / 2, A * np.exp(-1j * nb * k * x),
                    np.where(x > L / 2, A * np.exp(1j * nb * k * x + 1j * m * np.pi),
                             np.exp(1j * nr * k * x) + np.exp(-1j * nr * k * x + 1j * m * np.pi)))


def qnm_norm(m, nr=NR):
    """Eq. (47): <<F_m|F_m>> = (-1)^m 2 nR^2 L for the field of Eq. (2)."""
    return (-1) ** m * 2 * nr**2 * L


def transmission_exact(w, nr=NR, nb=NB):
    """Amplitude transmission t of the barrier (plane wave from the left, t defined by E = t exp(i nb w x) for x > L/2)."""
    k1, k2 = nb * w, nr * w
    r12 = (k1 - k2) / (k1 + k2)
    t12, t21 = 2 * k1 / (k1 + k2), 2 * k2 / (k1 + k2)
    ph = np.exp(1j * k2 * L)
    t = t12 * t21 * ph / (1 - r12**2 * ph**2)
    return t * np.exp(-1j * k1 * L)


def _field_overlap(m, w, nr=NR, nb=NB):
    """int_{-L/2}^{L/2} f_m(x) exp(i nb w x) dx, analytic."""
    k = qnm_freq(m, nr, nb)
    out = 0
    for s, ph in ((1, 1), (-1, np.exp(1j * m * np.pi))):
        q = s * nr * k + nb * w
        out += ph * (2 * np.sin(q * L / 2) / q if abs(q) > 1e-14 else L)
    return out


def transmission_qnm(w, modes, nr=NR, nb=NB):
    """QNM approximation to t via the scattered field, Eqs. (112), (113), (117):
    E_scat(x) = sum_m b_m/(w - w_m) f_m(x),  b_m = -(w/2) int f_m (eps - eps_B) E_in dx / <<F_m|F_m>>."""
    w = np.atleast_1d(np.asarray(w, float))
    Es = np.zeros_like(w, complex)
    for m in modes:
        wm = qnm_freq(m, nr, nb)
        fL2 = qnm_field(m, L / 2, nr, nb)
        b = np.array([-(wi / 2) * (nr**2 - nb**2) * _field_overlap(m, wi, nr, nb) for wi in w]) / qnm_norm(m, nr)
        Es += b / (w - wm) * fL2
    Etot = np.exp(1j * nb * w * L / 2) + Es
    return Etot * np.exp(-1j * nb * w * L / 2)


def modes_around(n0, N):
    """The 2N+1 QNM indices n0-N..n0+N (symmetric around the frequency of interest)."""
    return list(range(n0 - N, n0 + N + 1))


def green_exact(x0, w, nr=NR, nb=NB):
    """G(x0, x0, w) solving (d^2/dx^2 + n^2 w^2) G = -delta, outgoing, for |x0| < L/2."""
    k1, k2 = nb * w, nr * w
    # inside solutions matching outgoing waves at each side
    def inside(side):
        # u = a e^{i k2 x} + b e^{-i k2 x}, matched to e^{side i k1 x} at x = side L/2
        xs = side * L / 2
        val, der = np.exp(1j * side * k1 * xs), 1j * side * k1 * np.exp(1j * side * k1 * xs)
        a = (val * 1j * k2 * np.exp(-1j * k2 * xs) * -1 + der * np.exp(-1j * k2 * xs)) / (2j * k2) * -1
        # solve [e^{ik2 xs}, e^{-ik2 xs}; ik2 e^{ik2 xs}, -ik2 e^{-ik2 xs}] [a, b] = [val, der]
        M = np.array([[np.exp(1j * k2 * xs), np.exp(-1j * k2 * xs)], [1j * k2 * np.exp(1j * k2 * xs), -1j * k2 * np.exp(-1j * k2 * xs)]])
        a, b = np.linalg.solve(M, [val, der])
        u = a * np.exp(1j * k2 * x0) + b * np.exp(-1j * k2 * x0)
        du = 1j * k2 * (a * np.exp(1j * k2 * x0) - b * np.exp(-1j * k2 * x0))
        return u, du
    uL, duL = inside(-1)
    uR, duR = inside(+1)
    W = uL * duR - duL * uR
    return -uL * uR / W


def green_qnm(x0, w, modes, nr=NR, nb=NB):
    """Eq. (87): G = (c^2/2w) sum_m f_m(x0)^2 / (<<F_m|F_m>> (w_m - w))."""
    s = 0
    for m in modes:
        s += qnm_field(m, x0, nr, nb) ** 2 / (qnm_norm(m, nr) * (qnm_freq(m, nr, nb) - w))
    return s / (2 * w)


def qnm_hfield(m, x, nr=NR, nb=NB):
    """Magnetic QNM inside the barrier, g_m = (d f_m/dx) / (i w_m) (units eps0 = mu0 = c = 1).
    At x = +-L/2 this is the limit from the inside (g is continuous there)."""
    k = qnm_freq(m, nr, nb)
    d = 1j * nr * k * (np.exp(1j * nr * k * x) - np.exp(-1j * nr * k * x + 1j * m * np.pi))
    return d / (1j * k)


def total_field_coeff(m, w, nr=NR, nb=NB):
    """Eq. (110) in 1D: a_m = (i/2) sum_{x=+-L/2} [f_m (n x H_in) - g_m (E_in x n)], over <<F_m|F_m>>.
    Incoming plane wave from the left, E_in = exp(i nb w x), H_in = nb E_in."""
    out = 0
    for s in (+1, -1):
        x = s * L / 2
        Ein = np.exp(1j * nb * w * x)
        Hin = nb * Ein
        f, g = qnm_field(m, x, nr, nb), qnm_hfield(m, x, nr, nb)
        out += s * (-f * Hin + g * Ein)
    return 0.5j * out / qnm_norm(m, nr)


def transmission_qnm_total(w, modes, nr=NR, nb=NB):
    """QNM approximation to the total field inside the barrier, Eq. (115), evaluated just inside x = L/2;
    t follows from the Field Equivalence Principle (continuity of E at x = L/2)."""
    w = np.atleast_1d(np.asarray(w, float))
    x = L / 2
    E = np.zeros_like(w, complex)
    for m in modes:
        a = np.array([total_field_coeff(m, wi, nr, nb) for wi in w])
        E += a / (w - qnm_freq(m, nr, nb)) * qnm_field(m, x, nr, nb)
    return E * np.exp(-1j * nb * w * L / 2)
