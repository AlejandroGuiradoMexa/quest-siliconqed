import numpy as np

from sqed import barrier1d as B, cylinder as Cy, jc, materials as Mt


def test_barrier_qnm_satisfies_eq4():
    for m in (1, 3, 4, 7):
        w = B.qnm_freq(m)
        lhs = np.tan(B.NR * w * B.L) + 2j * B.NR * B.NB / (B.NR**2 + B.NB**2)
        assert abs(lhs) < 1e-10


def test_barrier_transmission_converges():
    w = np.array([4.0])
    te = B.transmission_exact(w)
    err = abs(B.transmission_qnm_total(w, range(-300, 301)) - te) / abs(te)
    assert err < 2e-5


def test_barrier_green_converges():
    g = B.green_qnm(0.1, 4.0, range(-3000, 3001))
    assert abs(g - B.green_exact(0.1, 4.0)) / abs(B.green_exact(0.1, 4.0)) < 1e-2


def test_cylinder_root():
    k = Cy.qnm(4, 3.48, 0.5, 4.2464 - 0.0043j)
    assert abs(k - (4.24637 - 0.00431j)) < 1e-4


def test_slab_effective_index_limits():
    ne = Mt.slab_te0(3.5, 1.0, 0.22, 1.278)
    assert 2.5 < ne < 3.5
    assert Mt.slab_te0(3.5, 1.0, 5.0, 1.278) > 3.45


def test_purcell_identity():
    """4 g^2 / (kappa gamma_rad) equals the Purcell formula (3/4pi^2) Q / V for a dipole at the field maximum."""
    lam, n, Q, Vn = 1.278e-6, 3.5, 4600, 0.26
    gam = 2.5e6
    mu = jc.dipole_from_rate(gam, n, lam)
    V = Vn * (lam / n) ** 3
    g = jc.coupling_g(mu, V, n, lam)
    kappa = 2 * np.pi * Mt.C0 / lam / Q
    assert np.isclose(4 * g**2 / (kappa * gam), jc.purcell_formula(Q, Vn), rtol=1e-12)


def test_master_equation_bad_cavity_rate():
    g, kappa, gamma = 1e9, 1e12, 1e6
    t = np.linspace(0, 1e-9, 5)
    _, rate = jc.emitter_decay(0.0, g, kappa, gamma, 0.0, t)
    assert np.isclose(rate, gamma + 4 * g**2 / kappa, rtol=1e-3)


def test_coupled_modes_match_liouvillian():
    g, kappa, gamma, gd, d = 3e10, 2e11, 1e8, 1e10, 5e10
    L = jc.liouvillian(d, g, kappa, gamma, gd)
    ev = np.linalg.eigvals(L)
    # coherences rho_{e0,g0} and rho_{g1,g0} evolve with -i x the coupled-mode eigenvalues
    cm = jc.coupled_modes(0.0, kappa, d, gamma + 2 * gd, g)
    for z in cm:
        assert np.min(abs(ev - (-1j * z))) < 1e-3 * abs(z)


def test_fem_cylinder_qnm():
    """The 2D FEM solver with PML reproduces an exact cylinder QNM."""
    from sqed import fem, geometry as G
    k = Cy.qnm(4, 3.48, 0.5, 4.2464 - 0.0043j)
    mesh = G.cylinder(R=0.5, h_in=0.08, h_out=0.2)
    S = fem.Maxwell2D(mesh, {"si": 3.48**2}, order=3)
    q = S.qnms(k, nev=4)
    assert abs(q[0][0] - k) / abs(k) < 1e-5
    # normalisation e^T M e = 1
    e = q[0][1]
    assert abs(e @ (S.M @ e) - 1) < 1e-10
