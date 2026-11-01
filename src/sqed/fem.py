"""2D frequency-domain Maxwell solver (in-plane E, Nedelec elements, NGSolve) with PML:
quasinormal modes (QNMs), their normalisation, and driven solutions.

Units: lengths in um, k = w/c in 1/um, eps0 = mu0 = c = 1. Time dependence exp(-i w t).
Equation: curl curl E - k^2 eps E = f, discretised as (A - k^2 M) e = f.
With a PML, A and M are complex symmetric; the QNMs are normalised with e^T M e = 1, i.e. the
PML-regularised form <<f|f>> = int eps f.f dA (Sauvan et al. 2013; Kristensen et al. 2020).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ngsolve as ngs
from ngsolve import HCurl, BilinearForm, LinearForm, GridFunction, curl, dx

ngs.SetNumThreads(4)


def _csr(bf, free):
    v, c, r = bf.mat.CSR()
    M = sp.csr_matrix((np.asarray(v), np.asarray(c), np.asarray(r)), shape=(bf.mat.height, bf.mat.width))
    return M[free][:, free].tocsc()


class Maxwell2D:
    def __init__(self, mesh, eps, order=3, dirichlet="outer"):
        """eps: dict material -> permittivity (default 1)."""
        self.mesh = mesh
        self.eps_dict = eps
        self.eps = mesh.MaterialCF(eps, default=1.0)
        self.fes = HCurl(mesh, order=order, complex=True, dirichlet=dirichlet)
        u, v = self.fes.TnT()
        self.a = BilinearForm(curl(u) * curl(v) * dx).Assemble()
        self.m = BilinearForm(self.eps * u * v * dx).Assemble()
        self.free = np.array(list(self.fes.FreeDofs()), bool)
        self.A = _csr(self.a, self.free)
        self.M = _csr(self.m, self.free)
        self.ndof = int(self.free.sum())
        pml_dofs = np.zeros(self.fes.ndof, bool)
        if "pml" in mesh.GetMaterials():
            for d in self.fes.GetDofs(mesh.Materials("pml")):
                pass
            pml_dofs = np.array(list(self.fes.GetDofs(mesh.Materials("pml"))), bool)
        self.pml = pml_dofs[self.free]
        self._lu = {}

    # ---------------------------------------------------------------- modes
    def qnms(self, k_target, nev=8, pml_max=0.5):
        """QNMs near k_target (complex k with Im k < 0), normalised with e^T M e = 1.
        Spurious PML modes (fraction of |e|^2 on PML dofs above pml_max) are dropped."""
        sig = complex(k_target) ** 2
        lu = sla.splu((self.A - sig * self.M).tocsc())
        op = sla.LinearOperator(self.A.shape, matvec=lambda x: lu.solve(self.M @ x), dtype=complex)
        mu, V = sla.eigs(op, k=nev, which="LM", tol=1e-12)
        out = []
        for i in range(len(mu)):
            k = np.sqrt(sig + 1 / mu[i])
            if k.real < 0:
                k = -k
            e = V[:, i]
            frac = np.sum(abs(e[self.pml]) ** 2) / np.sum(abs(e) ** 2)
            if frac > pml_max:
                continue
            e = e / np.sqrt(e @ (self.M @ e))
            out.append((k, e, frac))
        out.sort(key=lambda t: abs(t[0] - k_target))
        return out

    # ---------------------------------------------------------------- driven
    def solve(self, k, f):
        """(A - k^2 M) e = f for a free-dof vector f."""
        lu = sla.splu((self.A - complex(k) ** 2 * self.M).tocsc())
        return lu.solve(f)

    def rhs(self, cf, region=None):
        """Free-dof vector of int cf . v dA (over a material region if given)."""
        _, v = self.fes.TnT()
        lf = LinearForm(self.fes)
        lf += (cf * v) * (dx(definedon=self.mesh.Materials(region)) if region else dx)
        lf.Assemble()
        return np.asarray(lf.vec.FV().NumPy())[self.free].copy()

    def gf(self, e):
        g = GridFunction(self.fes)
        full = np.zeros(self.fes.ndof, complex)
        full[self.free] = e
        g.vec.FV().NumPy()[:] = full
        return g

    def eval(self, e, x, y):
        """E field (N x 2 complex) at points (x, y)."""
        g = self.gf(e) if not isinstance(e, GridFunction) else e
        pts = self.mesh(np.atleast_1d(x), np.atleast_1d(y))
        return np.asarray(g(pts))

    def integrate(self, cf, region=None):
        return ngs.Integrate(cf, self.mesh, definedon=self.mesh.Materials(region) if region else None)
