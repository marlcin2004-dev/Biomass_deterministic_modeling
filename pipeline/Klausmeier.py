from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class KlausmeierModel:
    """
    Klasa ułatwiająca rozwiązywanie i analizowanie Modelu Klausmeiera-Graya-Scotta
    """
    def __init__(self, a, d1, d2, m,
                 Lx, Ly, Nx, Ny,
                 ht=0.005, tol=1e-8, max_iter=8000,
                 start_water=1.0, start_biomass=1.0,
                 noise=0.05):

        self.a = a
        self.d1 = d1
        self.d2 = d2
        self.m = m

        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.ht = ht
        self.tol = tol
        self.max_iter = max_iter

        self.start_water = start_water
        self.start_biomass = start_biomass
        self.noise = noise

        self._build_grid()
        self._build_laplacian()
        self._initialize_fields()


    def _D2(self, N):
        e = np.ones(N)
        return sp.diags([e, -2*e, e], [-1, 0, 1], shape=(N, N), format="csr")

    def _build_grid(self):
        x = np.linspace(0, self.Lx, self.Nx)
        y = np.linspace(0, self.Ly, self.Ny)

        self.hx = x[1] - x[0]
        self.hy = y[1] - y[0]

        X, Y = np.meshgrid(x, y)
        self.Xf = X.ravel()
        self.Yf = Y.ravel()
        self.N = self.Nx * self.Ny

        # indeksy brzegów
        ind_left = np.where(self.Xf == x[0])[0]
        ind_right = np.where(self.Xf == x[-1])[0]
        ind_bot = np.where(self.Yf == y[0])[0]
        ind_top = np.where(self.Yf == y[-1])[0]

        self.boundary = np.unique(
            np.concatenate([ind_left, ind_right, ind_bot, ind_top])
        )
        self.interior = np.setdiff1d(np.arange(self.N), self.boundary)

    def _build_laplacian(self):
        Ix = sp.eye(self.Nx, format="csr")
        Iy = sp.eye(self.Ny, format="csr")

        L = (sp.kron(Iy, self._D2(self.Nx)) / self.hx**2 +
             sp.kron(self._D2(self.Ny), Ix) / self.hy**2)

        self.Au = sp.eye(self.N) - self.ht * self.d1 * L
        self.Av = sp.eye(self.N) - self.ht * self.d2 * L

    def _initialize_fields(self):

        self.u = self.start_water + self.noise * np.random.randn(self.N)
        self.v = self.start_biomass + self.noise * np.random.randn(self.N)

        self.u[self.boundary] = 0
        self.v[self.boundary] = 0


    def run(self):
        """
        Metoda przeprowadzająca symulację
        :return: u, v
        """
        for _ in range(self.max_iter):

            fu = self.a - self.u - self.u * self.v**2
            fv = self.u * self.v**2 - self.m * self.v

            bu = self.u + self.ht * fu
            bv = self.v + self.ht * fv

            bu[self.boundary] = 0
            bv[self.boundary] = 0


            u_new = spla.spsolve(self.Au, bu)
            v_new = spla.spsolve(self.Av, bv)

            u_new[self.boundary] = 0
            v_new[self.boundary] = 0

            if (np.linalg.norm(u_new - self.u, np.inf) < self.tol and
                    np.linalg.norm(v_new - self.v, np.inf) < self.tol):
                break

            self.u, self.v = u_new, v_new

        return self.u.reshape(self.Ny, self.Nx), self.v.reshape(self.Ny, self.Nx)


    def plot_biomass(self):
        v2d = self.v.reshape(self.Ny, self.Nx)
        plt.imshow(v2d, origin="lower", cmap="viridis")
        plt.colorbar(label="Biomass")
        plt.title(f"a = {self.a}")
        plt.show()

    def biomass_stats(self):
        """
        Metoda analizująca biomasę pomijając brzegi ponieważ one stale są równe zero i nie reprezentują symulacji na biomasie
        :return: średnia, maximum, wariancja
        """
        v_int = self.v[self.interior]
        return np.mean(v_int), np.max(v_int), np.var(v_int)