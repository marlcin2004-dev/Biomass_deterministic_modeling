import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tqdm as tqdm



def D2(N):
    e = np.ones(N)
    return sp.diags([e, -2*e, e], [-1, 0, 1], shape=(N, N), format="csr")

def KGS_a_change(a_l, d1, d2, m, Lx, Ly, Nx, Ny, ht = 0.005, max_iter = 1000, tol = 1e-6, start_biomas = 1, start_water = 1):
    """
    Funkcja pozwala obliczyć średni i maksymalny poziom biomasy dla danych parametów a w a_l
    :param a_l: Lista parametrów a (współczynnik opadów)
    :param d1:  Dyfuzja wody
    :param d2:  Dyfuzja biomasy
    :param m:   Współczynnik śmiertelności
    :param Lx:  Wielkość dziedziny na x
    :param Ly:  Wielkość dziedziny na y
    :param Nx:  Podział dziedziny x
    :param Ny:  Podział dziedziny y
    :param ht:  Krok czasowy
    :param max_iter:  Maksymalna liczba iteracji
    :param tol:     Tolerancja
    :param start_biomas: Warunek początkowwy biomasy
    :param start_water: Warunek początkowy wody
    :return: avg_biomass, max_biomass
    """
    avg_biomass = []
    max_biomass = []

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    Xf = X.ravel()
    Yf = Y.ravel()

    N = Nx * Ny

    ind_left = np.where(Xf == x[0])[0]
    ind_right = np.where(Xf == x[-1])[0]
    ind_bot = np.where(Yf == y[0])[0]
    ind_top = np.where(Yf == y[-1])[0]
    indices_boundary = np.concatenate([ind_left, ind_right, ind_bot, ind_top])

    indices_interior = np.setdiff1d(np.arange(N), indices_boundary)

    # start
    u = np.ones(N) * start_water
    v = np.ones(N) * start_biomas

    # Dirichlet
    u[indices_boundary] = 0.0
    v[indices_boundary] = 0.0

    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")

    L = sp.kron(Iy, D2(Nx), format="csr") / hx ** 2 + sp.kron(D2(Ny), Ix, format="csr") / hy ** 2

    Au = sp.eye(N, format="csr") - ht * d1 * L
    Av = sp.eye(N, format="csr") - ht * d2 * L

    for a in tqdm.tqdm(a_l):

        for _ in range(max_iter):
            fu = a - u - u * v ** 2
            fv = u * v ** 2 - m * v

            bu = u + ht * fu
            bv = v + ht * fv

            # Dirichlet
            bu[indices_boundary] = 0.0
            bv[indices_boundary] = 0.0

            u_new = spla.spsolve(Au, bu)
            v_new = spla.spsolve(Av, bv)

            if (np.linalg.norm(u_new - u, np.inf) < tol and
                    np.linalg.norm(v_new - v, np.inf) < tol):
                u, v = u_new, v_new
                break

            u, v = u_new, v_new

        avg_biomass.append(np.mean(v[indices_interior]))  # wywalamy te 0 na brzegach wtedy
        max_biomass.append(np.max(v))

    return  avg_biomass, max_biomass


