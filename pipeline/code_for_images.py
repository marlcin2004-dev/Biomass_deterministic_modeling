import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import tqdm as tqdm


def D2(N):
    e = np.ones(N)
    return sp.diags([e, -2*e, e], [-1, 0, 1], shape=(N, N), format="csr")




def KGS_pattern(a, d1, d2, m, Lx, Ly, Nx, Ny, ht=0.005, max_iter=5000, tol=1e-6, start_biomas = 1, start_water = 1):

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)
    Xf = X.ravel()
    Yf = Y.ravel()
    N = Nx * Ny

    # brzegi
    ind_left = np.where(Xf == x[0])[0]
    ind_right = np.where(Xf == x[-1])[0]
    ind_bot = np.where(Yf == y[0])[0]
    ind_top = np.where(Yf == y[-1])[0]
    boundary = np.unique(np.concatenate([ind_left, ind_right, ind_bot, ind_top]))

    # start
    noise = 0.05
    u = start_water * (1 + noise * np.random.randn(N))
    v = start_biomas * (1 + noise * np.random.randn(N))

    u[boundary] = 0.0
    v[boundary] = 0.0

    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")
    L = sp.kron(Iy, D2(Nx)) / hx**2 + sp.kron(D2(Ny), Ix) / hy**2

    Au = sp.eye(N) - ht * d1 * L
    Av = sp.eye(N) - ht * d2 * L

    for _ in tqdm.tqdm(range(max_iter)):
        fu = a - u - u * v**2
        fv = u * v**2 - m * v

        bu = u + ht * fu
        bv = v + ht * fv

        bu[boundary] = 0.0
        bv[boundary] = 0.0

        u_new = spla.spsolve(Au, bu)
        v_new = spla.spsolve(Av, bv)

        if (np.linalg.norm(u_new - u, np.inf) < tol and
            np.linalg.norm(v_new - v, np.inf) < tol):
            break

        u, v = u_new, v_new

    return u.reshape(Ny, Nx), v.reshape(Ny, Nx)

