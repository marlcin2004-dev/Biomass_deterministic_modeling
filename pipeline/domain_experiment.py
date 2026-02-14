from Klausmeier import KlausmeierModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def analyze_biomass_vs_size(start_size=10, end_size=30, step=2, **model_params):
    """
    Zlicza średnią biomasę dla rosnących wymiarów Lx = Ly.
    """
    sizes = np.arange(start_size, end_size + 1, step)
    mean_biomasses = []

    for L in tqdm(sizes):
        model = KlausmeierModel(Lx=L, Ly=L, **model_params)

        # warunki początkowe
        u0 = 2.0 * np.ones(model.Nx * model.Ny)
        v0 = 2.0 * np.ones(model.Nx * model.Ny) + 0.05 * np.random.randn(model.Nx * model.Ny)

        u0[model.boundary] = 0.0
        v0[model.boundary] = 0.0

        model.u = u0
        model.v = v0

        model.run()

        mean_v, _, _ = model.biomass_stats()
        mean_biomasses.append(mean_v)

    return sizes, np.array(mean_biomasses)




def stationary_v(a, m):
    """
    Niezerowy stan stacjonarny (większy pierwiastek).
    """
    if a <= 2 * m:
        return None

    return (a + np.sqrt(a ** 2 - 4 * m ** 2)) / (2 * m)


