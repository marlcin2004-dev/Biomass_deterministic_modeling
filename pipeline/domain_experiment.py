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

# Parametry

Nx, Ny = 10, 10
a_values = [0.9, 1.1, 1.3, 1.5]

base_params = dict(
    d1=1,
    d2=0.1,
    m=0.45,
    Nx=Nx,
    Ny=Ny
)

cmap = plt.get_cmap('coolwarm_r')
norm = mcolors.Normalize(vmin=min(a_values), vmax=max(a_values))

plt.figure(figsize=(5, 10))

for a in a_values:
    line_color = cmap(norm(a))

    params = base_params.copy()
    params["a"] = a

    sizes, results = analyze_biomass_vs_size(1, 50, 2, **params)

    # 3. Przypisanie koloru do wykresu
    plt.plot(sizes, results, linewidth=2, label=f"a = {a}", color=line_color)

    v_star = stationary_v(a, base_params["m"])
    if v_star is not None:
        plt.axhline(v_star, linestyle='--', color=line_color)

    threshold = 1e-3
    positive_indices = np.where(results > threshold)[0]

    if len(positive_indices) > 0:
        L_crit = sizes[positive_indices[0]]
        plt.axvline(L_crit, linestyle=':', color = line_color)

plt.plot([], [], 'k--', label='Stan stacjonarny $v^*$')
plt.plot([], [], 'k:', label='Rozmiar krytyczny $L_{crit}$')
plt.title("Zbieżność μv(|Ω|) do niezerowego stanu")
plt.xlabel("Rozmiar obszaru |Ω|")
plt.ylabel("Średnia biomasa μv")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
