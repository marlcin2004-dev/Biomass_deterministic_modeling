from Klausmeier import KlausmeierModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class BifurcationExperiment:

    def __init__(self, a_values, model_params):
        """
        Class used to show bifurcation, how parameter "a" affects the end result of a Kluasmeier-Gray-Scott model
        :array a_values:
        :dict KGS_model_params:
        """
        self.a_values = a_values
        self.model_params = model_params

    def run(self):
        """
        Symulacja bifurkacji: najpierw gałąź malejąca,
        potem rosnąca startująca z niezerowego rozwiązania.
        """

        # ===== GAŁĄŹ MALEJĄCA =====
        avg_down = []
        max_down = []

        previous_u = None
        previous_v = None

        nonzero_u = None
        nonzero_v = None

        print("Simulation for a descending...")

        for a in tqdm(self.a_values):

            model = KlausmeierModel(a=a, **self.model_params)

            # dodajemy stan z poprzedniego kroku
            if previous_u is not None:
                model.u = previous_u.copy()
                model.v = previous_v.copy()

            model.run()

            avg_v, max_v, _ = model.biomass_stats()

            avg_down.append(avg_v)
            max_down.append(max_v)

            previous_u = model.u.copy()
            previous_v = model.v.copy()

            # zapamiętaj pierwsze niezerowe stabilne rozwiązanie
            if nonzero_v is None and np.max(model.v) > 1.0:
                nonzero_u = model.u.copy()
                nonzero_v = model.v.copy()

        # ===== GAŁĄŹ ROSNĄCA =====
        avg_up = []
        max_up = []

        print("Simulation for a increasing...")

        # jeśli znaleziono niezerowe rozwiązanie to startujemy z niego
        if nonzero_v is not None:
            previous_u = nonzero_u.copy()
            previous_v = nonzero_v.copy()
        else:
            previous_u = None
            previous_v = None

        for a in tqdm(reversed(self.a_values)):

            model = KlausmeierModel(a=a, **self.model_params)

            if previous_u is not None:
                model.u = previous_u.copy()
                model.v = previous_v.copy()

            model.run()

            avg_v, max_v, _ = model.biomass_stats()

            avg_up.append(avg_v)
            max_up.append(max_v)

            previous_u = model.u.copy()
            previous_v = model.v.copy()

        # odwracamy żeby pasowało do osi
        avg_up.reverse()
        max_up.reverse()

        return avg_down, max_down, avg_up, max_up



