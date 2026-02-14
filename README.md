# Model Klausmeiera-Graya-Scotta: Analiza Ekosystemów Półpustynnych

## O projekcie
Projekt analizuje model **Klausmeiera-Graya-Scotta (KGS)**, który opisuje dynamikę roślinności w ekosystemach półpustynnych (*drylands*). 

Model wyjaśnia zjawisko **samoorganizacji przestrzennej** – powstawania wzorów takich jak cętki, labirynty czy pasy (*tiger bush*). Wynikają one z nieliniowego sprzężenia: rośliny ułatwiają infiltrację wody, co napędza ich wzrost, ale jednocześnie konkurują o nią z sąsiedztwem. Projekt bada mechanizmy tych procesów oraz ryzyko nagłego, katastrofalnego pustynnienia pod wpływem zmian klimatycznych.

## Model Matematyczny
Układ równań typu reakcja-dyfuzja w postaci bezwymiarowej:
- $\frac{\partial u}{\partial t} = a - u - uv^2 + d_1 \Delta u$
- $\frac{\partial v}{\partial t} = uv^2 - mv + d_2 \Delta v$

### Parametry:
* **$a$**: Poziom opadów deszczu (zasilanie w wodę).
* **$m$**: Współczynnik śmiertelności roślin.
* **$u$**: Dostępna woda w glebie.
* **$v$**: Gęstość biomasy roślinnej.
* **$d_1, d_2$**: Współczynniki dyfuzji odpowiednio wody i biomasy.

## Struktura Projektu

### Skrypty (.py)
* `Klausmeier.py` – **Rdzeń projektu**. Klasa zawierająca solver numeryczny i logikę modelu. Niezbędna do działania pozostałych skryptów.
* `bifurcation.py` – Analiza punktów krytycznych i zjawiska histerezy (tipping points).
* `domain_experiment.py` – Badanie wpływu rozmiaru obszaru $\Omega$ na stabilność rozwiązań.

### Notebooki (.ipynb)
Dla łatwiejszego zrozumienia kodu przygotowano przykłady wywołań i wizualizacje:
* `bifurcation_Mod_Det.ipynb` – Generowanie diagramów bifurkacyjnych.
* `patterns_Mod_Det.ipynb` – Symulacja "ogrodu zoologicznego" wzorów Turinga.
* `domain_changes_Mod_Det.ipynb` – Eksperymenty z rozmiarem domeny.

### Dane i wyniki
* `data/` – Folder zawierający parametry fizyczne.

## Instrukcja obsługi

1. **Zainstaluj wymagane biblioteki:**
   ```bash
   pip install -r requirements.txt
