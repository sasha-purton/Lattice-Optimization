from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
import matplotlib.pyplot as plt
import torch
from funcs import get_spectrum, initialize_lattice, r_factor, noise_lattice

TRIALS = 1000
q = torch.linspace(0, 180, 1000)

with MPRester(api_key="<api key>") as mpr:
    structure = mpr.get_structure_by_material_id("mp-22862")

# get true xrd spectrum
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()
calculator = XRDCalculator(wavelength="CuKa")
true_pattern = get_spectrum(conventional_structure, q)

# generate random lattice vectors with initial volume guess
pred_structure = initialize_lattice(conventional_structure, vol_estimate=6**3)
pred_pattern = get_spectrum(pred_structure, q)

# calculate losses and loop through progressively better xrd patterns
loss = r_factor(pred_pattern, true_pattern)
loss_tracker = [loss]

for i in range(TRIALS-1):
    new_pred = noise_lattice(pred_structure, noise_lvl=0.05)
    new_pattern = get_spectrum(new_pred, q)
    new_loss = r_factor(new_pattern, true_pattern)
    if new_loss < loss:
        loss = new_loss
        pred_structure = new_pred
    loss_tracker.append(loss)

# plot
x = torch.arange(1,TRIALS+1)
plt.plot(x, loss_tracker)
plt.xlabel("Trial Number")
plt.ylabel("Loss")
plt.title("Naive Monte Carlo Lattice Optimization")
plt.tight_layout()
plt.show()