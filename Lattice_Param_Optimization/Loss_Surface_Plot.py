from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
import matplotlib.pyplot as plt
import torch
from funcs import get_spectrum, r_factor

q = torch.linspace(0, 180, 1000)

with MPRester(api_key="<api key>") as mpr:
    structure = mpr.get_structure_by_material_id("mp-22862")

sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()
calculator = XRDCalculator(wavelength="CuKa")
true_pattern = get_spectrum(conventional_structure, q)

# loop through different volumes and calculate the loss
vol = conventional_structure.volume
scaled_vol = torch.linspace(vol*.9, vol*1.1, 100)
loss = []
for value in scaled_vol:
    scaled_structure = conventional_structure.scale_lattice(value)
    scaled_pattern = get_spectrum(scaled_structure, q)
    r = r_factor(scaled_pattern, true_pattern)
    loss.append(r)

# plot
plt.plot(scaled_vol/vol*100, loss)
plt.xlabel("Percent of Initial Volume")
plt.ylabel("Loss")
plt.title("NaCL XRD Spectrum Loss with Scaled Volume")
plt.tight_layout()
plt.show()