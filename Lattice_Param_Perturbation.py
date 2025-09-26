from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

def perturb_lattice(struct, perturbation: ArrayLike, inplace: bool = True) -> Structure:
    """Apply a perturbation to lattice vector lengths.

    Args:
        perturbation (float or list): Amount to perturb lattice vector
            lengths in Angstroms. Applies same perturbation to all three if float
            is given.
        inplace (bool): True applies the strain in-place, False returns a
            Structure copy. Defaults to True.

    Returns:
        Structure: self if inplace=True else new structure with perturbation applied.
    """
    old_lattice = struct.lattice.matrix
    norms = np.linalg.norm(old_lattice, axis=1)
    perturbation_matrix = (1+ np.array(perturbation) / norms) * np.eye(3)
    new_lattice = Lattice(np.dot(old_lattice.T, perturbation_matrix).T)
    struct = struct if inplace else struct.copy()
    struct.lattice = new_lattice
    return struct


with MPRester(api_key="<api key>") as mpr:
    structure = mpr.get_structure_by_material_id("mp-22862")

# perform symmetry finding and construct conventional cell
sga = SpacegroupAnalyzer(structure)
conventional_structure = sga.get_conventional_standard_structure()

# plot XRD pattern
calculator = XRDCalculator(wavelength="CuKa")
pattern = calculator.get_pattern(conventional_structure)
ax = calculator.get_plot(conventional_structure)
for line in ax.lines:
    line.set_color("blue")

calculator.get_plot(perturb_lattice(conventional_structure, 1, False), ax=ax)
plt.show()
