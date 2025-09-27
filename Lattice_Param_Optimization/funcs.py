from mp_api.client import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from jaxtyping import Float
import torch

def initialize_lattice(structure, vol_estimate=6**3) -> Structure:
    rand_lattice = np.random.rand(3,3)
    rand_volume = np.linalg.det(rand_lattice) 
    pred_lattice = rand_lattice * (vol_estimate / abs(rand_volume))**(1/3) * np.sign(rand_volume)
    structure = structure.copy()
    structure.lattice = pred_lattice
    return structure

def perturb_lattice(structure, perturbation: ArrayLike, inplace: bool = True) -> Structure:
    old_lattice = structure.lattice.matrix
    norms = np.linalg.norm(old_lattice, axis=1)
    perturbation_matrix = (1+ np.array(perturbation) / norms) * np.eye(3)
    new_lattice = Lattice(np.dot(old_lattice.T, perturbation_matrix).T)
    structure = structure if inplace else structure.copy()
    structure.lattice = new_lattice
    return structure

def noise_lattice(structure, noise_lvl=0.5) -> Structure:
    old_lattice = structure.lattice.matrix
    perturbation_matrix = noise_lvl * np.random.randn(3,3)
    new_lattice = Lattice(old_lattice + perturbation_matrix)
    structure = structure.copy()
    structure.lattice = new_lattice
    return structure


sqrt2π = np.sqrt(2 * np.pi)
twosqrttwolog2 = 2 * np.sqrt(2 * np.log(2))

def gaussian(
        x: Float[torch.Tensor, "n_points"],
        center: Float[torch.Tensor, "n_peaks"],
        fwhm: Float[torch.Tensor, "n_peaks"],
) -> Float[torch.Tensor, "n_points n_peaks"]:
    """Broadcasted GSAS-style Gaussian profile parameterized by FWHM (gamma)."""
    dx = x.unsqueeze(0) - center.unsqueeze(1)
    _fwhm = fwhm.unsqueeze(1)
    sigma = (_fwhm / (twosqrttwolog2)) ** 2
    expt = -(dx ** 2) / (2 * sigma)
    return (twosqrttwolog2 * torch.exp(expt)) / (_fwhm * sqrt2π)


def get_spectrum(structure: Structure, q_vals, var=0.1):
    # perform symmetry finding and construct conventional cell
    sga = SpacegroupAnalyzer(structure)
    conventional_structure = sga.get_conventional_standard_structure()

    # get xrd pattern
    calculator = XRDCalculator(wavelength="CuKa")
    pattern = calculator.get_pattern(conventional_structure, two_theta_range=(q_vals[0], q_vals[-1]))
    q = torch.from_numpy(pattern.x)
    intensities = torch.from_numpy(pattern.y)
    fwhm = torch.ones_like(q)*var

    # broaden peaks
    peaks = gaussian(q_vals, q, fwhm)
    spectrum = (peaks * intensities[:, None]).sum(0)
    spectrum_min = spectrum.min()
    spectrum_max = spectrum.max()
    spectrum = (spectrum - spectrum_min) / (spectrum_max - spectrum_min)
    return spectrum

def r_factor(pred_pattern: ArrayLike, true_pattern: ArrayLike) -> Float:
    error = ((pred_pattern-true_pattern)**2).sum(0)
    normalization = (true_pattern**2).sum(0)
    return error/normalization

