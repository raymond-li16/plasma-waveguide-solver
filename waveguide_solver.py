import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

"""
Methods for solving for waveguide structure, propagation lengths,
coupling efficiencies, transverse and longitudinal wavenumbers etc.
using the methods in 
Optical mode structure of the plasma waveguide, 
T. R. Clark and H. M. Milchberg, PRE 2000
"""

def parabolic_channel_eta(r, r_ch, n_e0, n_cr, delta_ne):
    """
    Calculate the parabolic channel index of refraction profile

    Parameters:
    r : array_like
        Radial coordinate (m)
    r_ch : float
        Channel radius (m)
    n_e0 : float
        Electron density on axis (m^-3)
    n_cr : float
        Critical density (m^-3)
    delta_ne : float
        Density difference between the channel center and r = r_ch (m^-3)
    Returns:
    eta : array_like
        Parabolic channel index of refraction profile
    """

    # Calculate the electron density profile
    n_e = n_e0 + delta_ne * (r / r_ch)**2

    # Calculate the index of refraction profile
    eta = np.sqrt(1 - n_e / n_cr)

    return eta

def transverse_wavenumber(r, eta):
    """
    Calculate the kappa function (transverse wavenumber) given
    a refractive index profile

    Parameters:
    r : array_like
        Radial coordinate (m)
    eta : array_like
        Index of refraction profile
    Returns:
    kappa : array_like
        transverse wavenumber for the waveguide modes
    """

def solve_waveguide_modes(r, eta, num_modes):
    """
    Solve for the waveguide modes using the finite difference method
    Equation to solve:
        Grad^2 E + kappa(r)

    Parameters:
    r : array_like
        Radial coordinate (m)
    eta : array_like
        Index of refraction profile
    num_modes : int
        Number of modes to solve for
    Returns:
    modes : array_like
        Eigenvalues corresponding to the waveguide modes
    """

    # Calculate the second derivative using finite differences
    dr = r[1] - r[0]
    d2_dr2 = diags([1, -2, 1], offsets=[-1, 0, 1], shape=(len(r), len(r))) / dr**2

    # Construct the operator for the waveguide modes
    operator = -d2_dr2 + np.diag(eta**2)

    # Solve for the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(operator, k=num_modes, which='SM')

    return eigenvalues, eigenvectors