import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from scipy.constants import c, e, m_e, epsilon_0

"""
Methods for solving for waveguide structure, propagation lengths,
coupling efficiencies, transverse and longitudinal wavenumbers etc.
using the methods in 
Optical mode structure of the plasma waveguide, 
T. R. Clark and H. M. Milchberg, PRE 2000
"""

def critical_density(lam):
    """
    Calculate the critical density for a given wavelength

    Parameters:
    lam : float
        Wavelength (m)

    Returns:
    n_cr : float
        Critical density (m^-3)
    """
    omega = 2 * np.pi * c / lam
    n_cr = epsilon_0 * m_e * omega**2 / e**2
    return n_cr

def parabolic_channel(r, r_ch, n_e0, delta_ne):
    """
    Calculate the parabolic channel density profile

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
    n_e = n_e0 + delta_ne * (r / r_ch)**2
    return n_e

def truncated_parabolic_channel(r, r_ch, r_cutoff, shock_flat, shock_taper, n_e0, delta_ne):
    """
    Calculate a truncated parabolic channel density profile
    The channel is described by 
    n(e) = n_e0 + delta_ne * (r / r_ch)^2 for r <= r_cutoff
    n(e) = n(r_cutoff) for r_cutoff < r < r_cutoff + shock_width
    n(e) = n(r_cutoff) * (1 - (r - r_cutoff + shock_flat) / shock_taper)
        for r_cutoff + shock_width <= r < r_cutoff + shock_width + shock_taper
    n(e) = 0 for r >= r_cutoff + shock_width + shock_taper

    Parameters:
    r : array_like
        Radial coordinate (m)
    r_ch : float
        Channel radius (m)
    r_cutoff : float
        Radius at which the density is truncated (m)
    shock_flat : float
        Width of the shock flat region (m)
    shock_taper : float
        Width of the shock taper (m)
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
    n_e = n_e0 + delta_ne * (r / r_ch)**2
    ne_max = n_e0 + delta_ne * (r_cutoff / r_ch)**2
    n_e[r > r_cutoff] = ne_max
    n_e = np.where(r > r_cutoff + shock_flat, 
                   ne_max * (1 - (r - (r_cutoff + shock_flat)) / shock_taper), n_e)
    n_e[r > (r_cutoff + shock_flat + shock_taper)] = 0

    return n_e

def transverse_wavenumber(n_e, n_cr, k0, beta):
    """
    Calculate the kappa function (transverse wavenumber) given
    a transverse electron density profile
    The wavenumber is returned as kappa^2 because otherwise we could get
    imaginary values

    Parameters:
    n_e : array_like
        Electron density profile (m^-3)
    n_cr : float
        Critical density (m^-3)
    k0 : float
        Free space wavenumber (m^-1)
    beta : float
        Longitudinal wavenumber (m^-1)

    Returns:
    kappa_squared : array_like
        transverse wavenumber squared (m^-2)
    """
    # Equation 3 in the paper, ignoring the chi term
    kappa_squared = k0**2 * (1 - beta**2 / k0**2 - n_e / n_cr)
    return kappa_squared

def solve_waveguide_fundamental_bvp(r, kappa_squared):
    """
    Solve for the waveguide fundamental mode as a boundary value problem
    Equation to solve:
        Grad^2 E + kappa(r)^2 E = 0
        E(0) = 1, E(r_max) = 0

    Parameters:
    r : array_like
        Radial coordinate (m)
    kappa_squared : array_like
        Transverse wavenumber profile squared (m^-2)
    Returns:
    mode_00 : array_like
        Fundamental mode profile (normalized)
    """
    kappa2_interp = interp1d(r, kappa_squared, kind='linear', fill_value='extrapolate')

    def ode(r, y):
        u, v = y
        dudr = v
        # Use L'Hopital's rule at r=0: lim_{r->0} v/r = dv/dr
        # which gives dvdr = -kappa2/2 * u
        dvdr = np.where(r == 0,
                        -kappa2_interp(r) * u / 2,
                        -kappa2_interp(r) * u - v / r)
        return np.vstack([dudr, dvdr])

    def bc(ya, yb):
        # ya = [u(0), u'(0)], yb = [u(r_max), u'(r_max)]
        return np.array([
            ya[0] - 1,      # u(0) = 1
            yb[0],          # u(r_max) = 0
        ])

    # Initial guess — Gaussian decaying from axis
    u_guess = np.exp(-r**2 / r[-1]**2)
    v_guess = np.gradient(u_guess, r)
    y_guess = np.vstack([u_guess, v_guess])

    sol = solve_bvp(ode, bc, r, y_guess, tol=1e-8, max_nodes=100000)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    mode_00 = sol.sol(r)[0]
    mode_00 = mode_00 / np.max(np.abs(mode_00))
    if mode_00[0] < 0:
        mode_00 = -mode_00

    return mode_00

def solve_waveguide_fundamental_ivp(r, kappa_squared):
    """
    Solve for the waveguide fundamental mode as an initial value problem
    Equation to solve:
        Grad^2 E + kappa(r)^2 E = 0
        E(0) = 1, E'(0) = 0

    Parameters:
    r : array_like
        Radial coordinate (m)
    kappa_squared : array_like
        Transverse wavenumber profile squared (m^-2)
    Returns:
    mode_00 : array_like
        Fundamental mode profile (normalized)
    """

    kappa2_interp = interp1d(r, kappa_squared, kind='linear', fill_value='extrapolate')

    def ode(r, y):
        u, v = y
        dudr = v
        dvdr = np.where(r == 0,
                        -kappa2_interp(r) * u / 2,
                        -kappa2_interp(r) * u - v / r)
        return [dudr, dvdr]

    # Initial conditions: u(0) = 1, u'(0) = 0
    y0 = [1.0, 0.0]

    sol = solve_ivp(ode, [r[0], r[-1]], y0, t_eval=r,
                    method='RK45', rtol=1e-10, atol=1e-12)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    mode_00 = sol.y[0]
    mode_00 = mode_00 / np.max(np.abs(mode_00))
    if mode_00[0] < 0:
        mode_00 = -mode_00

    return mode_00

if __name__ == "__main__":
    # Example of parabolic density profile
    r_ch = 20e-6  # Channel radius (m)
    r_cutoff = 20e-6  # Density cutoff radius (m)
    shock_flat = 5e-6  # Shock transition width (m)
    shock_taper = 5e-6  # Shock taper width (m)
    n_e0 = 1e24   # On-axis density (m^-3)
    delta_ne = 1e24  # Density difference (m^-3)
    lam = 570e-9 # wavelength (m)
    n_cr = critical_density(lam)  # Critical density for 800 nm light

    r = np.linspace(1e-6, 1000e-6, 5000)  # Radial grid (m)
    n_e = truncated_parabolic_channel(r, r_ch, r_cutoff, shock_flat, shock_taper, n_e0, delta_ne)
    k0 = 2 * np.pi / lam  # Free space wavenumber (m^-1)
    beta = k0 * 0.9996
    kappa2 = transverse_wavenumber(n_e, n_cr, k0, beta)

    mode_00 = solve_waveguide_fundamental_ivp(r, kappa2)
    gaussian_00 = np.exp(-r**2 / r_ch**2)

    # Plotting
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax0.plot(r * 1e6, mode_00, label='Fundamental Mode (normalized)')
    ax0.set_xlabel('Radius (µm)')
    ax0.set_ylabel('Mode Amplitude')
    ax0.set_xlim(left=0, right=100)

    ax1.plot(r * 1e6, kappa2)
    ax2 = ax1.twinx()
    ax2.plot(r * 1e6, n_e, 'r--', label='Electron Density (x1e23 m^-3)')
    ax2.set_ylabel('Electron Density', color='r')
    ax2.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=100)
    ax1.set_xlabel('Radius (µm)')
    ax1.set_ylabel('$\kappa^2$ (m^-2)', color='b')
    # ax1.set_ylim(bottom=-np.amax(kappa2), top=np.amax(kappa2))

    fig.tight_layout()
    fig.savefig("mode.png")
