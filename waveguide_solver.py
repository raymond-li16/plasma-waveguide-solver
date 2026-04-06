import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.interpolate import interp1d
from scipy.special import jv, yv
from scipy.signal import peak_widths, find_peaks
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    r : numpy array
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
    eta : numpy array
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
    r : numpy array
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
    eta : numpy array
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
    n_e : numpy array
        Electron density profile (m^-3)
    n_cr : float
        Critical density (m^-3)
    k0 : float
        Free space wavenumber (m^-1)
    beta : float
        Longitudinal wavenumber (m^-1)

    Returns:
    kappa_squared : numpy array
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
    r : numpy array
        Radial coordinate (m)
    kappa_squared : numpy array
        Transverse wavenumber profile squared (m^-2)
    Returns:
    mode_00 : numpy array
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

def solve_waveguide_fundamental_ivp(r, kappa_squared, m=0):
    """
    Solve for the waveguide fundamental mode as an initial value problem
    Equation to solve:
        Grad^2 E + kappa(r)^2 E = 0
        E(0) = 1, E'(0) = 0
    Assuming that E = u(r) exp(i m phi) for azimuthal mode number m

    Parameters:
    r : numpy array
        Radial coordinate (m)
    kappa_squared : numpy array
        Transverse wavenumber profile squared (m^-2)
    m : int
        Azimuthal mode number (default 0 for fundamental mode)
    
    Returns:
    mode_00 : numpy array
        Fundamental mode profile (normalized)
    """

    kappa2_interp = interp1d(r, kappa_squared, kind='linear', fill_value='extrapolate')

    def ode(r, y):
        u, v = y
        k2 = kappa2_interp(r)
        return [v, (-k2 + m**2/r**2) * u - v / r]

    eps = r[r > 0][0]  # first nonzero r point
    # High order solutions should approach a bessel function near r = 0
    # which is approximated by the polynomial u(r) = r^m near r = 0,
    # so we can use this as our initial condition for the IVP solver
    u0 = eps**m
    du0 = m * eps**(m - 1)
    y0 = [u0, du0]
    r_start = eps

    sol = solve_ivp(ode, [r_start, r[-1]], y0, t_eval=r[r >= r_start],
                    method='RK45', rtol=1e-6, atol=1e-10)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    mode_00 = sol.y[0]
    mode_00 = mode_00 / np.max(np.abs(mode_00))
    if mode_00[0] < 0:
        mode_00 = -mode_00

    return mode_00

def free_space_mode(r, kappa0):
    """
    Finds the free-space E-field mode profile for a given kappa0
    which will just be a bessel beam with characteristic
    transverse wavenumber kappa0

    Parameters:
    r : numpy array
        Radial coordinate (m)
    kappa0 : float
        Free-space transverse wavenumber (m^-1)

    Returns:
    mode : numpy array
        Free-space mode profile (normalized to have max amplitude of 1)
    """
    mode = jv(0, kappa0 * r)
    mode = mode / np.max(np.abs(mode))
    return mode

def calculate_rout(r, n_e):
    """
    Calculates r_out, which is the largest value of r for which n_e > 0

    Parameters:
    r : numpy array
        Radial coordinate (m), assumed to go from small to large values
    kappa_squared : numpy array
        Square of the transverse wavenumber profile (m^-2)

    Returns:
    r_out : float
        The radius at which kappa^2 transitions from negative to positive (m)
    """
    return r[np.where(n_e > 0)[0][-1]]

def calculate_eta(r, free_space_mode, quasibound_mode, r_out):
    """
    Calculates the coupling ratio eta between the free-space mode
    and the quasibound mode using the formula

    eta = integral_0^r_out [u(r)^2 2 pi r dr] / integral_0^r_out [u_free(r)^2 2 pi r dr]
    where u(r) is the normalized quasibound mode and
    u_free(r) is the normalized free-space mode
    where r_out is the maximum radius for which kappa^2 < 0

    u(r) and u_free(r) are normalized to have a mean amplitude of 1 for r > R/2. This 
    is because all quasibound modes eventually look like free-space modes at
    large radius since electron density goes to 0. The quasibound mode is required
    to have an electron density of zero for r > R/2.

    Parameters:
    r : numpy array
        Radial coordinate (m)
    free_space_mode : numpy array
        Free-space mode profile
    quasibound_mode : numpy array
        Quasibound mode profile
    r_out : float
        The radius at which kappa^2 transitions from negative to positive (m)
    
    Returns:
    eta : float
        Coupling ratio between the free-space mode and the quasibound mode
    """
    norm_region = r > (r[-1] / 2)
    u_free = free_space_mode
    u = quasibound_mode

    u_free_norm = np.mean(np.abs(u_free[norm_region]))
    u_norm = np.mean(np.abs(u[norm_region]))

    inside_rmax = r <= r_out

    u_free_normalized = u_free[inside_rmax] / u_free_norm
    u_normalized = u[inside_rmax] / u_norm
    r_in = r[inside_rmax]

    numerator = np.trapz(u_normalized**2 * 2 * np.pi * r_in, r_in)
    denominator = np.trapz(u_free_normalized**2 * 2 * np.pi * r_in, r_in)
    eta = numerator / denominator

    return eta

def find_eigenmodes(r, k0, beta_arr, n_e, n_cr, n_fwhm_points=10, threshold=0.1, m=0):
    """
    Searches for eigenmodes of the waveguide by scanning over a range of beta values
    and looking for peaks in the eta function. Each peak in eta corresponds to an
    eigenmode of the waveguide.

    Because some eigenmodes of the waveguide can be very exact in beta, this algorithm
    dynamically adjusts the beta scan range around every potential peak to find the
    exact beta value with high precision. The minimum resolution to find each peak is
    determined by n_fwhm_points, which is the minimum number of points to resolve the fwhm of each 
    peak by

    Parameters:
    r : numpy array
        Radial coordinate (m)
    k0 : float
        Free space wavenumber (m^-1)
    beta_arr : numpy array
        Array of beta values (longitudinal wavenumber) to scan over (m^-1)
    n_e : numpy array
        Electron density profile (m^-3)
    n_cr : float
        Critical density (m^-3)
    n_fwhm_points : int
        Minimum number of points to resolve the fwhm of each peak in eta (default 10)
    threshold : float
        Minimum height of peaks in eta to consider as potential eigenmodes (default 0.1)
    m : int
        Azimuthal mode number (default 0 for fundamental mode)

    Returns:
    beta_modes : list of float
        List of beta values corresponding to eigenmodes (m^-1)
    eta_modes : list of float
        List of eta values corresponding to eigenmodes
    beta_all : numpy array
        Array of all beta values scanned (m^-1)
    eta_all : numpy array
        Array of eta values corresponding to all beta values scanned
    """
    beta_all = beta_arr.copy()
    print("Performing initial coarse scan over beta values...")
    eta_all = find_etas(r, k0, beta_all, n_e, n_cr, m)

    unresolved_peaks = check_peaks_resolved(eta_all, n_fwhm_points, threshold)

    while len(unresolved_peaks) > 0:
        unresolved_betas = beta_all[unresolved_peaks] / k0
        print(f"Found unresolved peaks at beta/k0 = {unresolved_betas}, refining scan around these peaks...")
        for peak_idx in unresolved_peaks:
            # Find the neighbouring points around the unresolved peak
            # and create a finer beta array in that range
            beta_left = beta_all[peak_idx - 1] if peak_idx > 0 else beta_all[peak_idx]
            beta_right = beta_all[peak_idx + 1] if peak_idx < len(beta_all) - 1 else beta_all[peak_idx]

            beta_fine = np.linspace(beta_left, beta_right, n_fwhm_points + 2)
            eta_fine = find_etas(r, k0, beta_fine, n_e, n_cr, m)

            # Merge the fine scan data into the global arrays
            beta_all, eta_all = merge_high_res_data(beta_all, eta_all, beta_fine, eta_fine)

        unresolved_peaks = check_peaks_resolved(eta_all, n_fwhm_points, threshold)

    # Find the final peaks in the merged data to extract mode values
    peaks, _ = find_peaks(eta_all, height=threshold)
    beta_modes = list(beta_all[peaks])
    eta_modes = list(eta_all[peaks])

    return beta_modes, eta_modes, beta_all, eta_all

def find_etas(r, k0, beta_arr, n_e, n_cr, m=0):
    """
    Computes the coupling efficiency eta between the waveguide mode and the 
    free-space mode for each propagation constant in beta_arr.

    For each beta, the waveguide mode profile is found by solving the waveguide
    Helmholtz equation, and the coupling coefficient eta is found as defined in
    Clark et. al. 2000 equation 5

    Parameters:
    r : numpy array
        Radial coordinate array (m)
    k0 : float
        Free-space wavenumber (m^-1)
    beta_arr : numpy array
        Array of propagation constants to scan over (m^-1)
    n_e : numpy array
        Refractive index profile of the waveguide as a function of r
    n_cr : float
        Critical refractive index, used to compute the transverse wavenumber
    m : int, optional
        Azimuthal mode number (default 0, i.e. the fundamental azimuthal mode)

    Returns:
    eta_all : np.ndarray
        Array of coupling efficiencies corresponding to each value in beta_arr
    """
    eta_all = []
    for beta in tqdm(beta_arr, total=len(beta_arr), desc="Calculating eta for beta values"):
        kappa2 = transverse_wavenumber(n_e, n_cr, k0, beta)
        mode_00 = solve_waveguide_fundamental_ivp(r, kappa2, m)
        free_space_mode_profile = free_space_mode(r, np.sqrt(k0**2 - beta**2))
        r_out = calculate_rout(r, n_e)
        eta = calculate_eta(r, free_space_mode_profile, mode_00, r_out)
        eta_all.append(eta)
    return np.array(eta_all)

def check_peaks_resolved(eta_all, n_fwhm_points, threshold=0.1):
    """
    Checks if each peak in eta_all is resolved by at least n_fwhm_points points. 
    If any peak is not resolved, returns False.

    Parameters:
    eta_all : numpy array
        Array of eta values corresponding to all beta values scanned
    n_fwhm_points : int
        Minimum number of points to resolve the fwhm of each peak in eta
    threshold : float
        Minimum height of peaks to consider (default 0.1)

    Returns:
    unresolved_peaks : list of int
        List of indices of peaks that are not resolved. If all peaks are resolved, this list will be empty.
    """
    # Find peaks above the threshold
    peaks, _ = find_peaks(eta_all, prominence=threshold)

    if len(peaks) == 0:
        return True, []

    # Use rel_height=0.5 to measure width at half the peak's prominence,
    # which accounts for a non-zero baseline by measuring from the peak's
    # base (as determined by prominence) rather than from 0.
    widths, _, _, _ = peak_widths(eta_all, peaks, rel_height=0.5)
    print(widths)

    # Check which peaks have fewer than n_fwhm_points points across their FWHM
    unresolved_mask = widths < n_fwhm_points
    unresolved_peaks = list(peaks[unresolved_mask])

    return unresolved_peaks
        
def merge_high_res_data(x_coarse, y_coarse, x_fine, y_fine):
    """
    Merges the data from the coarse and fine scan
    by replacing the data from the coarse scan with the data 
    from the fine scan in the range of x values covered by the fine scan.

    Parameters:
    x_coarse : numpy array
        Array of x values from the coarse scan
    y_coarse : numpy array
        Array of y values from the coarse scan
    x_fine : numpy array
        Array of x values from the fine scan
    y_fine : numpy array
        Array of y values from the fine scan

    Returns:
    x_merged : numpy array
        Array of x values from the merged scan
    y_merged : numpy array
        Array of y values from the merged scan, 
        where the y value for each x value is the one from the 
        fine scan if it exists, and otherwise the one from the coarse scan
    """
    # Mask out coarse points that fall within the range of the fine scan
    fine_min, fine_max = x_fine.min(), x_fine.max()
    mask = (x_coarse <= fine_min) | (x_coarse >= fine_max)

    # Concatenate the filtered coarse data with the fine data
    x_merged = np.concatenate([x_coarse[mask], x_fine])
    y_merged = np.concatenate([y_coarse[mask], y_fine])

    # Sort by x values
    sort_indices = np.argsort(x_merged)

    return x_merged[sort_indices], y_merged[sort_indices]

def calculate_L_att(beta_all, eta_all, threshold=0.1):
    """
    Calculates the attenuation length L_att for each mode by finding the FWHM of each peak
    in eta_all. The attenuation length is 1/FWHM(peak)

    Parameters:
    beta_all : numpy array
        Array of beta values corresponding to all eta values (m^-1)
    eta_all : numpy array
        Array of eta values corresponding to all beta values scanned
    threshold : float
        Minimum height of peaks to consider for attenuation length calculation (default 0.1)

    Returns:
    L_att : list of float
        List of attenuation lengths for each mode (m)
    """
    # Find peaks above the threshold
    peaks, _ = find_peaks(eta_all, prominence=threshold)

    if len(peaks) == 0:
        return []

    # peak_widths returns widths in units of array indices, so we need to
    # convert to physical units. We use the interpolated left and right
    # intersection points (ips) returned by peak_widths to do this.
    _, _, left_ips, right_ips = peak_widths(eta_all, peaks, rel_height=0.5)

    # Interpolate the beta values at the fractional indices of the left and right
    # intersection points to get the FWHM in physical units
    index_arr = np.arange(len(beta_all))
    left_betas = np.interp(left_ips, index_arr, beta_all)
    right_betas = np.interp(right_ips, index_arr, beta_all)

    fwhm_betas = right_betas - left_betas

    L_att = list(1.0 / fwhm_betas)

    return L_att

if __name__ == "__main__":
    # Replicating the conditions in Clark et. al. 2000
    r_ch = 20e-6  # Channel radius (m)
    r_cutoff = 20e-6  # Density cutoff radius (m)
    shock_flat = 5e-6  # Shock transition width (m)
    shock_taper = 5e-6  # Shock taper width (m)
    n_e0 = 1e24   # On-axis density (m^-3)
    delta_ne = 1e24  # Density difference (m^-3)
    lam = 570e-9 # wavelength (m)
    n_cr = critical_density(lam)  # Critical density for 800 nm light

    r = np.linspace(1e-6, 1000e-6, 1000)  # Radial grid (m)
    n_e = truncated_parabolic_channel(r, r_ch, r_cutoff, shock_flat, shock_taper, n_e0, delta_ne)
    k0 = 2 * np.pi / lam  # Free space wavenumber (m^-1)
    beta = k0 * 0.999913
    kappa0 = np.sqrt(k0**2 - beta**2)
    kappa2 = transverse_wavenumber(n_e, n_cr, k0, beta)

    t = time.time()
    mode_00 = solve_waveguide_fundamental_ivp(r, kappa2, m=1)
    elapsed_ivp = time.time() - t
    print("Time elapsed: IVP solver: {:.4f} seconds".format(elapsed_ivp))
    free_space_mode_profile = free_space_mode(r, kappa0)

    # Plotting
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax0.plot(r * 1e6, mode_00, label='Guided Mode')
    ax0.plot(r * 1e6, free_space_mode_profile, label='Free Space Mode')
    ax0.legend()
    ax0.set_xlabel('Radius (µm)')
    ax0.set_ylabel('Mode Amplitude')
    ax0.set_xlim(left=0, right=100)

    ax1.plot(r * 1e6, kappa2 + 1/r**2)
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

    r_out = calculate_rout(r, n_e)
    eta = calculate_eta(r, free_space_mode_profile, mode_00, r_out)
    print(f"eta = {eta:.4f}")
    print(f"r_out = {r_out*1e6:.2f} µm")
