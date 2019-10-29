import numpy as np


def lifted_temperature(vortex, old_temperature, old_pressure, new_pressure):
    """Compute temperature of air parcel when lifted adiabatically to new pressure level.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    old_temperature: parcel temperature at original pressure level (K)
    old_pressure: parcel's original pressure level (Pa OR hPa)
    new_pressure: parcels's new pressure level (Pa OR hPa)
    Pressures/temperature accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Temperature of lifted parcel(s) (K)
    """
    new_temperature = old_temperature * (new_pressure / old_pressure) ** (vortex.Rd / vortex.cp)
    return new_temperature


def temperature_from_entropy(vortex, entropy, pressure):
    """Compute parcel temperature from specific entropy.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    entropy: parcel entropy eta = cp*ln(theta) (J/kg/K)
    pressure: parcel pressure (Pa)
    Entropy/pressure accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Temperature (K)
    """
    potential_temperature = np.exp(entropy/vortex.cp)
    temperature = potential_temperature*(pressure/vortex.p0)**(vortex.Rd/vortex.cp)
    return temperature


def potential_temperature_from_entropy(vortex, entropy):
    """Compute parcel potential temperature from specific entropy.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    entropy: parcel entropy eta = cp*ln(theta) (J/kg/K)
    Entropy accepted as float to compute for single parcel or numpy array to compute for many parcels.

    Returns
    -------
    Dry potential temperature (K)
    """
    potential_temperature = np.exp(entropy/vortex.cp)
    return potential_temperature


def entropy_from_temperature(vortex, temperature, pressure):
    """Compute parcel specific entropy from temperature.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    temperature: Parcel temperature (K)
    pressure: Parcel pressure (Pa)
    Temperature/pressure accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Parcel entropy (J/kg/K)
    """
    potential_temperature = temperature*(vortex.p0/pressure)**(vortex.Rd/vortex.cp)
    entropy = vortex.cp*np.log(potential_temperature)
    return entropy


def angular_momentum_radius_from_height(vortex, angular_momentum_parcel, z):
    """Compute radius at which given value of specific angular momentum occurs in reference vortex, for fixed height.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    angular_momentum_parcel: Value of angular momentum M required (m^2/s)
    z: Fixed height (m)
    M/z accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Radius r (m) at which reference angular momentum M_m(r, z) = M
    """
    # Convert M and z to numpy arrays if given as floats
    if isinstance(angular_momentum_parcel, np.float32) or isinstance(angular_momentum_parcel, np.float64) or isinstance(
            angular_momentum_parcel, float):
        z = np.array([z])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final radius

    # Specify initial interval to bisect
    lower_bound_r = 0.
    upper_bound_r = vortex.radial_edge

    # Initialise arrays for radius
    low_r = np.full(angular_momentum_parcel.shape, lower_bound_r)
    high_r = np.full(angular_momentum_parcel.shape, upper_bound_r)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_r - lower_bound_r) / dt_bound))

    for _ in np.arange(0, steps): # Bisection loop
        # Determine radius at interval midpoint
        mid_r = (low_r + high_r) / 2.
        # Determine difference between desired M and M_m at this radius
        angular_momentum_mid = vortex.angular_momentum(mid_r, z)
        delta_angular_momentum_mid = angular_momentum_mid - angular_momentum_parcel
        # Identify where M_m is too small/large for desired M
        pos_angular_momentum = np.where(delta_angular_momentum_mid > 0.)
        neg_angular_momentum = np.where(delta_angular_momentum_mid < 0.)
        # If M_m too small, restrict radius to upper half of interval
        low_r[neg_angular_momentum] = mid_r[neg_angular_momentum]
        # If M_m too large, restrict radius to lower half of interval
        high_r[pos_angular_momentum] = mid_r[pos_angular_momentum]

    # Take midpoint of final radius interval
    mid_r_new = (low_r + high_r) / 2.
    return mid_r_new


def reference_position(vortex, angular_momentum_parcel, entropy_parcel):
    """Compute reference position of parcel with given specific angular momentum and entropy.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    angular_momentum_parcel: parcel's specific angular momentum M (m^2/s)
    entropy_parcel: parcel's specific entropy (J/kg/K)
    M/entropy accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Reference radius r_* (m)
    Reference height z_* (m)
    """
    # Convert M and entropy to numpy arrays if given as floats
    if isinstance(entropy_parcel, np.float32) or isinstance(entropy_parcel, np.float64) or isinstance(entropy_parcel,
                                                                                                      float):
        entropy_parcel = np.array([entropy_parcel])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final position

    # Specify initial height interval to bisect
    lower_bound_z = 0.
    upper_bound_z = vortex.vertical_top

    # Initialise arrays for reference height
    low_z = np.full(angular_momentum_parcel.shape, lower_bound_z)
    high_z = np.full(angular_momentum_parcel.shape, upper_bound_z)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_z - lower_bound_z) / dt_bound))

    for _ in np.arange(0, steps): # Bisection loop
        # Determine height at interval midpoint
        mid_z = (low_z + high_z) / 2.
        # Determine radius at this height for which reference angular momentum equals parcel angular momentum M
        r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z)
        # Determine difference between parcel entropy and reference entropy at this radius/height
        entropy_mid = vortex.entropy(r_conserving_angular_momentum, mid_z)
        delta_entropy_mid = entropy_mid - entropy_parcel
        # Identify where reference entropy is too small/large compared to parcel entropy
        pos_entropy = np.where(delta_entropy_mid > 0.)
        neg_entropy = np.where(delta_entropy_mid < 0.)
        # If reference entropy too small, restrict height to upper half of interval
        low_z[neg_entropy] = mid_z[neg_entropy]
        # If reference entropy too large, restrict height to lower half of interval
        high_z[pos_entropy] = mid_z[pos_entropy]

    # Take midpoint of final height interval to obtain reference height
    mid_z_new = (low_z + high_z) / 2.
    # Find radius at reference height for which reference angular momentum equals parcel angular momentum M
    r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z_new)
    return r_conserving_angular_momentum, mid_z_new


def position_at_isobaric_surface(vortex, angular_momentum_parcel, reference_pressure_parcel):
    """Find point (r_mu, z_mu) at which surfaces of constant angular momentum and reference pressure intersect,
    required to separate available energy integral into two paths.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    angular_momentum_parcel: parcel's angular momentum M (m^2/s)
    reference_pressure_parcel: reference pressure p_m at parcel's position (Pa)
    M/p_m accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Radius r_mu at which integral path is separated into thermodynamic/mechanical component (m)
    Height z_mu at which integral path is separated into thermodynamic/mechanical component (m)
    """
    # Convert angular momentum M and reference pressure p_m to numpy arrays if given as floats
    if (isinstance(reference_pressure_parcel, np.float32) or isinstance(reference_pressure_parcel, np.float64) or
        isinstance(reference_pressure_parcel, float)):
        reference_pressure_parcel = np.array([reference_pressure_parcel])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final position

    # Specify initial height interval to bisect
    lower_bound_z = 0.
    upper_bound_z = vortex.vertical_top

    # Initialise arrays for reference height
    low_z = np.full(angular_momentum_parcel.shape, lower_bound_z)
    high_z = np.full(angular_momentum_parcel.shape, upper_bound_z)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_z - lower_bound_z) / dt_bound))

    for _ in np.arange(0, steps): # Bisection loop
        # Determine height at interval midpoint
        mid_z = (low_z + high_z) / 2.
        # Determine radius at this height for which reference angular momentum equals parcel angular momentum M
        r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z)
        # Determine difference between parcel's reference pressure and the reference pressure at this radius/height
        pressure_mid = vortex.pressure(r_conserving_angular_momentum, mid_z)
        delta_pressure_mid = pressure_mid - reference_pressure_parcel
        # Identify where reference pressure is too small/large compared to parcel's original reference pressure
        pos_pressure = np.where(delta_pressure_mid > 0.)
        neg_pressure = np.where(delta_pressure_mid < 0.)
        # If reference pressure too large, restrict height to upper half of interval
        low_z[pos_pressure] = mid_z[pos_pressure]
        # If reference pressure too small, restrict height to lower half of interval
        high_z[neg_pressure] = mid_z[neg_pressure]

    # Take midpoint of final height interval to obtain z_mu
    mid_z_new = (low_z + high_z) / 2.
    # Find radius at z_mu for which reference angular momentum equals parcel angular momentum M
    r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z_new)
    return r_conserving_angular_momentum, mid_z_new


def available_acoustic_energy(vortex, entropy, p, r, z):
    """Compute available acoustic energy Pi_1 (Equation 3.9, Tailleux & Harris 2019) for parcel relative to vortex.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    entropy: parcel's specific entropy (J/kg/K)
    p: parcel's pressure (Pa)
    r: parcel's radius (m)
    z: parcel's height (m)
    Parcel properties accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Parcel's available acoustic energy (J/kg)
    """
    temperature_in_situ = temperature_from_entropy(vortex, entropy, p)
    density_in_situ = p/(vortex.Rd * temperature_in_situ)
    enthalpy_in_situ = vortex.cp * temperature_in_situ
    enthalpy_at_pref = vortex.cp * lifted_temperature(vortex, temperature_in_situ, p, vortex.pressure(r, z))
    pressure_head = (vortex.pressure(r, z) - p)/density_in_situ
    return enthalpy_in_situ - enthalpy_at_pref + pressure_head


def vortex_available_energy(vortex, M, entropy, r, z):
    """Compute parcel's vortex available energy A_e (Equation 3.13, Tailleux & Harris 2019).

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    M: parcel's specific angular momentum (m^2/s)
    entropy: parcel's specific entropy (J/kg/K)
    r: parcel's radius (m)
    z: parcel's height (m)
    Parcel properties accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Vortex available energy A_e (J/kg)
    """
    # Compute terms of A_e according to Equation 3.13 (Tailleux & Harris, 2019)
    r_ref, z_ref = reference_position(vortex, M, entropy)
    M_terms = 0.5 * M**2 * (1. / r ** 2 - 1. / r_ref ** 2) + 0.125 * vortex.f ** 2 * (r ** 2 - r_ref ** 2)
    geopotential_terms = vortex.geopotential(z) - vortex.geopotential(z_ref)
    temperature_at_p0 = temperature_from_entropy(vortex, entropy, vortex.pressure(r, z))
    enthalpy_at_p0 = vortex.cp * temperature_at_p0
    enthalpy_at_reference = vortex.cp * lifted_temperature(vortex, temperature_at_p0, vortex.pressure(r, z),
                                                           vortex.pressure(r_ref, z_ref))
    return M_terms + geopotential_terms + enthalpy_at_p0 - enthalpy_at_reference


def pi_e(vortex, M, entropy, r, z):
    """Compute thermodynamic component Pi_e of vortex available energy A_e (Equation 3.19, Tailleux & Harris 2019).

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    M: parcel's specific angular momentum (m^2/s)
    entropy: parcel's specific entropy (J/kg/K)
    r: parcel's radius (m)
    z: parcel's height (m)
    Parcel properties accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Thermodynamic component Pi_e of vortex available energy (J/kg)
    """
    # Compute terms of Pi_e according to Equation 3.19 (Tailleux & Harris, 2019)
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    r_ref, z_ref = reference_position(vortex, M, entropy)
    M_terms = 0.5 * M**2 * (1. / r_mu ** 2 - 1. / r_ref ** 2) + 0.125 * vortex.f ** 2 * (r_mu ** 2 - r_ref ** 2)
    geopotential_terms = vortex.geopotential(z_mu) - vortex.geopotential(z_ref)
    temperature_at_p0 = temperature_from_entropy(vortex, entropy, vortex.pressure(r, z))
    enthalpy_at_p0 = vortex.cp * temperature_at_p0
    enthalpy_at_reference = vortex.cp * lifted_temperature(vortex, temperature_at_p0, vortex.pressure(r, z),
                                                           vortex.pressure(r_ref, z_ref))
    return M_terms + geopotential_terms + enthalpy_at_p0 - enthalpy_at_reference


def pi_k(vortex, M, r, z):
    """Compute mechanical component Pi_k of vortex available energy A_e (Equation 3.20, Tailleux & Harris 2019).

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    M: parcel's specific angular momentum (m^2/s)
    r: parcel's radius (m)
    z: parcel's height (m)
    Parcel properties accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Mechanical component Pi_k of vortex available energy (J/kg)
    """
    # Compute terms of Pi_k according to Equation 3.20 (Tailleux & Harris, 2019)
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    M_terms = 0.5 * M**2 * (1. / r ** 2 - 1. / r_mu ** 2) + 0.125 * vortex.f ** 2 * (r ** 2 - r_mu ** 2)
    geopotential_terms = vortex.geopotential(z) - vortex.geopotential(z_mu)
    return M_terms + geopotential_terms


def total_available_energy(vortex, M, entropy, p, r, z):
    """Compute sum of available acoustic energy and vortex available energy, Pi_1 + A_e.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    M: parcel's specific angular momentum (m^2/s)
    entropy: parcel's specific entropy (J/kg/K)
    p: parcel's pressure (Pa)
    r: parcel's radius (m)
    z: parcel's height (m)
    Parcel properties accepted as floats to compute for single parcel or numpy arrays to compute for many parcels.

    Returns
    -------
    Sum of available acoustic energy and vortex available energy (J/kg).
    """
    aae = available_acoustic_energy(vortex, entropy, p, r, z)
    ape = vortex_available_energy(vortex, M, entropy, r, z)
    return aae + ape


def vortex_available_energy_perturbations_M_entropy(vortex, r, z):
    """Compute vortex available energy for a parcel at a fixed position with perturbations in specfic angular
    momentum and entropy.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: radius (m)
    z: height (m)

    Returns
    -------
    2D numpy array of angular momentum perturbations (m^2/s)
    2D numpy array of entropy perturbations (J/kg/K)
    2D numpy array of vortex available energy A_e at (r, z) using perturbed angular momentum and entropy arrays (J/kg)
    """
    # Compute reference values of specific angular momentum and entropy at (r, z)
    base_M = vortex.angular_momentum(r, z)
    base_entropy = vortex.entropy(r, z)
    # Create grids of perturbed specific angular momentum and entropy comprising 100 evenly-spaced sample points
    # between their minimum/maximum values anywhere in the vortex domain
    all_M = vortex.gridded_variable(vortex.angular_momentum)
    all_entropy = vortex.gridded_variable(vortex.entropy)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100),
                                       np.linspace(all_entropy.min(), all_entropy.max(), 100))
    # Compute vortex available energy at (r, z) with perturbed angular momentum/entropy values
    ae_M_entropy = vortex_available_energy(vortex, M_grid, entropy_grid, r, z)
    return M_grid-base_M, entropy_grid-base_entropy, ae_M_entropy


def vortex_available_energy_perturbations_mu_pressure(vortex, r, z):
    """Compute vortex available energy for a parcel at a fixed position with perturbations in mu (specific angular
    momentum squared) and p_* (the reference pressure at the parcel's reference position (r_*, z_*)).

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: radius (m)
    z: height (m)

    Returns
    -------
    2D numpy array of mu perturbations (m^4/s^2)
    2D numpy array of p_* perturbations (kg/m/s^2)
    2D numpy array of vortex available energy A_e at (r, z) using perturbed mu and p_* arrays (J/kg)
    """
    # Compute reference values of mu and p_* at (r, z)
    base_mu = vortex.mu(r, z)
    base_pressure = vortex.pressure(r, z)
    # Get grids of specific angular momentum and entropy perturbations comprising 100 evenly-spaced sample points
    # between their minimum/maximum values anywhere in the vortex domain, and vortex available energy
    # at perturbed values
    M_perturbations, entropy_perturbations, ae_M_entropy = vortex_available_energy_perturbations_M_entropy(vortex, r, z)
    # Convert M and entropy perturbations to actual perturbed values
    M_grid = M_perturbations + vortex.angular_momentum(r, z)
    entropy_grid = entropy_perturbations + vortex.entropy(r, z)
    # Transform M and entropy perturbed values into mu and p_*
    mu_grid = M_grid ** 2
    r_ref, z_ref = reference_position(vortex, M_grid, entropy_grid)
    p_ref_grid = vortex.pressure(r_ref, z_ref)
    return base_mu - mu_grid, base_pressure - p_ref_grid, ae_M_entropy


def vortex_available_energy_perturbations_r_z(vortex, r, z):
    """Compute vortex available energy for parcels from all positions in the reference state when
    brought adiabatically along constant angular momentum surfaces to (r, z).

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: radius of fixed point (m)
    z: height of fixed point (m)

    Returns
    -------
    2D numpy array of radial perturbations (m)
    2D numpy array of vertical perturbations (m)
    2D numpy array of vortex available energy A_e when reference parcel at each perturbed position
                      is brought adiabatically to (r, z) conserving angular momentum (J/kg)
    """
    # Get r/z grid of whole domain
    r_grid, z_grid = vortex.grid()
    # Compute vortex available energy for each reference state parcel when brought to (r, z)
    ae_r_z = vortex_available_energy(vortex, vortex.gridded_variable(vortex.angular_momentum),
                                     vortex.gridded_variable(vortex.entropy), r, z)
    return r_grid - r, z_grid - z, ae_r_z


def pi_k_perturbations(vortex, r, z, v_range):
    """Compute eddy kinetic energy and Pi_k for a range of perturbations around reference azimuthal wind at a point.

    Parameters
    ----------
    vortex: instance of Vortex class determining reference state
    r: radius (m)
    z: height (m)
    v_range: maximum range in azimuthal wind to sample around reference wind (in both directions) (m/s)

    Returns
    -------
    1D numpy array of azimuthal wind perturbations around reference wind (m/s)
    1D numpy array of Pi_k at perturbation winds (J/kg)
    1D numpy array of eddy kinetic energy at perturbation winds (J/kg)
    """
    # Find azimuthal wind v of reference vortex at point
    base_v = vortex.azimuthal_wind(r, z)
    # Create array of perturbed v values with spacing 1e-4 m/s
    lower_v = max(0., base_v - v_range)
    upper_v = base_v + v_range
    number_steps = (upper_v - lower_v) / 0.0001
    all_v = np.linspace(lower_v, upper_v, number_steps)
    # Convert to perturbed angular momentum values
    all_M = vortex.angular_momentum_from_azimuthal_wind(all_v, r)
    # Compute Pi_k for perturbed angular momentum values
    perturbed_pi_k = np.squeeze(pi_k(vortex, all_M, r, z))
    # Compute eddy kinetic energy for perturbations
    quadratic_perturbed_velocity = 0.5 * (all_v - base_v) ** 2
    return all_v - base_v, perturbed_pi_k, quadratic_perturbed_velocity
