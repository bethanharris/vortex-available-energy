import numpy as np
from Vortex import Vortex


def lifted_temperature(vortex, old_temperature, old_pressure, new_pressure):
    new_temperature = old_temperature * (new_pressure / old_pressure) ** (vortex.Rd / vortex.cp)
    return new_temperature


def temperature_from_entropy(vortex, entropy, pressure):
    potential_temperature = np.exp(entropy/vortex.cp)
    temperature = potential_temperature*(pressure/vortex.p0)**(vortex.Rd/vortex.cp)
    return temperature


def potential_temperature_from_entropy(vortex, entropy):
    potential_temperature = np.exp(entropy/vortex.cp)
    return potential_temperature


def entropy_from_temperature(vortex, temperature, pressure):
    potential_temperature = temperature*(vortex.p0/pressure)**(vortex.Rd/vortex.cp)
    entropy = vortex.cp*np.log(potential_temperature)
    return entropy


def angular_momentum_radius_from_height(vortex, angular_momentum_parcel, z):
    if isinstance(angular_momentum_parcel, np.float32) or isinstance(angular_momentum_parcel, np.float64) or isinstance(angular_momentum_parcel, float):
        z = np.array([z])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final position

    lower_bound_r = 0.
    upper_bound_r = 200000.

    low_r = np.full(angular_momentum_parcel.shape, lower_bound_r)
    high_r = np.full(angular_momentum_parcel.shape, upper_bound_r)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_r - lower_bound_r) / dt_bound))

    # Bisect
    for _ in np.arange(0, steps):
        mid_r = (low_r + high_r) / 2.
        angular_momentum_mid = vortex.angular_momentum(mid_r, z)
        delta_angular_momentum_mid = angular_momentum_mid - angular_momentum_parcel
        pos_angular_momentum = np.where(delta_angular_momentum_mid > 0.)
        neg_angular_momentum = np.where(delta_angular_momentum_mid < 0.)

        low_r[neg_angular_momentum] = mid_r[neg_angular_momentum]
        high_r[pos_angular_momentum] = mid_r[pos_angular_momentum]

        mid_r_new = (low_r + high_r) / 2.
    return mid_r_new


def reference_position(vortex, angular_momentum_parcel, entropy_parcel):
    if isinstance(entropy_parcel, np.float32) or isinstance(entropy_parcel, np.float64) or isinstance(entropy_parcel, float):
        entropy_parcel = np.array([entropy_parcel])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final position

    lower_bound_z = 0.
    upper_bound_z = 16000.

    low_z = np.full(angular_momentum_parcel.shape, lower_bound_z)
    high_z = np.full(angular_momentum_parcel.shape, upper_bound_z)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_z - lower_bound_z) / dt_bound))

    # Bisect
    for _ in np.arange(0, steps):
        mid_z = (low_z + high_z) / 2.
        r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z)
        entropy_mid = vortex.entropy(r_conserving_angular_momentum, mid_z)
        delta_entropy_mid = entropy_mid - entropy_parcel
        pos_entropy = np.where(delta_entropy_mid > 0.)
        neg_entropy = np.where(delta_entropy_mid < 0.)

        low_z[neg_entropy] = mid_z[neg_entropy]
        high_z[pos_entropy] = mid_z[pos_entropy]

        mid_z_new = (low_z + high_z) / 2.

    r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z_new)

    return r_conserving_angular_momentum, mid_z_new


def position_at_isobaric_surface(vortex, angular_momentum_parcel, reference_pressure_parcel):
    if isinstance(reference_pressure_parcel, np.float32) or isinstance(reference_pressure_parcel, np.float64) or isinstance(reference_pressure_parcel, float):
        reference_pressure_parcel = np.array([reference_pressure_parcel])
        angular_momentum_parcel = np.array([angular_momentum_parcel])

    dt_bound = 1e-10  # desired precision of final position

    lower_bound_z = 0.
    upper_bound_z = 16000.

    low_z = np.full(angular_momentum_parcel.shape, lower_bound_z)
    high_z = np.full(angular_momentum_parcel.shape, upper_bound_z)

    # Find number of bisection steps required to reach desired accuracy
    steps = np.ceil(np.log2((upper_bound_z - lower_bound_z) / dt_bound))

    # Bisect
    for _ in np.arange(0, steps):
        mid_z = (low_z + high_z) / 2.
        r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z)
        pressure_mid = vortex.pressure(r_conserving_angular_momentum, mid_z)
        delta_pressure_mid = pressure_mid - reference_pressure_parcel
        pos_pressure = np.where(delta_pressure_mid > 0.)
        neg_pressure = np.where(delta_pressure_mid < 0.)

        low_z[pos_pressure] = mid_z[pos_pressure]
        high_z[neg_pressure] = mid_z[neg_pressure]

        mid_z_new = (low_z + high_z) / 2.

    r_conserving_angular_momentum = angular_momentum_radius_from_height(vortex, angular_momentum_parcel, mid_z_new)

    return r_conserving_angular_momentum, mid_z_new


def available_elastic_energy(vortex, entropy, p, r, z):
    temperature_in_situ = temperature_from_entropy(vortex, entropy, p)
    density_in_situ = p/(vortex.Rd * temperature_in_situ)
    enthalpy_in_situ = vortex.cp * temperature_in_situ
    enthalpy_at_pref = vortex.cp * lifted_temperature(vortex, temperature_in_situ, p, vortex.pressure(r, z))
    pressure_head = (vortex.pressure(r, z) - p)/density_in_situ
    return enthalpy_in_situ - enthalpy_at_pref + pressure_head


def available_potential_energy(vortex, M, eta, r, z):
    r_ref, z_ref = reference_position(vortex, M, eta)
    M_terms = 0.5 * M**2 * (1. / r ** 2 - 1. / r_ref ** 2) + 0.125 * vortex.f ** 2 * (r ** 2 - r_ref ** 2)
    geopotential_terms = vortex.geopotential(z) - vortex.geopotential(z_ref)
    temperature_at_p0 = temperature_from_entropy(vortex, eta, vortex.pressure(r, z))
    enthalpy_at_p0 = vortex.cp * temperature_at_p0
    enthalpy_at_reference = vortex.cp * lifted_temperature(vortex, temperature_at_p0, vortex.pressure(r, z), vortex.pressure(r_ref, z_ref))
    return M_terms + geopotential_terms + enthalpy_at_p0 - enthalpy_at_reference


def pi_e(vortex, M, eta, r, z):
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    r_ref, z_ref = reference_position(vortex, M, eta)
    M_terms = 0.5 * M**2 * (1. / r_mu ** 2 - 1. / r_ref ** 2) + 0.125 * vortex.f ** 2 * (r_mu ** 2 - r_ref ** 2)
    geopotential_terms = vortex.geopotential(z_mu) - vortex.geopotential(z_ref)
    temperature_at_p0 = temperature_from_entropy(vortex, eta, vortex.pressure(r, z))
    enthalpy_at_p0 = vortex.cp * temperature_at_p0
    enthalpy_at_reference = vortex.cp * lifted_temperature(vortex, temperature_at_p0, vortex.pressure(r, z), vortex.pressure(r_ref, z_ref))
    return M_terms + geopotential_terms + enthalpy_at_p0 - enthalpy_at_reference


def pi_k(vortex, M, r, z):
    r_mu, z_mu = position_at_isobaric_surface(vortex, M, vortex.pressure(r, z))
    M_terms = 0.5 * M**2 * (1. / r ** 2 - 1. / r_mu ** 2) + 0.125 * vortex.f ** 2 * (r ** 2 - r_mu ** 2)
    geopotential_terms = vortex.geopotential(z) - vortex.geopotential(z_mu)
    return M_terms + geopotential_terms


def available_energy(vortex, M, eta, p, r, z):
    aee = available_elastic_energy(vortex, eta, p, r, z)
    ape = available_potential_energy(vortex, M, eta, r, z)
    return aee + ape


def available_potential_energy_perturbations_M_entropy(vortex, r, z):
    base_M = vortex.angular_momentum(r, z)
    base_entropy = vortex.entropy(r, z)
    all_M = vortex.gridded_variable(vortex.angular_momentum)
    all_entropy = vortex.gridded_variable(vortex.entropy)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_entropy.min(), all_entropy.max(), 100))
    ae_M_eta = available_potential_energy(vortex, M_grid, entropy_grid, r, z)
    return M_grid-base_M, entropy_grid-base_entropy, ae_M_eta


def available_potential_energy_perturbations_mu_pressure(vortex, r, z):
    base_mu = vortex.mu(r, z)
    base_pressure = vortex.pressure(r, z)
    all_M = vortex.gridded_variable(vortex.angular_momentum)
    all_entropy = vortex.gridded_variable(vortex.entropy)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_entropy.min(), all_entropy.max(), 100))
    mu_grid = M_grid**2
    r_ref, z_ref = reference_position(vortex, M_grid, entropy_grid)
    p_ref_grid = vortex.pressure(r_ref, z_ref)
    ae_M_eta = available_potential_energy(vortex, M_grid, entropy_grid, r, z)
    return base_mu-mu_grid, base_pressure-p_ref_grid, ae_M_eta


def available_potential_energy_perturbations_r_z(vortex, r, z):
    r_grid, z_grid = vortex.grid()
    ae_r_z = available_potential_energy(vortex, vortex.gridded_variable(vortex.angular_momentum), vortex.gridded_variable(vortex.entropy), r, z)
    return r_grid-r, z_grid-z, ae_r_z
