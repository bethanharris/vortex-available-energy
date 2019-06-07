from derived_variables import *
from utils import r_z_grid


def lifted_temperature(old_temperature, old_pressure, new_pressure):
    new_temperature = old_temperature * (new_pressure / old_pressure) ** (Rd / cp)
    return new_temperature


def temperature_from_entropy(entropy, pressure):
    potential_temperature = np.exp(entropy/cp)
    temperature = potential_temperature*(pressure/p0)**(Rd/cp)
    return temperature

def potential_temperature_from_entropy(entropy):
    potential_temperature = np.exp(entropy/cp)
    return potential_temperature

def entropy_from_temperature(temperature, pressure):
    potential_temperature = temperature*(p0/pressure)**(Rd/cp)
    entropy = cp*np.log(potential_temperature)
    return entropy


def angular_momentum_radius_from_height(angular_momentum_parcel, z):
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
        angular_momentum_mid = angular_momentum(mid_r, z)
        delta_angular_momentum_mid = angular_momentum_mid - angular_momentum_parcel
        pos_angular_momentum = np.where(delta_angular_momentum_mid > 0.)
        neg_angular_momentum = np.where(delta_angular_momentum_mid < 0.)

        low_r[neg_angular_momentum] = mid_r[neg_angular_momentum]
        high_r[pos_angular_momentum] = mid_r[pos_angular_momentum]

        mid_r_new = (low_r + high_r) / 2.
    return mid_r_new


def reference_position(angular_momentum_parcel, entropy_parcel):
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
        r_conserving_angular_momentum = angular_momentum_radius_from_height(angular_momentum_parcel, mid_z)
        entropy_mid = entropy(r_conserving_angular_momentum, mid_z)
        delta_entropy_mid = entropy_mid - entropy_parcel
        pos_entropy = np.where(delta_entropy_mid > 0.)
        neg_entropy = np.where(delta_entropy_mid < 0.)

        low_z[neg_entropy] = mid_z[neg_entropy]
        high_z[pos_entropy] = mid_z[pos_entropy]

        mid_z_new = (low_z + high_z) / 2.

    r_conserving_angular_momentum = angular_momentum_radius_from_height(angular_momentum_parcel, mid_z_new)

    return r_conserving_angular_momentum, mid_z_new


def available_elastic_energy(eta, p, r, z):
    temperature_in_situ = temperature_from_entropy(eta, p)
    density_in_situ = p/(Rd * temperature_in_situ)
    enthalpy_in_situ = cp * temperature_in_situ
    enthalpy_at_pref = cp * lifted_temperature(temperature_in_situ, p, pressure(r, z))
    pressure_head = (pressure(r, z) - p)/density_in_situ
    return enthalpy_in_situ - enthalpy_at_pref + pressure_head


def available_potential_energy(M, eta, r, z):
    r_ref, z_ref = reference_position(M, eta)
    M_terms = 0.5 * M**2 * (1. / r ** 2 - 1. / r_ref ** 2) + 0.125 * f ** 2 * (r ** 2 - r_ref ** 2)
    geopotential_terms = geopotential(z) - geopotential(z_ref)
    temperature_at_p0 = temperature_from_entropy(eta, pressure(r, z))
    enthalpy_at_p0 = cp * temperature_at_p0
    enthalpy_at_reference = cp * lifted_temperature(temperature_at_p0, pressure(r, z), pressure(r_ref, z_ref))
    return M_terms + geopotential_terms + enthalpy_at_p0 - enthalpy_at_reference


def available_energy(M, eta, p, r, z):
    aee = available_elastic_energy(eta, p, r, z)
    ape = available_potential_energy(M, eta, r, z)
    return aee + ape


def available_energy_perturbations_M_eta(r, z):
    base_M = angular_momentum(r, z)
    base_entropy = entropy(r, z)
    r_grid, z_grid = r_z_grid()
    all_M = angular_momentum(r_grid, z_grid)
    all_entropy = entropy(r_grid, z_grid)
    M_grid, entropy_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_entropy.min(), all_entropy.max(), 100))
    ae_M_eta = available_energy(M_grid, entropy_grid, pressure(r, z), r, z)
    return M_grid-base_M, entropy_grid-base_entropy, ae_M_eta


def available_energy_perturbations_M_pressure(r, z):
    base_M = angular_momentum(r, z)
    base_pressure = pressure(r, z)
    r_grid, z_grid = r_z_grid()
    all_M = angular_momentum(r_grid, z_grid)
    all_pressure = pressure(r_grid, z_grid)
    M_grid, pressure_grid = np.meshgrid(np.linspace(all_M.min(), all_M.max(), 100), np.linspace(all_pressure.min(), all_pressure.max(), 100))
    ae_M_pressure = available_energy(M_grid, np.ones_like(M_grid)*entropy(r, z), pressure_grid, r, z)
    return M_grid-base_M, pressure_grid-base_pressure, ae_M_pressure


def available_energy_perturbations_eta_pressure(r, z):
    base_eta = entropy(r, z)
    base_pressure = pressure(r, z)
    r_grid, z_grid = r_z_grid()
    all_eta = entropy(r_grid, z_grid)
    all_pressure = pressure(r_grid, z_grid)
    eta_grid, pressure_grid = np.meshgrid(np.linspace(all_eta.min(), all_eta.max(), 100), np.linspace(all_pressure.min(), all_pressure.max(), 100))
    ae_eta_pressure = available_energy(np.ones_like(eta_grid)*angular_momentum(r, z), eta_grid, pressure_grid, r, z)
    return eta_grid-base_eta, pressure_grid-base_pressure, ae_eta_pressure


def available_energy_perturbations_r_z(r, z):
    r_grid, z_grid = r_z_grid()
    ae_r_z = available_energy(angular_momentum(r_grid, z_grid), entropy(r_grid, z_grid), r, z)
    return r_grid-r, z_grid-z, ae_r_z
