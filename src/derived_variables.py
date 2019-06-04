import numpy as np
from construct_vortex import *


p0 = 100000.
cp = 1004.5


def angular_momentum(r, z):
    return r*azimuthal_wind(r, z) + 0.5*f*r**2


def potential_temperature(r, z):
    return temperature(r, z) * (p0/pressure(r ,z))**(Rd/cp)


def entropy(r, z):
    return cp * np.log(potential_temperature(r, z))


def geopotential(z):
    return g * z
