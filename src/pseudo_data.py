#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Generates pseudo data from a reference model.
To guarantee reproducibility do NOT run that script unless for a very good reason. '''

import numpy as np
from gptt import dt_latlon, cos_central_angle, read_station_file, r_E
from scipy.integrate import quad
from example import c_act

class GCP(object):

    def __init__(self, st1, st2):
        cos_ca = cos_central_angle(st1, st2)
        self.ca = np.arccos(cos_ca)
        sin_ca = np.sqrt(1 - cos_ca**2)
        self.x1, self.y1, self.z1 = self._to_cartesian(st1)
        x2, y2, z2 = self._to_cartesian(st2)
        self.wx = (x2 - self.x1*cos_ca)/sin_ca
        self.wy = (y2 - self.y1*cos_ca)/sin_ca
        self.wz = (z2 - self.z1*cos_ca)/sin_ca

    @staticmethod
    def _to_cartesian(st):
        t = np.deg2rad(90 - st['lat'])
        p = np.deg2rad(st['lon'])
        return np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)

    def __call__(self, t):
        # FIXME It were better to have the parametrization for the interval [0,1]
        res = np.empty_like(t, dtype=dt_latlon)

        sin_t = np.sin(t)
        cos_t = np.cos(t)

        x = self.x1*cos_t + self.wx*sin_t
        y = self.y1*cos_t + self.wy*sin_t
        z = self.z1*cos_t + self.wz*sin_t

        res['lon'] = np.rad2deg(np.arctan2(y, x))
        res['lat'] = 90 - np.rad2deg(np.arccos(z))
        return res


# Read station file
stations = read_station_file('../dat/stations.dat')

# To keep compute time moderate consider just halve
stations = stations[::2]

# Indices for all combinations of stations with duplicates dropped
idx, idy = np.tril_indices(stations.size, -1)

# A record dtype
dt_obs = np.dtype( [('stnm1', 'S5'), ('stnm2', 'S5'), ('tt', np.float32), ('err', np.float32)] )

# Numpy iterator to go through all combinations
it = np.nditer(op=(idx, idy, None), op_dtypes=(np.int, np.int, dt_obs))

for i, j, rec in it:
    gcp = GCP(stations[i], stations[j]) # Great circle parametrization
    integrand = lambda t: r_E/c_act(gcp(t)) # Function to be integrated
    rec['tt'] = quad(integrand, 0, gcp.ca)[0] # Calculate travel time
    rec['err'] = np.random.normal(loc=0, scale=0.02) # Draw an error
    rec['stnm1'] = stations[i]['stnm'] # Store station name 1
    rec['stnm2'] = stations[j]['stnm'] # Store station name 2

pseudo_data = it.operands[-1] # Retrieve data from iterator

# Write data to file
header = ' '.join(dt_obs.names)
fmt = '%5s %5s %7.3f %7.3f' # Formatter string
np.savetxt('../dat/pseudo_data.dat', pseudo_data, header=header, fmt=fmt)

