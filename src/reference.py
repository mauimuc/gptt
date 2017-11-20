#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Generates pseudo data from the reference model.
To guarantee reproducibility do NOT run that script unless for a very good reason. '''

import numpy as np
from gptt import dt_latlon, great_circle_distance, cos_central_angle, read_station_file, r_E


def c_act(crd):
    ''' Toy model for the wave velocity [m/s] to be recovered. '''
    c = np.full_like(crd, 4e3, dtype=np.float)
    x1 = np.array((66.3,14.6), dtype=dt_latlon)
    gcd_x1 = great_circle_distance(crd, x1)
    c += 100*np.exp(-gcd_x1/40e3)
    x2 = np.array((67.5,20), dtype=dt_latlon)
    gcd_x2 = great_circle_distance(crd, x2)
    c -= 80*np.exp(-gcd_x2/65e3)
    return c

class GCP(object):
    ''' Parametrization of the great circle segment '''

    def __init__(self, st1, st2):
        ''' Pass two station locations '''
        cos_ca = cos_central_angle(st1, st2) # Central angle
        self.ca = np.arccos(cos_ca)
        sin_ca = np.sqrt(1 - cos_ca**2)
        self.x1, self.y1, self.z1 = self._to_cartesian(st1) # Cartesian coordinates
        x2, y2, z2 = self._to_cartesian(st2) # Cartesian coordinates
        # Pre-process point w
        self.wx = (x2 - self.x1*cos_ca)/sin_ca
        self.wy = (y2 - self.y1*cos_ca)/sin_ca
        self.wz = (z2 - self.z1*cos_ca)/sin_ca

    @staticmethod
    def _to_cartesian(st):
        ''' Convert station location to Cartesian coordinates (on the unit sphere) '''
        t = np.deg2rad(90 - st['lat']) # to co-latitude [rad]
        p = np.deg2rad(st['lon'])
        return np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)

    def __call__(self, t):
        ''' Pass t in 0 to 2*pi
        Returns coordinates of the great circle in latitude and longitude '''
        # Allocate
        res = np.empty_like(t, dtype=dt_latlon)
        # Pre-process
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        # Great circle w.r.t. Cartesian coordinates
        x = self.x1*cos_t + self.wx*sin_t
        y = self.y1*cos_t + self.wy*sin_t
        z = self.z1*cos_t + self.wz*sin_t
        # To latitude and longitude
        res['lon'] = np.rad2deg(np.arctan2(y, x))
        res['lat'] = 90 - np.rad2deg(np.arccos(z))
        return res

# Observational error
err_obs = 0.5 # Standard deviation [s]

# Structured array dtype to store pseudo records
dt_obs = np.dtype( [('stnm1', 'S5'), ('stnm2', 'S5'), ('tt', np.float32), ('err', np.float32)] )

if __name__ == '__main__':
    from scipy.integrate import quad

    # Read station file
    stations = read_station_file('../dat/stations.dat')

    # To keep compute time moderate just consider halve the stations
    stations = stations[:20]

    # Indices for all combinations of stations with duplicates dropped
    idx, idy = np.tril_indices(stations.size, -1)

    # Numpy iterator to go through all combinations
    it = np.nditer(op=(idx, idy, None), op_dtypes=(np.int, np.int, dt_obs))

    for i, j, rec in it:
        gcp = GCP(stations[i], stations[j]) # Great circle parametrization
        integrand = lambda t: r_E/c_act(gcp(t)) # The function to be integrated
        rec['tt'] = quad(integrand, 0, gcp.ca)[0] # Calculate travel time [s]
        rec['err'] = np.random.normal(loc=0, scale=err_obs) # Draw an error [s]
        rec['stnm1'] = stations[i]['stnm'] # Store station name 1
        rec['stnm2'] = stations[j]['stnm'] # Store station name 2

    pseudo_data = it.operands[-1] # Retrieve data from iterator

    # Write data to file
    header = ' '.join(dt_obs.names)
    fmt = '%5s %5s %7.3f %7.3f' # Formatter string
    np.savetxt('../dat/pseudo_data.dat', pseudo_data, header=header, fmt=fmt)

