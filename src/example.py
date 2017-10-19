#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance

def c_act(crd):
    ''' Toy model for the surface wave velocity to be recovered. '''
    c = np.full_like(crd, 4000, dtype='float')
    x1 = np.array((66,14.5), dtype=dt_latlon)
    gcd_x1 = great_circle_distance(crd, x1)
    c += 60*np.exp(-gcd_x1/40000)
    x2 = np.array((67.5,20), dtype=dt_latlon)
    gcd_x2 = great_circle_distance(crd, x2)
    c -= 40*np.exp(-gcd_x2/65000)
    return c

if __name__ == '__main__':
    from file_IO import read_station_file
    from gptt import cos_central_angle, gauss_kernel, great_circle_path, line_element, dt_xyz
    from scipy.integrate import simps
    from matplotlib import pyplot as plt

    # Read coordinates of the NORSA Array
    stations = read_station_file('../dat/stations.dat')

    # Indices for all combinations of stations with duplicates dropped
    idx, idy = np.tril_indices(stations.size, -1)

    central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))


    ds = central_angle.min()/4
    num = np.round(central_angle/ds, 0).astype(int)
    points = np.empty( num.sum(), dtype=dt_xyz)
    ts = np.empty_like(points, dtype='float')
    les = np.empty_like(points, dtype='float')


    it = np.nditer( (stations[idx], stations[idy], central_angle, num, num.cumsum(), None), \
                    op_dtypes=(None, None, None, None, None, object))
    for st1, st2, ca, npts, i, slc in it:
        s = slice(i-npts,i,1)
        t = np.linspace(0, ca, npts)
        slc[...] = s
        les[s] = line_element(st1, st2, t)
        ts[s] = t
        points[s] = great_circle_path(st1, st2, t)

    slcs = it.operands[-1]
    c = c_act(points)

    mu_C = np.full_like(points, 4000, dtype='float')
    cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], sigma=20000**2)

    # Calculation of travel times
    it = np.nditer( (stations[idx], stations[idy], range(slcs.size), None), \
                    op_dtypes=(None, None, None, float) )
    for st1, st2, i, T12 in it:
        slc = slcs[i]
        c12 = c[slc]
        le12 = les[slc]
        t12 = ts[slc]
        T12[...] = simps(le12/c12, t12)

        cor_CT12 = simps(cov_CC[:,slc]*le12/mu_C[slc]**2, t12)

    ## Calculation of correlations




