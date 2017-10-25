#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance, dt_float

def c_act(crd):
    ''' Toy model for the surface wave velocity to be recovered. '''
    c = np.full_like(crd, 4000, dtype=dt_float)
    x1 = np.array((66,14.5), dtype=dt_latlon)
    gcd_x1 = great_circle_distance(crd, x1)
    c += 60*np.exp(-gcd_x1/40000)
    x2 = np.array((67.5,20), dtype=dt_latlon)
    gcd_x2 = great_circle_distance(crd, x2)
    c -= 40*np.exp(-gcd_x2/65000)
    return c

if __name__ == '__main__':
    from file_IO import read_station_file
    from gptt import cos_central_angle, gauss_kernel, great_circle_path, line_element, dt_xyz, to_xyz
    from scipy.integrate import simps
    from matplotlib import pyplot as plt

    # Read coordinates of the NORSA Array
    stations = read_station_file('../dat/stations.dat')

    # Indices for all combinations of stations with duplicates dropped
    idx, idy = np.tril_indices(stations.size, -1)

    # Determine how fine great circle segments are going to be sampled
    central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))
    ds = central_angle.min()/4 # Spacing in [rad]
    # Number of samples per path
    npts = np.round(central_angle/ds, 0).astype(int)
    # Indices for array slicing
    index = npts.cumsum() - npts

    # Allocate array for sampling points
    # FIXME There is quite a bunch of duplicates
    points = np.empty(npts.sum(), dtype=dt_xyz)
    # Allocate memory for path parametrization
    ts = np.empty_like(points, dtype=dt_float)

    # Calculate sampling points and parametrization
    it = np.nditer( (stations[idx], stations[idy], central_angle, npts, index) )
    for st1, st2, ca, n, i in it:
        slc = slice(i,i+n,1)
        t = np.linspace(0, ca, n)
        ts[slc] = t
        points[slc] = great_circle_path(st1, st2, t)

    # Actual velocity model
    c = c_act(points)
    # A priori assumptions
    # TODO add a grid of design points
    lat, lon = np.mgrid[64.5:69.5:40j, 11.5:23:40j]
    design_points = np.rec.fromarrays( (lat, lon), dtype=dt_latlon)
    points = np.concatenate( (points, to_xyz(design_points).flatten()) )
    mu_C = np.full_like(points, 4000, dtype=dt_float)
    cov_CC = 40**2*gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], sigma=9000**2)

    # Calculate actual and prior travel times
    it = np.nditer( (stations[idx], stations[idy], index, npts, None, None), \
                    op_dtypes=(None, None, None, None, dt_float, dt_float) )
    for st1, st2, i, n, T12, mu_T12_pri in it:
        slc = slice(i,i+n,1)
        c12 = c[slc]
        mu_C12 = mu_C[slc]
        t12 = ts[slc]
        T12[...] = simps(6371000/c12, t12)
        mu_T12_pri[...] = simps(6371000/mu_C12, t12)

    T_act, mu_T_pri  = it.operands[-2:]
    # Pseudo travel time observations
    D = T_act + np.random.normal(loc=0, scale=0.01, size=T_act.size).astype(dt_float)


    it = np.nditer((stations[idx], stations[idy], D, index, npts))
    for st1, st2, D12, i, n in it:
        slc = slice(i,i+n,1)
        t12 = ts[slc]
        mu_T12 = simps(6371000/mu_C[slc], t12)
        cor_CT = -simps(cov_CC[:,slc]*6371000/mu_C[slc]**2, t12, axis=-1).astype(dt_float)
        var_TT = -simps(cor_CT[slc]*6371000/mu_C[slc]**2, t12, axis=-1)
        var_TT+= 0.01**2

        mu_C += cor_CT/var_TT*(D12 - mu_T12)
        cov_CC -= np.dot(cor_CT[:,np.newaxis],cor_CT[np.newaxis,:])/var_TT


    it = np.nditer( (stations[idx], stations[idy], index, npts, None), \
                    op_dtypes=(None, None, None, None, float) )
    for st1, st2, i, n, T12 in it:
        slc = slice(i,i+n,1)
        c12 = mu_C[slc]
        t12 = ts[slc]
        T12[...] = simps(6371000/c12, t12)
    mu_T_pst = it.operands[-1]

    print np.sum((mu_T_pri - T_act)**2)
    print np.sum((mu_T_pst - T_act)**2)

    lat, lon = np.mgrid[64.5:69.5:41j, 11.5:23:41j]
    C = mu_C[-40*40:].reshape( (40,40) )
    plt.pcolormesh(lat, lon, C, vmin=3960, vmax=4060)
    plt.scatter(stations['lat'], stations['lon'])
    plt.figure()
    plt.pcolormesh(lat, lon, c_act(design_points), vmin=3960, vmax=4060)
    plt.scatter(stations['lat'], stations['lon'])
    plt.show()

