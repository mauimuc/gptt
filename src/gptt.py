#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Module to store routines for Bayesian surface wave tomography '''

import numpy as np
from scipy.integrate import simps

r_E = 6371000.

# Structured arrays to hold coordinates
dt_float = np.float64
dt_latlon = np.dtype( [('lat', dt_float), ('lon', dt_float)] )
dt_xyz = np.dtype( [('x', dt_float), ('y', dt_float), ('z', dt_float)] )
dt_rtp = np.dtype( [('r', dt_float), ('t', dt_float), ('p', dt_float)] )


def to_latlon(crd):
    ''' Transform intro latitude and longitude (degrees) '''
    dtype = np.result_type(crd)
    res = np.empty_like(crd, dtype=dt_latlon)
    if dtype == dt_latlon:
        res = crd
    elif dtype == dt_rtp:
        res['lat'] = 90. - np.rad2deg(crd['t'])
        res['lon'] = np.rad2deg(crd['p'])
    elif dtype == dt_xyz:
        r = np.sqrt(crd['x']**2 + crd['y']**2 + crd['z']**2)
        res['lon'] = np.rad2deg(np.arctan2(crd['y'], crd['x']))
        res['lat'] = 90. - np.rad2deg(np.arccos(crd['z']/r))
    else:
        raise NotImplementedError
    return res

def to_rtp(crd, r=r_E):
    dtype = np.result_type(crd)
    res = np.empty_like(crd, dtype=dt_rtp)
    if dtype == dt_latlon:
        res['r'] = r
        res['t'] = np.deg2rad(90 - crd['lat'])
        res['p'] = np.deg2rad(crd['lon'])
    elif dtype == dt_rtp:
        res = crd
    elif dtype == dt_xyz:
        res['r'] = np.sqrt(crd['x']**2 + crd['y']**2 + crd['z']**2)
        res['p'] = np.arctan2(crd['y'], crd['x'])
        res['t'] = np.arccos(crd['z']/res['r'])
    else:
        raise NotImplementedError
    return res

def to_xyz(crd, r=r_E):
    ''' Transform intro Cartesian coordinates '''
    dtype = np.result_type(crd)
    res = np.empty_like(crd, dtype=dt_xyz)
    if dtype == dt_latlon:
        t = np.deg2rad(90 - crd['lat'])
        p = np.deg2rad(crd['lon'])
        res['x'] = r*np.sin(t)*np.cos(p)
        res['y'] = r*np.sin(t)*np.sin(p)
        res['z'] = r*np.cos(t)
        return res
    elif dtype == dt_rtp:
        res['x'] = crd['r']*np.sin(crd['t'])*np.cos(crd['p'])
        res['y'] = crd['r']*np.sin(crd['t'])*np.sin(crd['p'])
        res['z'] = crd['r']*np.cos(crd['t'])
    elif dtype == dt_xyz:
        res = crd
    else:
        raise NotImplementedError
    return res


def great_circle_distance(crd1, crd2):
    cos_sigma = cos_central_angle(to_latlon(crd1), to_latlon(crd2))
    # XXX Due to rounding errors cos_sigma > 1 -> NaN
    cos_sigma = np.where(cos_sigma>1., 1., cos_sigma)
    return np.arccos(cos_sigma)*r_E


def cos_central_angle(crd1, crd2):
    ''' Coordinates as numpy structured type in degrees '''
    phi1 = np.deg2rad(crd1['lat']) # latitude in rad
    phi2 = np.deg2rad(crd2['lat']) # latitude in rad
    cos_Delta_lam = np.cos(np.deg2rad(crd1['lon'] - crd2['lon']))
    return np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*cos_Delta_lam


def great_circle_path(u, v, t):
    cos_sigma = cos_central_angle(to_latlon(u), to_latlon(v))
    sin_sigma = np.sqrt(1 - cos_sigma**2)
    u = to_xyz(u)
    v = to_xyz(v)
    ux, uy, uz = u['x'], u['y'], u['z']
    vx, vy, vz = v['x'], v['y'], v['z']
    wx = (vx - ux*cos_sigma)/sin_sigma
    wy = (vy - uy*cos_sigma)/sin_sigma
    wz = (vz - uz*cos_sigma)/sin_sigma
    tx = ux*np.cos(t) + wx*np.sin(t)
    ty = uy*np.cos(t) + wy*np.sin(t)
    tz = uz*np.cos(t) + wz*np.sin(t)
    return np.rec.fromarrays( (tx, ty, tz), dtype=dt_xyz)


def gauss_kernel(crd1, crd2, tau, ell):
    d = great_circle_distance(crd1, crd2)
    return tau**2*np.exp(-(d/ell)**2).astype(dt_float)



class StationPair(object):
    def __init__(self, st1, st2, indices, error):
        self.st1 = st1
        self.st2 = st2
        self.d = None # Observed value
        self.error = error # Standard deviation
        self.indices = indices # Discretization

    @property
    def npts(self):
        return self.indices.size

    @property
    def cos_central_angle(self):
        return cos_central_angle(self.st1, self.st2)

    @property
    def spacing(self):
        return self.central_angle/(self.npts - 1)

    @property
    def central_angle(self):
        return np.arccos(self.cos_central_angle)


    @property
    def great_circle_path(self):
        st1 = np.array( (self.st1['lat'], self.st1['lon']), dtype=dt_latlon)
        st2 = np.array( (self.st2['lat'], self.st2['lon']), dtype=dt_latlon)
        t = np.linspace(0, self.central_angle, self.npts)
        return to_latlon(great_circle_path(st1, st2, t))

    def T(self, c):
        ''' Pass a velocity model and return the according travel time '''
        return simps(r_E/c[self.indices], dx=self.spacing)

    def var_DD(self, mean, cov):
        ''' Calculate the variance by passing the model's mean and covariance '''
        indices = np.reshape(self.indices, (-1,1)) # Well, indexing is cryptic
        cor = simps(r_E*cov[indices.T, indices]/mean[self.indices]**2, \
                    dx=self.spacing, axis=-1)
        return simps(r_E*cor/mean[self.indices]**2, dx=self.spacing) + self.error**2

    def cor_CT(self, mean, cov):
        ''' Pass a model's mean and covariance and return correlations amongst
            model and travel time '''
        return -simps(r_E*cov[:,self.indices]/mean[self.indices]**2, \
                      dx=self.spacing, axis=-1)

