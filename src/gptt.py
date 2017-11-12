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
dt_latlon = np.dtype( [('lat', np.float), ('lon', np.float)] )


def read_station_file(fname):
    ''' Reads the station file and returns a structured array '''
    my_dt = [ ('stnm', 'S5'), ('lat', np.float), ('lon', np.float), ('elv', np.float) ]
    return np.genfromtxt(fname, dtype=my_dt)


def great_circle_distance(crd1, crd2):
    ''' pass coordinates crd1 und crd1 as a structured array '''
    cos_sigma = cos_central_angle(crd1, crd2)
    # XXX Due to rounding errors cos_sigma > 1 -> NaN
    cos_sigma = np.where(cos_sigma>1., 1., cos_sigma)
    return np.arccos(cos_sigma)*r_E


def cos_central_angle(crd1, crd2):
    ''' Coordinates as numpy structured type in degrees '''
    phi1 = np.deg2rad(crd1['lat']) # latitude in rad
    phi2 = np.deg2rad(crd2['lat']) # latitude in rad
    cos_Delta_lam = np.cos(np.deg2rad(crd1['lon'] - crd2['lon']))
    return np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*cos_Delta_lam


def gauss_kernel(crd1, crd2, tau, ell):
    d = great_circle_distance(crd1, crd2)
    return tau**2*np.exp(-(d/ell)**2)


class StationPair(object):
    def __init__(self, st1, st2, indices, error):
        self.st1 = st1
        self.st2 = st2
        self.d = None # Observed value
        self.error = error # Standard deviation
        self.indices = indices # Discretization

    def __str__(self):
        return '%5s -- %-5s' % (self.st1['stnm'], self.st2['stnm'])

    @property
    def npts(self):
        return len(self.indices)

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
    def _xyz1(self):
        t = np.deg2rad(90 - self.st1['lat'])
        p = np.deg2rad(self.st1['lon'])
        return np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)
    @property
    def _xyz2(self):
        t = np.deg2rad(90 - self.st2['lat'])
        p = np.deg2rad(self.st2['lon'])
        return np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)

    @property
    def sin_central_angle(self):
        return np.sqrt(1 - self.cos_central_angle**2)

    @property
    def great_circle_path(self):
        x1, y1, z1 = self._xyz1
        x2, y2, z2 = self._xyz2
        wx = (x2 - x1*self.cos_central_angle)/self.sin_central_angle
        wy = (y2 - y1*self.cos_central_angle)/self.sin_central_angle
        wz = (z2 - z1*self.cos_central_angle)/self.sin_central_angle
        t = np.linspace(0, self.central_angle, self.npts)
        tx = x1*np.cos(t) + wx*np.sin(t)
        ty = y1*np.cos(t) + wy*np.sin(t)
        tz = z1*np.cos(t) + wz*np.sin(t)
        res = np.empty_like(t, dtype=dt_latlon)
        res['lon'] = np.rad2deg(np.arctan2(ty, tx))
        res['lat'] = 90 - np.rad2deg(np.arccos(tz))
        return res

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


def mu_T(pairs, mean):
    ''' Calculate mean travel time '''
    N = len(pairs)
    res = np.empty(N)
    for i in range(N):
        res[i] = simps(r_E/mean[pairs[i].indices], dx=pairs[i].spacing)
    return res

def cov_TT(pairs, mean, cov):
    N = len(pairs)
    res = np.empty((N,N))
    for i in range(N):
        ds_i = pairs[i].spacing
        idx_i = pairs[i].indices
        cor = -simps(cov[:,idx_i]*r_E/mean[idx_i]**2, dx=ds_i, axis=-1)
        for j in range(i, N):
            ds_j = pairs[j].spacing
            idx_j = pairs[j].indices
            res[i,j] = -simps(cor[idx_j]*r_E/mean[idx_j]**2, dx=ds_j)
            if i!=j:
                res[j,i] = res[i,j]
    return res

def misfit(d, mean, cov):
    L = np.linalg.cholesky(cov)
    return (np.linalg.solve(L, d - mean)**2).sum()

