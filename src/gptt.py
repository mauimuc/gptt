#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Module to store routines for Bayesian surface wave tomography '''

import numpy as np

# Structured arrays to hold coordinates
dt_latlon = np.dtype( [('lat', 'float'), ('lon', 'float')] )
dt_xyz = np.dtype( [('x', 'float'), ('y', 'float'), ('z', 'float')] )
dt_rtp = np.dtype( [('r', 'float'), ('t', 'float'), ('p', 'float')] )


def to_latlon(crd):
    ''' Transform intro latitude and longitude (degrees) '''
    dtype = np.result_type(crd)
    res = np.empty_like(crd, dtype=dt_latlon)
    if dtype == dt_latlon:
        res = crd
    elif dtype == dt_rtp:
        res['lat'] = 90 - np.rad2deg(crd['t'])
        res['lon'] = np.rad2deg(crd['p'])
    elif dtype == dt_xyz:
        r = np.sqrt(crd['x']**2 + crd['y']**2 + crd['z']**2)
        res['lon'] = np.rad2deg(np.arctan2(crd['y'], crd['x']))
        res['lat'] = 90 - np.rad2deg(np.arccos(crd['z']/r))
    else:
        raise NotImplementedError
    return res

def to_rtp(crd, r=6371000):
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

def to_xyz(crd, r=6371000):
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

def _inner(crd1, crd2):
    a = to_xyz(crd1)
    b = to_xyz(crd2)
    return a['x']*b['x'] + a['y']*b['y'] + a['z']*b['z']


def great_circle_distance(crd1, crd2):
    r1 = to_rtp(crd1)['r']
    r2 = to_rtp(crd2)['r']
    cos_sigma = cos_central_angle(crd1, crd2)
    # XXX Due to rounding errors cos_sigma > 1 -> NaN
    cos_sigma[cos_sigma>1.] = 1.
    return np.arccos(cos_sigma)*np.sqrt(r1*r2)

def cos_central_angle(crd1, crd2):
    r1 = to_rtp(crd1)['r']
    r2 = to_rtp(crd2)['r']
    return _inner(crd1, crd2)/r1/r2

def great_circle_path(u, v, t):
    cos_sigma = cos_central_angle(u, v)
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

def line_element(u, v, t):
    cos_sigma = cos_central_angle(u, v)
    sin_sigma = np.sqrt(1 - cos_sigma**2)
    u = to_xyz(u)
    v = to_xyz(v)
    ux, uy, uz = u['x'], u['y'], u['z']
    vx, vy, vz = v['x'], v['y'], v['z']
    wx = (vx - ux*cos_sigma)/sin_sigma
    wy = (vy - uy*cos_sigma)/sin_sigma
    wz = (vz - uz*cos_sigma)/sin_sigma
    tx = wx*np.cos(t) - ux*np.sin(t)
    ty = wy*np.cos(t) - uy*np.sin(t)
    tz = wz*np.cos(t) - uz*np.sin(t)
    return np.sqrt(tx**2 + ty**2 + tz**2)

def gauss_kernel(crd1, crd2, sigma):
    d = great_circle_distance(crd1, crd2)
    return np.exp(-d**2/sigma)

