#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Module to store routines for Bayesian surface wave tomography '''

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

dt_latlon = np.dtype( [('lat', 'float'), ('lon', 'float')] )
dt_xyz = np.dtype( [('x', 'float'), ('y', 'float'), ('z', 'float')] )
dt_rtp = np.dtype( [('r', 'float'), ('t', 'float'), ('p', 'float')] )


def to_latlon(crd):
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
    cos_sigma = _inner(crd1, crd2)/r1/r2
    return np.arccos(cos_sigma)*np.sqrt(r1*r2)



class Point(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self): pass
    @abstractproperty
    def x(self): pass
    @abstractproperty
    def y(self): pass
    @abstractproperty
    def z(self): pass
    @abstractproperty
    def r(self): pass
    @abstractproperty
    def t(self): pass
    @abstractproperty
    def p(self): pass
    @property
    def lat(self):
        return 90 - self.t*180/np.pi
    @property
    def lon(self):
        return self.p*180/np.pi
    def __repr__(self):
        return 'lat=%f lon=%f' % (self.lat, self.lon)

class Point_latlon(Point):
    def __init__(self, lat, lon, rad=6371000):
        self._lat = lat
        self._lon = lon
        self._rad = rad
    @property
    def r(self):
        return float(self._rad)
    @property
    def t(self):
        return (90-self._lat)*np.pi/180
    @property
    def p(self):
        return self._lon*np.pi/180
    @property
    def x(self):
        return self.r*np.sin(self.t)*np.cos(self.p)
    @property
    def y(self):
        return self.r*np.sin(self.t)*np.sin(self.p)
    @property
    def z(self):
        return self.r*np.cos(self.t)

class Point_xyz(Point):
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z
    @property
    def r(self):
        return np.sqrt(self._x**2 + self._y**2 + self._z**2)
    @property
    def t(self):
        return np.arccos(self._z/self.r)
    @property
    def p(self):
        return np.arctan2(self._y, self._x)
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def z(self):
        return self._z

def inner(p1, p2):
    return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z

def outer(p1, p2):
    x = p1.y*p2.z - p1.z*p2.y
    y = p1.z*p2.x - p1.x*p2.z
    z = p1.x*p2.y - p1.y*p2.x
    return Point_xyz(x, y, z)


class PointPair(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    @property
    def cos_central_angle(self):
        return inner(self.p1, self.p2)/self.p1.r/self.p2.r
    @property
    def sin_central_angle(self):
        return outer(self.p1, self.p2).r/self.p1.r/self.p2.r
    @property
    def central_angle(self):
        return np.arccos(self.cos_central_angle)

    def great_circle(self, t):
        w_x = (self.p2.x - self.p1.x*self.cos_central_angle)/self.sin_central_angle
        w_y = (self.p2.y - self.p1.y*self.cos_central_angle)/self.sin_central_angle
        w_z = (self.p2.z - self.p1.z*self.cos_central_angle)/self.sin_central_angle
        t_x = self.p1.x*np.cos(t) + w_x*np.sin(t)
        t_y = self.p1.y*np.cos(t) + w_y*np.sin(t)
        t_z = self.p1.z*np.cos(t) + w_z*np.sin(t)
        return t_x, t_y, t_z



