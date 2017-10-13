#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt


def inner(s_lat, s_lon, t_lat, t_lon):
    ''' inner product in spherical coordinates
        lat, lon in degrees '''
    t_th = np.deg2rad( t_lat )
    s_th = np.deg2rad( s_lat )
    return np.cos(s_th)*np.cos(t_th)*np.cos(np.deg2rad(s_lon-t_lon)) + np.sin(s_th)*np.sin(t_th)


s_lat = 60
s_lon = 15
t_lat = 70
t_lon = 20

m = Basemap(llcrnrlon=0., llcrnrlat=55., urcrnrlon=40.,urcrnrlat=72.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)
m.drawcoastlines()
m.drawgreatcircle(s_lon, s_lat, t_lon, t_lat, linewidth=2, color='r')


lons, lats = np.mgrid[m.llcrnrlon:m.urcrnrlon:65j,
                      m.llcrnrlat:m.urcrnrlat:65j]
d = np.arccos(inner(lats, lons, s_lat, s_lon))
K = np.exp(-(d*6)**2/2)

m.pcolormesh(lons, lats, K, latlon=True)
plt.colorbar()


plt.show()

