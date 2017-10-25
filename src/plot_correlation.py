#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from file_IO import read_station_file
from example import c_act
from gptt import dt_latlon, cos_central_angle, great_circle_path, to_latlon, gauss_kernel, line_element
from scipy.integrate import simps

#Options
params = {'text.usetex' : True,
          'font.size' : 9,
          'pgf.rcfonts': False, }
plt.rcParams.update(params)


# Read coordinates of the NORSA Array
stations = read_station_file('../dat/stations.dat')

plt.figure(figsize=(4, 4))
lllon = stations['lon'].min() - 1
lllat = stations['lat'].min() - 0.5
urlon = stations['lon'].max() + 1
urlat = stations['lat'].max() + 0.5
m = Basemap(llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat,\
            resolution='i', projection='merc',\
            lat_0=65, lon_0=17, lat_ts=50)
m.drawcoastlines(color='gray')
m.drawparallels(range(60,70,5), labels=[1,0,0,0])
m.drawmeridians(range(10,30,5), labels=[0,0,0,1])

st1 = np.array( (65.5, 21), dtype=dt_latlon )
st2 = np.array( (69, 13), dtype=dt_latlon )

# My own parametrization of the great circle segment
ds = np.deg2rad(0.05)
ca12 = np.arccos(cos_central_angle(st1, st2))
n12 = np.round(ca12/ds, 0).astype(int)
t12 = np.linspace(0, ca12, n12)
le12 = line_element(st1, st2, t12)
pt12 = to_latlon(great_circle_path(st1, st2, t12))
m.plot(pt12['lon'], pt12['lat'], '-x', latlon=True)


# Make a lat lon grid with extent of the map
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:150j, lllon:urlon:150j], dtype=dt_latlon)
# Correlations amongst great circle segment and grid
K = gauss_kernel(pt12.reshape((-1,1,1)), grid, 20000**2)
# Integrate travel time
cor_TC = simps(K, t12, axis=0)
# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:151j, lllon:urlon:151j]
m.pcolormesh(lon, lat, cor_TC, latlon=True, cmap='Reds', rasterized=True)
#m.imshow(cor_TC)

plt.savefig('../doc/correlation.pgf', transparent=True, \
            bbox_inches='tight', pad_inches=0.01)

