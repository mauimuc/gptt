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

#Options
params = {'text.usetex' : True,
          'font.size' : 9,
          'pgf.rcfonts': False, }
plt.rcParams.update(params)


# Read coordinates of the NORSA Array
stations = read_station_file('../dat/stations.dat')

plt.figure(figsize=(4,4))
lllon = stations['lon'].min() - 1
lllat = stations['lat'].min() - 0.5
urlon = stations['lon'].max() + 1
urlat = stations['lat'].max() + 0.5
m = Basemap(llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat,\
            resolution='i', projection='merc', lat_0=65, lon_0=17, lat_ts=50)
m.drawcoastlines(color='gray', linewidth=0.5)
m.drawparallels(range(60,70,5), labels=[1,0,0,0], linewidth=0.5, dashes=(2,2))
m.drawmeridians(range(10,30,5), labels=[0,0,0,1], linewidth=0.5, dashes=(2,2))

# Combinations of all stations dropping duplicates
idx, idy = np.tril_indices(stations.size, -1)
# Plot great circle for all combinations
for (lat1, lon1), (lat2, lon2) in zip(stations[idx], stations[idy]):
    m.drawgreatcircle(lon1, lat1, lon2, lat2, linewidth=0.5, color='g', alpha=0.5)

# My own parametrization of the great circle segment
#from gptt import cos_central_angle, great_circle_path, to_latlon
#ds = np.deg2rad(0.05)
#for st1, st2 in zip(stations[idx], stations[idy]):
#    ca = np.arccos(cos_central_angle(st1, st2))
#    num = np.round(ca/ds,0).astype(int)
#    t = np.linspace(0, ca, num)
#    points = to_latlon(great_circle_path(st1, st2, t))
#    m.plot(points['lon'], points['lat'], 'x', latlon=True, linewidth=0.5, color='g', alpha=0.5)

from gptt import dt_latlon
# Make a lat lon grid with extent of the map
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:30j, lllon:urlon:30j], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model
# Plot velocity model
m.imshow(c, cmap='seismic', vmin=3940, vmax=4060)
cbar = m.colorbar(location='right', pad="5%")
ticks = np.linspace(3950, 4050, 5)
cbar.set_ticks(ticks)
cbar.set_label(r'$\frac ms$', rotation='horizontal')
plt.savefig('../doc/setting.pgf', transparent=True, \
            bbox_inches='tight', pad_inches=0.01)
#plt.show()
