#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Plotting ... '''

from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon
from reference import c_act
import h5py

fh = h5py.File(argv[1], 'r')

points = fh['points']
stations = fh['stations']
mu_C = fh['mu'][-1,:]
sd_C = fh['sd'][-1,:]

plt.rcParams.update(rcParams)
fig = plt.figure(figsize=(6.5,4))
fig.subplots_adjust(wspace=0.02)

ax_mu = fig.add_subplot(121)
m = prepare_map(ax_mu)
x, y = m(points['lon'], points['lat'])
pcol = ax_mu.tripcolor(x, y, mu_C, cmap='seismic', rasterized=True, vmin=c_act.min, vmax=c_act.max)
cbar = m.colorbar(pcol, location='bottom', pad="5%")
cbar.set_ticks(c_act.levels(20))
cbar.solids.set_edgecolor("face")
m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', color='g', s=4)

# Make a lat, lon grid with extent of the map
N = 150j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
levels = c_act.levels(20)
m.contour(grid['lon'], grid['lat'], c_act(grid), latlon=True, leverls=levels, colors='k')

ax_sd = fig.add_subplot(122)
m = prepare_map(ax_sd, pls=[0,0,0,0])
pcol = ax_sd.tripcolor(x, y, sd_C, cmap='Reds', rasterized=True, vmin=33, vmax=40)
cbar = m.colorbar(pcol, location='bottom', pad="5%")
cbar.set_ticks([34, 35, 36, 37, 38, 39])
cbar.solids.set_edgecolor("face")
m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', color='g', s=4)

out_file = argv[1].replace('dat/','fig_').replace('hdf5', 'pgf')
plt.savefig(out_file, bbox_inches='tight')


