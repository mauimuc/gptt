#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Plotting ... '''

import numpy as np
from matplotlib import pyplot as plt
from plotting import prepare_map, rcParams
import h5py

fh = h5py.File('../dat/example.hdf5', 'r')

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
pcol = ax_mu.tripcolor(x, y, mu_C, vmin=3940, vmax=4060, cmap='seismic', rasterized=True)
cbar = m.colorbar(pcol, location='bottom', pad="5%")
cbar.set_ticks([3950, 3975, 4000, 4025, 4050])
cbar.solids.set_edgecolor("face")
m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', color='g', s=4)

ax_sd = fig.add_subplot(122)
m = prepare_map(ax_sd, pls=[0,0,0,0])
pcol = ax_sd.tripcolor(x, y, sd_C, cmap='Reds', vmin=20, rasterized=True)
cbar = m.colorbar(pcol, location='bottom', pad="5%")
cbar.set_ticks([20, 25, 30, 35])
cbar.solids.set_edgecolor("face")
m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', color='g', s=4)

plt.savefig('../fig_example.pgf', bbox_inches='tight')


