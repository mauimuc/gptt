#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from plotting import rcParams, prepare_map
from example import pairs
import h5py

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure(figsize=(3,3))
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)

# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )

# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Parametrization of the great circle path
pt12 = pair.great_circle_path


fh = h5py.File('../dat/example.hdf5', 'r')

points = fh['points']
cov_CC = fh['cov'][-1,:,:]



K = cov_CC[pair.indices,:]

# Integrate travel time
cor_TC = simps(K, dx=pair.spacing, axis=0)
cor_TC = np.ma.masked_array(cor_TC, cor_TC<0.01)

vmax = np.abs(cor_TC).max()

x, y = m(points['lon'], points['lat'])
pcol = ax_map.tripcolor(x, y, cor_TC, cmap='bwr', vmin=-vmax, vmax=vmax, rasterized=True)
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.set_ticks(np.linspace(-vmax, vmax, 7)[1:-1].round(3))
cbar.solids.set_edgecolor("face")


plt.savefig('../fig_correlation_pst.pgf')
