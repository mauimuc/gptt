#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the prior correlation amongst model and observation as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon, gauss_kernel
from scipy.integrate import simps
from plotting import rcParams, prepare_map
from example import pairs, ell, tau

stations = pairs.stations

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure()
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)
# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )


# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)


# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Parametrization of the great circle path
path = pair.great_circle_path

# Make a lat, lon grid with the extent of the path
N = 200j
lllat = min(pair.st1['lat'], pair.st2['lat']) - 0.5
urlat = max(pair.st1['lat'], pair.st2['lat']) + 0.5
lllon = min(pair.st1['lon'], pair.st2['lon']) - 1
urlon = max(pair.st1['lon'], pair.st2['lon']) + 1
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)

# Calculate Correlations amongst great circle segment and grid
K = gauss_kernel(path.reshape((-1,1,1)), grid, tau=tau, ell=ell)


# Highlight parametrization of the great circle segment
m.plot(path['lon'], path['lat'], latlon=True, lw=0.5, color='g')
m.scatter(path['lon'][1:-1], path['lat'][1:-1], latlon=True, s=1, marker='.', color='g')

# Integrate travel time
cor_TC = simps(K, dx=pair.spacing, axis=0)
cor_TC = np.ma.masked_less(cor_TC, 0.01)
vmax = cor_TC.max()

# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
pcol = m.pcolormesh(lon, lat, cor_TC, latlon=True, cmap='PuOr', rasterized=True, \
                    vmin=-vmax, vmax=vmax, zorder=0)
# Make colorbar
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_correlation_pri.pgf')


