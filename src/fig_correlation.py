#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon, cos_central_angle, gauss_kernel, StationPair
from scipy.integrate import simps
from plotting import rcParams, prepare_map
from example import stations, pairs, ell, tau, points

# Prepare map
plt.figure(figsize=(4,4))
plt.rcParams.update(rcParams)
m = prepare_map()


# Plot discretization
m.scatter(points['lon'], points['lat'], s=1, marker='.', color='k', latlon=True)
# Plot stations
m.scatter(stations['lon'], stations['lat'], s=15, marker='.', color='g', latlon=True)
# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Highlight parametrization of the great circle segment
pt12 = pair.great_circle_path
m.plot(pt12['lon'], pt12['lat'], latlon=True, lw=0.5, color='b')
m.scatter(pt12['lon'][1:-1], pt12['lat'][1:-1], latlon=True, s=1, marker='.', color='b')

# Make a lat lon grid with extent of the map
N = 150j
lllat = min(pair.st1['lat'], pair.st2['lat']) - 0.5
urlat = max(pair.st1['lat'], pair.st2['lat']) + 0.5
lllon = min(pair.st1['lon'], pair.st2['lon']) - 1
urlon = max(pair.st1['lon'], pair.st2['lon']) + 1
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)

# Calculate Correlations amongst great circle segment and grid
K = gauss_kernel(pt12.reshape((-1,1,1)), grid, tau=tau, ell=ell)
# Integrate travel time
cor_TC = simps(K, dx=pair.spacing, axis=0)
cor_TC = cor_TC/cor_TC.max()
cor_TC = np.ma.masked_array(cor_TC, cor_TC<0.05)

# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
m.pcolormesh(lon, lat, cor_TC, latlon=True, cmap='seismic', vmin=-1, vmax=1, rasterized=True, zorder=0)

plt.savefig('../fig_correlation.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)


