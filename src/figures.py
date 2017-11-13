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
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon
from example import stations, pairs, ell, tau, points, c_act

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure(figsize=(3,3))
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)


# Plot discretization
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)
sct = m.scatter(points['lon'], points['lat'], lw=0, marker='.', s=4, latlon=True, color='g', rasterized=True)

plt.savefig('../fig_discretization.pgf')
# only remove discretization from figure
sct.remove()

# Plot ray coverage
# Plot great circle for all combinations
gcs = list()
for pair in pairs:
    p, = m.drawgreatcircle(pair.st1['lon'], pair.st1['lat'], \
                        pair.st2['lon'], pair.st2['lat'], \
                        linewidth=0.5, color='g', alpha=0.5)
    gcs.append(p)

plt.savefig('../fig_path_coverage.pgf')
# remove
for p in gcs:
    p.remove()


# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )


# Plot reference velocity model

# Make a lat, lon grid with extent of the map
N = 60j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model

ims = m.imshow(c, cmap='seismic', vmin=3940, vmax=4060)
cbar = plt.colorbar(ims, cax=ax_cbr, orientation='horizontal')
ticks = np.linspace(3950, 4050, 5)
cbar.set_ticks(ticks)
cbar.set_label(r'$^m/_s$')
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_reference_model.pgf')
#remove
ims.remove()
for a in ax_cbr.artists:
    a.remove()

#### Plot covariance ####

# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Parametrization of the great circle path
pt12 = pair.great_circle_path

# Make a lat lon grid with extent of the map
N = 200j
lllat = min(pair.st1['lat'], pair.st2['lat']) - 0.5
urlat = max(pair.st1['lat'], pair.st2['lat']) + 0.5
lllon = min(pair.st1['lon'], pair.st2['lon']) - 1
urlon = max(pair.st1['lon'], pair.st2['lon']) + 1
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)

# Calculate Correlations amongst great circle segment and grid
K = gauss_kernel(pt12.reshape((-1,1,1)), grid, tau=tau, ell=ell)

# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
KK = np.ma.masked_less((K[4] + K[12] + K[-4])/K.max(), 0.01)
pcol = m.pcolormesh(lon, lat, KK, latlon=True, cmap='Purples', vmin=0, vmax=1, rasterized=True, zorder=1)

cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1])
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_kernel.pgf')
pcol.remove()

#### Plot correlation kernel ####

# Highlight parametrization of the great circle segment
m.plot(pt12['lon'], pt12['lat'], latlon=True, lw=0.5, color='g')
m.scatter(pt12['lon'][1:-1], pt12['lat'][1:-1], latlon=True, s=1, marker='.', color='g')

# Integrate travel time
cor_TC = simps(K, dx=pair.spacing, axis=0)
cor_TC = cor_TC/cor_TC.max()
cor_TC = np.ma.masked_array(cor_TC, cor_TC<0.01)

# Plot correlation kernel; pcolor needs points in between
m.pcolormesh(lon, lat, cor_TC, latlon=True, cmap='Purples', vmin=0, vmax=1, rasterized=True, zorder=0)

plt.savefig('../fig_correlation.pgf')

