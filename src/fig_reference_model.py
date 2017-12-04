#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the reference velocity model '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon, ListPairs, read_station_file
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon, cmap_mu
from reference import c_act, dt_obs


plt.rcParams.update(rcParams)


# Read station coordinates
all_stations = read_station_file('../dat/stations.dat')
# Read pseudo data
pseudo_data = np.genfromtxt('../dat/pseudo_data.dat', dtype=dt_obs)
# Instantiate
pairs = ListPairs(pseudo_data, all_stations)
# Stations
stations = pairs.stations


# Prepare map
fig = plt.figure()
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)
# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )


# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True, zorder=2)


# Make a lat, lon grid with extent of the map
N = 60j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model

# Plot reference model
ims = m.imshow(c, cmap=cmap_mu, vmin=3900, vmax=4100)
# Make colorbar
cbar = plt.colorbar(ims, cax=ax_cbr, orientation='horizontal')
cbar.set_label(r'$^m/_s$')
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_reference_model.pgf', transparent=True)

# Contour lines
#cnt = m.contour(grid['lon'], grid['lat'], c, levels=c_act.levels(20), latlon=True, colors='k', linewidths=0.5)


