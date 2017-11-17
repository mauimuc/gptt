#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the reference velocity model '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon
from example import pairs
from pseudo_data import c_act

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


# Make a lat, lon grid with extent of the map
N = 60j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model

# Plot reference model
ims = m.imshow(c, cmap='seismic', vmin=3940, vmax=4060)
# Make colorbar
cbar = plt.colorbar(ims, cax=ax_cbr, orientation='horizontal')
ticks = np.linspace(3950, 4050, 5)
cbar.set_ticks(ticks)
cbar.set_label(r'$^m/_s$')
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_reference_model.pgf')

