#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the path coverage as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon
from example import pairs, c_act


# Prepare map
fig = plt.figure(figsize=(4,4))
plt.rcParams.update(rcParams)
m = prepare_map()

# Plot great circle for all combinations
for pair in pairs:
    m.drawgreatcircle(pair.st1['lon'], pair.st1['lat'], \
                      pair.st2['lon'], pair.st2['lat'], \
                      linewidth=0.5, color='g', alpha=0.5)


# Make a lat lon grid with extent of the map
N = 30j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model

# Plot velocity model
m.imshow(c, cmap='seismic', vmin=3940, vmax=4060)
cbar = m.colorbar(location='right', pad="5%")
ticks = np.linspace(3950, 4050, 5)
cbar.set_ticks(ticks)
cbar.set_label(r'$\frac ms$', rotation='horizontal')
cbar.solids.set_edgecolor("face")
plt.savefig('../fig_path_coverage.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)

