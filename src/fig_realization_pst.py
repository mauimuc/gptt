#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Plotting ... '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon
from plotting import rcParams, prepare_map, lllat, lllon, urlat, urlon, cmap_mu
from reference import c_act
import h5py

plt.rcParams.update(rcParams)

fh = h5py.File('../dat/example.hdf5', 'r')
points = fh['points']
mu_C_pst = fh['mu'][-1,:]
cov_CC_pst = fh['cov_CC_pst'][:,:]

# Prepare map
fig = plt.figure()
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)
# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )

# Draw realization
r = np.random.multivariate_normal(mu_C_pst, cov_CC_pst)

# Plot realization
x, y = m(points['lon'], points['lat'])
pcol = ax_map.tripcolor(x, y, r, cmap=cmap_mu, rasterized=True, vmin=3900, vmax=4100)
# Make colorbar
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.set_label(r'$^m/_s$')
cbar.solids.set_edgecolor("face")

# Save PGF file
plt.savefig('../fig_realization_pst.pgf', transparent=True)


