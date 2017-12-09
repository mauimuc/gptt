#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Plotting ... '''

from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map, cmap_sd
import h5py

plt.rcParams.update(rcParams)

# FIXME Prone to errors
infile = argv[1]
outfile = infile.replace('dat/', 'fig_').replace('.hdf5', '_sd_V_pst.pgf')


fh = h5py.File(infile, 'r')
points = fh['points']
sd_C_pst = np.sqrt(np.diagonal(fh['cov_CC_pst']))

# Prepare map
fig = plt.figure()
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)
# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )

# Plot realization
x, y = m(points['lon'], points['lat'])
pcol = ax_map.tripcolor(x, y, sd_C_pst, cmap=cmap_sd, rasterized=True)
# Make colorbar
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.set_label(r'$^m/_s$')
cbar.solids.set_edgecolor("face")

# Save PGF file
plt.savefig(outfile, transparent=True)



