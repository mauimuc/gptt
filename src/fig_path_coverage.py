#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the path coverage as PGF file '''

from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
from example import stations, pairs

plt.rcParams.update(rcParams)

# Prepare map
m = prepare_map()

# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)

# Plot great circle for all combinations
for pair in pairs:
    m.drawgreatcircle(pair.st1['lon'], pair.st1['lat'], \
                      pair.st2['lon'], pair.st2['lat'], \
                      linewidth=0.5, color='g', alpha=0.5)

plt.savefig('../fig_path_coverage.pgf')


