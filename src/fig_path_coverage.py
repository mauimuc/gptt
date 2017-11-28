#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the path coverage as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
from gptt import read_station_file, ListPairs
from reference import dt_obs


plt.rcParams.update(rcParams)


# Read station coordinates
all_stations = read_station_file('../dat/stations.dat')
# Read pseudo data
pseudo_data = np.genfromtxt('../dat/pseudo_data.dat', dtype=dt_obs)
# Instantiate
pairs = ListPairs(pseudo_data, all_stations)
# Stations
stations = pairs.stations
# Discretization
points = pairs.points


# Prepare map
m = prepare_map()

# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)

# Plot great circle for all combinations
for pair in pairs:
    path = pair.great_circle_path
    if path.size == 2:
        continue
    m.plot(path['lon'], path['lat'], latlon=True, \
           linewidth=0.5, color='g', alpha=0.35)

plt.savefig('../fig_path_coverage.pgf', transparent=True)


