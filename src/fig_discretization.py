#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the discretization as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
from gptt import read_station_file, ListPairs
from pseudo_data import dt_obs


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
# Plot discretization
m.scatter(points['lon'], points['lat'], lw=0, marker='.', s=4, latlon=True, color='g', rasterized=True)

plt.savefig('../fig_discretization.pgf')

