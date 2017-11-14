#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the discretization as PGF file '''

from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
from example import stations, points

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure(figsize=(3,3))
m = prepare_map()

# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)
# Plot discretization
m.scatter(points['lon'], points['lat'], lw=0, marker='.', s=4, latlon=True, color='g', rasterized=True)

plt.savefig('../fig_discretization.pgf')

