#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the prior covariance kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from gptt import dt_latlon, gauss_kernel
from plotting import rcParams, prepare_map
from configparser import ConfigParser
from reference import dt_obs
from gptt import read_station_file, ListPairs

plt.rcParams.update(rcParams)


# Read parameter file
config = ConfigParser()
with open('parameter.ini') as fh:
    config.readfp(fh)

# Kernel Parameters
tau = config.getfloat('Prior', 'tau') # A priori uncertainty; standard deviation
ell = config.getfloat('Prior', 'ell') # Characteristic length

# Read station coordinates
station_file = config.get('Observations', 'station_file')
all_stations = read_station_file(station_file)
# Read pseudo data
data_file = config.get('Observations', 'data')
pseudo_data = np.genfromtxt(data_file, dtype=dt_obs)

# Observations
pairs = ListPairs(pseudo_data, all_stations)
# Only those stations occurring in the data
stations = pairs.stations


# Prepare map
fig = plt.figure()
ax_map = fig.add_subplot(111)
m = prepare_map(ax=ax_map)
# Add axes for the colorbar
bbox = ax_map.get_position()
ax_cbr = fig.add_axes( (bbox.x0, bbox.y0 - 0.06, bbox.width, 0.04) )


# Plot station locations
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True, zorder=10)


# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Middle point of the great circle path
p = pair.great_circle_path[12]

# Make a lat, lon grid round the middle point
N = 100j
lllat = p['lat'] - 0.5
urlat = p['lat'] + 0.5
lllon = p['lon'] - 1
urlon = p['lon'] + 1
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)

# Calculate kernel at the middle point
K = gauss_kernel(p, grid, tau=tau, ell=ell)
K = np.ma.masked_less(K, K.max()/50)
vmax = K.max()

# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
pcol = m.pcolormesh(lon, lat, K, latlon=True, cmap='PuOr', rasterized=True, \
                    vmin=-vmax, vmax=vmax, zorder=1)
# Make colorbar
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_kernel_pri.pgf')
