#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from plotting import rcParams, prepare_map
import h5py
from configparser import ConfigParser
from reference import dt_obs
from gptt import read_station_file, ListPairs

plt.rcParams.update(rcParams)


# Read parameter file
config = ConfigParser()
with open('parameter.ini') as fh:
    config.readfp(fh)

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
m.scatter(stations['lon'], stations['lat'], lw=0, color='g', latlon=True)

# Find stations which are the furtherest apart
pair = max(pairs, key=lambda pair: pair.central_angle)
# Parametrization of the great circle path
pt12 = pair.great_circle_path


fh = h5py.File('../dat/example.hdf5', 'r')

points = fh['points']
x, y = m(points['lon'], points['lat'])
cov_CC = fh['cov_CC_pst'][:,:]


K = cov_CC[pair.indices,:]
vmax = np.abs(K[12,:]).max()

pcol = ax_map.tripcolor(x, y, K[12,:], cmap='PuOr', vmin=-vmax, vmax=vmax, rasterized=True, zorder=0)
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
cbar.set_ticks(np.linspace(-vmax, vmax, 7)[1:-1].round(-1))
cbar.solids.set_edgecolor("face")

plt.savefig('../fig_kernel_pst.pgf')
pcol.remove()
for a in ax_cbr.artists:
    a.remove()

# Integrate travel time
cor_TC = simps(K, dx=pair.spacing, axis=0)
cor_TC = np.ma.masked_array(cor_TC, cor_TC<0.01)

vmax = np.abs(cor_TC).max()

pcol = ax_map.tripcolor(x, y, cor_TC, cmap='PuOr', vmin=-vmax, vmax=vmax, rasterized=True, zorder=0)
cbar = plt.colorbar(pcol, cax=ax_cbr, orientation='horizontal')
#cbar.set_ticks(np.linspace(-vmax, vmax, 7)[1:-1].round(3))
cbar.solids.set_edgecolor("face")


plt.savefig('../fig_correlation_pst.pgf')
