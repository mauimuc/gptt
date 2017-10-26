#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

from gptt import dt_latlon, cos_central_angle, great_circle_path, to_latlon, gauss_kernel
from scipy.integrate import simps
from plotting import np, plt, m, lllat, lllon, urlat, urlon, stations

# Plot stations
m.scatter(stations['lon'], stations['lat'], latlon=True)

# Stations
st1 = stations[0]
st2 = stations[4]

# Parametrization of the great circle segment
ca12 = np.arccos(cos_central_angle(st1, st2))
t12 = np.linspace(0, ca12, 50)
pt12 = to_latlon(great_circle_path(st1, st2, t12))
m.plot(pt12['lon'], pt12['lat'], latlon=True)


# Make a lat lon grid with extent of the map
N = 150j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
# Correlations amongst great circle segment and grid
K = gauss_kernel(pt12.reshape((-1,1,1)), grid, 15000**2)
# Integrate travel time
cor_TC = simps(K, t12, axis=0)
# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
m.pcolormesh(lon, lat, cor_TC/cor_TC.max(), latlon=True, cmap='seismic', vmin=-1, vmax=1, rasterized=True)

plt.savefig('../fig_correlation.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)
