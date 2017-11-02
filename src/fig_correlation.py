#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

from gptt import dt_latlon, cos_central_angle, great_circle_path, to_latlon, gauss_kernel
from scipy.integrate import simps
from plotting import np, plt, m, lllat, lllon, urlat, urlon, stations

# Stations
st1 = stations[0]
st2 = stations[4]

# Parametrization of the great circle segment
ca12 = np.arccos(cos_central_angle(st1, st2))
t12, ds = np.linspace(0, ca12, 25, retstep=True)
pt12 = to_latlon(great_circle_path(st1, st2, t12))
m.plot(pt12['lon'], pt12['lat'], latlon=True, marker='.')


# Make a lat lon grid with extent of the map
N = 150j
#lllat = min(st1['lat'], st2['lat']) - 0.5
#urlat = max(st1['lat'], st2['lat']) + 0.5
#lllon = min(st1['lon'], st2['lon']) - 1
#urlon = max(st1['lon'], st2['lon']) + 1
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
# Correlations amongst great circle segment and grid
ell = 15000
K = gauss_kernel(pt12.reshape((-1,1,1)), grid, ell)
# Integrate travel time
cor_TC = simps(K, t12, axis=0)
# Plot correlation kernel; pcolor needs points in between
lat, lon = np.mgrid[lllat:urlat:N+1j, lllon:urlon:N+1j]
m.pcolormesh(lon, lat, cor_TC/cor_TC.max(), latlon=True, cmap='seismic', vmin=-1, vmax=1, rasterized=True)

plt.savefig('../fig_correlation.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)

with open('../def_correlation.tex', 'w') as fh:
    fh.write(r'\def\SFWcorrell{%i}' % ell + '\n')
    fh.write(r'\def\SFWcorrds{%.3f}' % np.rad2deg(ds) + '\n')

