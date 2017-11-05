#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the path coverage as PGF file '''

from file_IO import read_station_file
from example import c_act
from gptt import dt_latlon
from plotting import np, plt, prepare_map, lllat, lllon, urlat, urlon, stations

m = prepare_map()

# Combinations of all stations dropping duplicates
idx, idy = np.tril_indices(stations.size, -1)
# Plot great circle for all combinations
for st1, st2 in zip(stations[idx], stations[idy]):
    m.drawgreatcircle(st1['lon'], st1['lat'], st2['lon'], st2['lat'], \
                      linewidth=0.5, color='g', alpha=0.5)


# Make a lat lon grid with extent of the map
N = 30j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model

# Plot velocity model
m.imshow(c, cmap='seismic', vmin=3940, vmax=4060)
cbar = m.colorbar(location='right', pad="5%")
ticks = np.linspace(3950, 4050, 5)
cbar.set_ticks(ticks)
cbar.set_label(r'$\frac ms$', rotation='horizontal')
plt.savefig('../fig_path_coverage.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)

