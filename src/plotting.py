#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from file_IO import read_station_file

#Options
params = {'text.usetex' : True,
          'font.size' : 9,
          'savefig.dpi': 300,
          'savefig.pad_inches': 0.0,
          'pgf.texsystem' : 'pdflatex',
          'pgf.rcfonts': False, }
plt.rcParams.update(params)


# Read coordinates of the NORSA Array
stations = read_station_file('../dat/stations.dat')

plt.figure(figsize=(4, 4))
lllon = stations['lon'].min() - 1
lllat = stations['lat'].min() - 0.5
urlon = stations['lon'].max() + 1
urlat = stations['lat'].max() + 0.5
m = Basemap(llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat,\
            resolution='i', projection='merc',\
            lat_0=65, lon_0=17, lat_ts=50)
m.drawcoastlines(color='gray', linewidth=0.5)
m.drawparallels((65,69), labels=[1,0,0,0], linewidth=0.5, dashes=(2,2))
m.drawmeridians((15,20), labels=[0,0,1,0], linewidth=0.5, dashes=(2,2))


