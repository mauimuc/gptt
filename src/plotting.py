#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

from mpl_toolkits.basemap import Basemap
from file_IO import read_station_file

#Options
rcParams = {'text.usetex' : True,
            'font.size' : 9,
            'savefig.dpi': 300,
            'savefig.pad_inches': 0.0,
            'pgf.texsystem' : 'pdflatex',
            'pgf.rcfonts': False, }


# Read coordinates of the NORSA Array
stations = read_station_file('../dat/stations.dat')
# Lower left corner
lllon = stations['lon'].min() - 1
lllat = stations['lat'].min() - 0.5
# Upper right corner
urlon = stations['lon'].max() + 1
urlat = stations['lat'].max() + 0.5


def prepare_map(ax=None, pls=[1,0,0,0]):
    ''' Prepare map and plot coast lines, parallels and meridians and returns the Basemap class. '''
    m = Basemap(llcrnrlon=lllon, llcrnrlat=lllat, urcrnrlon=urlon, urcrnrlat=urlat,\
                resolution='i', projection='merc',\
                lat_0=65, lon_0=17, lat_ts=50, ax=ax)
    m.drawcoastlines(color='gray', linewidth=0.5)
    m.drawparallels((65,69), labels=pls,       linewidth=0.5, dashes=(2,2))
    m.drawmeridians((15,20), labels=[0,0,1,0], linewidth=0.5, dashes=(2,2))
    return m


