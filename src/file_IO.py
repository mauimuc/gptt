#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Module storing routines for file IO '''

import numpy as np
from gptt import dt_latlon

def read_station_file(fname):
    ''' Extract latitude and longitude from station file.
        A structure array is returned. '''
    return np.genfromtxt(fname, dtype=dt_latlon, usecols=(1, 2))[:16]

def read_station_file2(fname):
    my_dt = [ ('stnm', 'S5'), ('lat',float), ('lon',float), ('elv', 'f') ]
    return np.genfromtxt(fname, dtype=my_dt, )

# name lat lon in degrees, elevation in meter, space separated
