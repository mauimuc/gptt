#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Module storing routines for file IO '''

import numpy as np
from gptt import dt_latlon

def read_station_file(fname):
    ''' Reads the station file and returns a structured array '''
    my_dt = [ ('stnm', 'S5'), ('lat',float), ('lon',float), ('elv', 'f') ]
    return np.genfromtxt(fname, dtype=my_dt, )

