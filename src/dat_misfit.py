#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Write misfit to text file; an ugly hack '''

import numpy as np
import h5py

with h5py.File('../dat/misfit_rnd.hdf5', 'r') as fh:
    misfit = fh['misfit']
    data = np.full(np.shape(misfit['evd'])+(6,) , np.nan)
    data[:,0] = misfit['evd']
    data[:,1] = misfit['val']

with h5py.File('../dat/misfit_all.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
    data[[0,-1],2] = misfit['val']

with h5py.File('../dat/misfit_rnd2.hdf5', 'r') as fh:
    misfit = fh['misfit']
    data[:,3] = misfit['val']

with h5py.File('../dat/misfit_dsc.hdf5', 'r') as fh:
    misfit = fh['misfit']
    data[:,4] = misfit['val']

with h5py.File('../dat/misfit_asc.hdf5', 'r') as fh:
    misfit = fh['misfit']
    data[:,5] = misfit['val']


fmt = '%3i ' + 5*'%6.2f '
hdr = 'evd rnd1 all rnd2 dsc asc'
np.savetxt('../dat/misfit.dat', data, fmt=fmt, header=hdr, comments='')

