#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
import h5py


with h5py.File('../dat/misfit_all.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot([0,19], np.sqrt(misfit), 'x--', label='all at once')

with h5py.File('../dat/misfit_rnd.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), 'o--', label='shuffle')

with h5py.File('../dat/misfit_rnd2.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), 'o--', label='shuffle')

with h5py.File('../dat/misfit_dsc.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), 'x--', label='descending')

with h5py.File('../dat/misfit_asc.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), 'x--', label='ascending')

plt.legend()
plt.show()

