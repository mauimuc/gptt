#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
import h5py


with h5py.File('../dat/example.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit))

with h5py.File('../dat/all_at_once.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), label='all at once')

with h5py.File('../dat/example_descending.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), label='descending')

with h5py.File('../dat/example_ascending.hdf5', 'r') as fh:
    misfit = fh['misfit'][:]
plt.plot(np.sqrt(misfit), label='ascending')

plt.legend()
plt.show()

