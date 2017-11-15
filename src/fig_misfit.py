#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

import numpy as np
from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
import h5py

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure(figsize=(6,1))

from all_at_once import misfit
misfit = np.sqrt(misfit)
plt.plot((0,190), misfit, '.--')
print misfit

with h5py.File('../dat/example.hdf5', 'r') as fh:
    misfit = np.sqrt(fh['misfit'])
plt.plot(misfit)
print misfit[[0,-1]]

#with h5py.File('../dat/succession_sorted.hdf5', 'r') as fh:
#    misfit = np.sqrt(fh['misfit'])
#plt.plot(misfit)
#print misfit[[0,-1]]

plt.savefig('../fig_misfit.pgf')

