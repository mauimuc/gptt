#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Save a plot of the correlation kernel as PGF file '''

from matplotlib import pyplot as plt
from plotting import rcParams, prepare_map
import h5py

plt.rcParams.update(rcParams)

# Prepare map
fig = plt.figure(figsize=(6,1))

fh = h5py.File('../dat/example.hdf5', 'r')

misfit = fh['misfit'][:]

plt.plot(misfit)

plt.savefig('../fig_misfit.pgf')

