#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Calculates the misfit '''

from gptt import cov_TT, mu_T, misfit
from example import pairs
import h5py

fh = h5py.File('../dat/example.hdf5', 'r')

points = fh['points']
mu_C = fh['mean']
cov_CC = fh['cov']
d = fh['d']

misfit = np.empty(num.C.shape[0])
for i in range(mu_C.shape[0]):
    mean = mu_T(pairs, mean=mu_C[i,:])
    cova = cov_TT(pairs, mean=mu_C[i,:], cov=cov_CC[i,:,:])
    misfit[i] = misfit(d, mean, cova)


