#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import gauss_kernel, f_mu_T, f_cov_TT, misfit
from example import pairs, mu_C_pri, points, stations, tau, ell, epsilon
import h5py

# Sort station pairs by their great circle distance
pairs = sorted(pairs, key=lambda pair: pair.central_angle)

d = [pair.d for pair in pairs]

# A priori assumptions
mu_C = mu_C_pri(points) # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

# Open HDF5 file handle
fh = h5py.File('../dat/succession_sorted.hdf5', 'w')
# Create dataset for misfit
dset_misfit = fh.create_dataset('misfit', (len(pairs) + 1, ), dtype=np.float32)
# Calculate and save prior misfit
mu_T = f_mu_T(pairs, mu_C)
cov_TT = f_cov_TT(pairs, mu_C, cov_CC)
dset_misfit[0] = misfit(d, mu_T, cov_TT)

# Successively consider evidence
for i in range(len(pairs)):
    # To be considered evidence
    pair = pairs[i]
    # Prior mean
    mu_T = pair.T(mu_C)
    # Correlations amongst model and travel time
    cor_CT = pair.cor_CT(mean=mu_C, cov=cov_CC)
    # Prior variance
    var_DD = pair.var_DD(mean=mu_C, cov=cov_CC)
    # Update posterior mean
    mu_C += cor_CT/var_DD*(pair.d - mu_T)
    # Update posterior co-variance
    cov_CC -= np.dot(cor_CT[:,np.newaxis], cor_CT[np.newaxis,:])/var_DD
    # Screen output
    print 'Combination', pair, '%3i/%3i' % (i, len(pairs))
    # Calculate and save misfit
    mu_T = f_mu_T(pairs, mu_C)
    cov_TT = f_cov_TT(pairs, mu_C, cov_CC)
    dset_misfit[i+1] = misfit(d, mu_T, cov_TT)

# Close HDF5 file
fh.close()
