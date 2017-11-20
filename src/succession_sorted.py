#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import gauss_kernel
from example import pairs, mu_C_pri, tau, ell
import h5py

# Sort station pairs by their great circle distance
pairs.sort(key=lambda p: p.central_angle)
# Discretization
points = pairs.points

# A priori assumptions
mu_C = mu_C_pri(points) # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

# Open HDF5 file handle
# TODO: Better use groups and store in a single HDF5 file
fh = h5py.File('../dat/succession_sorted.hdf5', 'w')
# Store stations
fh.create_dataset('stations', data=pairs.stations)
# Store discretization
fh.create_dataset('points', data=points)
# Create datasets for mean and standard deviation
dset_mu = fh.create_dataset('mu', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)
dset_sd = fh.create_dataset('sd', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)

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
    # Save mean and standard deviation
    dset_mu[i+1,:] = mu_C
    dset_sd[i+1,:] = np.sqrt(cov_CC.diagonal())

# Close HDF5 file
fh.close()

## Sort station pairs by their great circle distance
#pairs.sort(key=lambda p: p.central_angle, reverse=True)
## Discretization
#points = pairs.points

# A priori assumptions
mu_C = mu_C_pri(points) # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

# Open HDF5 file handle
# TODO: Better use groups and store in a single HDF5 file
fh = h5py.File('../dat/succession_reverse.hdf5', 'w')
# Store stations
fh.create_dataset('stations', data=pairs.stations)
# Store discretization
fh.create_dataset('points', data=points)
# Create datasets for mean and standard deviation
dset_mu = fh.create_dataset('mu', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)
dset_sd = fh.create_dataset('sd', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)

order = np.arange(len(pairs))
np.random.shuffle(order)
# Successively consider evidence
for j, i in enumerate(order):
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
    print 'Combination', pair, '%3i/%3i' % (j, len(pairs))
    # Save mean and standard deviation
    dset_mu[j+1,:] = mu_C
    dset_sd[j+1,:] = np.sqrt(cov_CC.diagonal())


# Close HDF5 file
fh.close()

