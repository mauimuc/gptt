#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Calculate the Bayesian posterior considering data all at once. '''

import numpy as np
from gptt import f_mu_T, f_cov_TT, gauss_kernel, misfit
from example import mu_C_pri, tau, ell, pairs, d
import h5py

# Open HDF5 file handle
fh = h5py.File('../dat/all_at_once.hdf5', 'w')

# Store stations
fh.create_dataset('stations', data=pairs.stations)

# Get discretization
points = pairs.points
# Store discretization
fh.create_dataset('points', data=points)

# A priori mean and assumed covariance
mu_C = mu_C_pri(points)
cov_CC = gauss_kernel(points.reshape(1,-1),points.reshape(-1,1), ell=ell, tau=tau)
# Prior predictive mean and covariance
mu_D_pri = f_mu_T(pairs, mu_C)
cov_DD_pri = f_cov_TT(pairs, mu_C, cov_CC)
# Store prior assumptions
dset_mu = fh.create_dataset('mu', (2,) + mu_C.shape)
dset_mu[0,:] = mu_C
dset_sd = fh.create_dataset('sd', (2,) + mu_C.shape)
dset_sd[0,:] = np.sqrt(cov_CC.diagonal())
fh.create_dataset('mu_D_pri', data=mu_D_pri)
fh.create_dataset('cov_DD_pri', data=cov_DD_pri)


# Correlations amongst data and model
# TODO: Have cor_CD as a method of ListPairs
cor_CD = np.empty( mu_C.shape + np.shape(d) )
for i in range(len(pairs)):
    cor_CD[:,i] = pairs[i].cor_CT(mean=mu_C, cov=cov_CC)

# Cholesky factorization
# XXX SciPy cho_factor and cho_solve preform a little better
L = np.linalg.cholesky(cov_DD_pri)
M = np.linalg.solve(L, cor_CD.T)
# Posterior mean an covariance
mu_C += np.dot(M.T, np.linalg.solve(L, d - mu_D_pri))
cov_CC -= np.dot(M.T, M)

# Posterior mean and covariance
mu_D_pst = f_mu_T(pairs, mu_C)
cov_DD_pst = f_cov_TT(pairs, mu_C, cov_CC)

# Store posterior mean and covariances
dset_mu[1,:] = mu_C
dset_sd[1,:] = np.sqrt(cov_CC.diagonal())
fh.create_dataset('mu_D_pst', data=mu_D_pst)
fh.create_dataset('cov_DD_pst', data=cov_DD_pst)

# Prior and posterior misfit
#misfit = (misfit(d, mu_D_pri, cov_DD_pri), \
#          misfit(d, mu_D_pst, cov_DD_pst))

# Close HDF5 file
fh.close()
