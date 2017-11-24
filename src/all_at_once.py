#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Calculate the Bayesian posterior considering data all at once. '''

import numpy as np
from gptt import gauss_kernel, ListPairs, read_station_file
from ConfigParser import ConfigParser
from reference import dt_obs
import h5py

# Read parameter file
config = ConfigParser()
with open('../par/example.ini') as fh:
    config.readfp(fh)

# Kernel Parameters
# TODO Estimate hyper-parameters
tau = config.getfloat('Prior', 'tau') # A priori uncertainty; standard deviation
ell = config.getfloat('Prior', 'ell') # Characteristic length
mu  = config.getfloat('Prior', 'mu') # Constant a priori velocity model


# Read station coordinates
station_file = config.get('Observations', 'station_file')
all_stations = read_station_file(station_file)
# Read pseudo data
data_file = config.get('Observations', 'data')
pseudo_data = np.genfromtxt(data_file, dtype=dt_obs)

# Observations
pairs = ListPairs(pseudo_data, all_stations)

# Open HDF5 file handle
fh = h5py.File('../dat/all_at_once.hdf5', 'w')

# Store stations
fh.create_dataset('stations', data=pairs.stations)

# Get discretization
points = pairs.points
# Store discretization
fh.create_dataset('points', data=points)

# A priori mean and assumed covariance
mu_C = np.full_like(points, mu, dtype=np.float)
cov_CC = gauss_kernel(points.reshape(1,-1),points.reshape(-1,1), ell=ell, tau=tau)
# Prior predictive mean and covariance
mu_D_pri = pairs.mu_T(mu_C)
cov_DD_pri = pairs.cov_TT(mu_C, cov_CC)
# Store prior assumptions
dset_mu = fh.create_dataset('mu', (2,) + mu_C.shape)
dset_mu[0,:] = mu_C
dset_sd = fh.create_dataset('sd', (2,) + mu_C.shape)
dset_sd[0,:] = np.sqrt(cov_CC.diagonal())
fh.create_dataset('cov_DD_pri', data=cov_DD_pri)
# Store prior misfit
dset_misfit = fh.create_dataset('misfit', (2, ))
dset_misfit[0] = pairs.misfit(mu_C, cov_CC)


# Correlations amongst data and model
# TODO: Have cor_CD as a method of ListPairs
cor_CD = np.empty( mu_C.shape + pairs.d.shape )
for i in range(len(pairs)):
    cor_CD[:,i] = pairs[i].cor_CT(mean=mu_C, cov=cov_CC)

# Cholesky factorization
# XXX SciPy cho_factor and cho_solve preform a little better
L = np.linalg.cholesky(cov_DD_pri)
M = np.linalg.solve(L, cor_CD.T)
# Posterior mean an covariance
mu_C += np.dot(M.T, np.linalg.solve(L, pairs.d - mu_D_pri))
cov_CC -= np.dot(M.T, M)

# Posterior mean and covariance
mu_D_pst = pairs.mu_T(mu_C)
cov_DD_pst = pairs.cov_TT(mu_C, cov_CC)

# Store posterior mean and covariances
dset_mu[1,:] = mu_C
dset_sd[1,:] = np.sqrt(cov_CC.diagonal())
fh.create_dataset('cov_DD_pst', data=cov_DD_pst)
# Store posterior misfit
dset_misfit[1] = pairs.misfit(mu_C, cov_CC)

# Close HDF5 file
fh.close()
