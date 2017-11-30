#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

from sys import argv, stdout
import numpy as np
from gptt import dt_latlon, read_station_file, ListPairs, gauss_kernel
from reference import dt_obs
from ConfigParser import ConfigParser
import h5py
from random import shuffle

# Read parameter file
config = ConfigParser(defaults={'succession': 'native'})
with open(argv[1]) as fh:
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

# XXX Appears a little clumsy to me
if config.get('Observations', 'succession') == 'native':
    pass
elif config.get('Observations', 'succession') == 'ascending':
    pairs.sort(key=lambda p: p.central_angle)
elif config.get('Observations', 'succession') == 'descending':
    pairs.sort(key=lambda p: p.central_angle, reverse=True)
elif config.get('Observations', 'succession') == 'random':
    shuffle(pairs)
else:
    raise NotImplementedError


# Sampling points
points = pairs.points
# TODO Add design points; e.g. corners of the plot
#from plotting import lllat, lllon, urlat, urlon
#grid = np.rec.fromarrays(np.mgrid[lllat:urlat:5j, lllon:urlon:5j], dtype=dt_latlon)
#points = np.concatenate( (pairs.points, grid.flatten()))

# A priori assumptions
mu_C = np.full_like(points, mu, dtype=np.float) # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell)

# Open HDF5 file handle
output_file = config.get('Output', 'filename')
fh = h5py.File(output_file, 'w')
# Store stations
fh.create_dataset('stations', data=pairs.stations)
# Store discretization
fh.create_dataset('points', data=points)
# Store prior covariance matrix
fh.create_dataset('cov_CC_pri', data=cov_CC)
# Create datasets for mean, standard deviation and misfit
dset_mu = fh.create_dataset('mu', (len(pairs) + 1, ) + mu_C.shape)
dset_sd = fh.create_dataset('sd', (len(pairs) + 1, ) + mu_C.shape)
increment = 10
shp_misfit = (max(1, len(pairs)/increment) + 1, )
dt_misfit = np.dtype( [('evd', np.int), ('val', float)] )
dset_misfit = fh.create_dataset('misfit', shp_misfit, dtype=dt_misfit)
# Save prior mean and standard deviation
dset_mu[0,:] = mu_C
dset_sd[0,:] = np.sqrt(cov_CC.diagonal())
# Store prior misfit

# Successively consider evidence
for i in range(len(pairs)):
    # Misfit
    if i % 10 == 0:
        dset_misfit[i/increment] = i, pairs.misfit(mu_C, cov_CC)
    # To be considered evidence
    pair = pairs[i]
    # Prior mean
    mu_T = pair.T(mu_C)
    # Correlations amongst model and travel time
    # XXX Single precision causes a speed up of dot()
    cor_CT = pair.cor_CT(mean=mu_C, cov=cov_CC)
    # Prior variance
    var_DD = pair.var_DD(mean=mu_C, cov=cov_CC)
    # Update posterior mean
    mu_C += cor_CT/var_DD*(pair.d - mu_T)
    # Update posterior co-variance
    cov_CC -= np.dot(cor_CT[:,np.newaxis], cor_CT[np.newaxis,:])/var_DD
    # Save mean and standard deviation
    dset_mu[i+1,:] = mu_C
    dset_sd[i+1,:] = np.sqrt(cov_CC.diagonal())
    # Screen output; a very basic progress bar
    p = int(100.*(i+1)/len(pairs)) # Progress
    stdout.write('\r[' + p*'#' + (100-p)*'-' + '] %3i' % p + '%' )
stdout.write('\n')


# Save posterior covariance matrix
fh.create_dataset('cov_CC_pst', data=cov_CC)
# Store posterior misfit
dset_misfit[-1] = len(pairs), pairs.misfit(mu_C, cov_CC)
# Close HDF5 file
fh.close()

