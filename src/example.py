#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

from sys import argv, stdout
import numpy as np
from gptt import dt_latlon, StationPair, read_station_file, ListPairs, gauss_kernel
from scipy.integrate import simps
from reference import dt_obs
from ConfigParser import ConfigParser
import h5py

# Read parameter file
config = ConfigParser()
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

# XXX Clumsy
if config.has_option('Observations', 'sort'):
    if config.get('Observations', 'sort') == 'ascending':
        pairs.sort(key=lambda p: p.central_angle)
    elif config.get('Observations', 'sort') == 'descending':
        pairs.sort(key=lambda p: p.central_angle, reverse=True)
    else:
        raise NotImplementedError


# Sampling points
points = pairs.points
# TODO Add design points; e.g. corners of the plot
#from plotting import lllat, lllon, urlat, urlon
#grid = np.rec.fromarrays(np.mgrid[lllat:urlat:3j, lllon:urlon:3j], dtype=dt_latlon)
#points = np.concatenate( (pairs.points, grid.flatten()))

# A priori assumptions
mu_C = np.full_like(points, mu, dtype=np.float) # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

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
dset_misfit = fh.create_dataset('misfit', (2, ))
# Save prior mean and standard deviation
dset_mu[0,:] = mu_C
dset_sd[0,:] = np.sqrt(cov_CC.diagonal())
# Store prior misfit
dset_misfit[0] = pairs.misfit(mu_C, cov_CC)

# Successively consider evidence
for i in range(len(pairs)):
    # To be considered evidence
    pair = pairs[i]
    # Prior mean
    mu_T = pair.T(mu_C)
    # Correlations amongst model and travel time
    # XXX Single precision causes a speed up of dot()
    cor_CT = pair.cor_CT(mean=mu_C, cov=cov_CC).astype(np.float32)
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
dset_misfit[1] = pairs.misfit(mu_C, cov_CC)
# Close HDF5 file
fh.close()

# Write parameters for being used in the LaTeX document
# TODO move the below code into its own script
with open('../def_example.tex', 'w') as fh:
    fh.write(r'\def\SFWnst{%i}' % pairs.stations.size + '\n')
    fh.write(r'\def\SFWnobs{%i}' % len(pairs) + '\n')
    fh.write(r'\def\SFWminsamples{%i}' % pairs.min_samples + '\n')
    fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(pairs.ds) + '\n')
    fh.write(r'\def\SFWtau{%i}' % tau + '\n')
    fh.write(r'\def\SFWell{%i}' % ell + '\n')
    fh.write(r'\def\SFWepsilon{%.1f}' % pairs[0].error + '\n')
    fh.write(r'\def\SFWmuCpri{%i}' % mu  + '\n')
    fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')


