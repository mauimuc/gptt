#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance, cos_central_angle, r_E, StationPair, read_station_file, ListPairs
from scipy.integrate import simps
from reference import dt_obs, err_obs


def mu_C_pri(crd):
    ''' A priori velocity [m/s] '''
    return np.full_like(crd, 4e3, dtype=np.float)


# Read station coordinates
all_stations = read_station_file('../dat/stations.dat')

# Read pseudo data
pseudo_data = np.genfromtxt('../dat/pseudo_data.dat', dtype=dt_obs)


# TODO Estimate hyper-parameters
# TODO Use a parameter file
# Kernel parameters
ell = 16000 # Characteristic length
tau = 40  # A priori uncertainty; standard deviation

# Observations
pairs = ListPairs(pseudo_data, all_stations)


# Observations
d = [pair.d for pair in pairs]

if __name__ == '__main__':
    from gptt import gauss_kernel, f_mu_T, f_cov_TT, misfit
    import h5py

    # Sampling points
    points = pairs.points
    # TODO Add design points; e.g. corners of the plot
    #from plotting import lllat, lllon, urlat, urlon
    #grid = np.rec.fromarrays(np.mgrid[lllat:urlat:3j, lllon:urlon:3j], dtype=dt_latlon)
    #points = np.concatenate( (pairs.points, grid.flatten()))

    # A priori assumptions
    mu_C = mu_C_pri(points) # The velocity models a priori mean
    # A priori covariance
    cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

    # Open HDF5 file handle
    fh = h5py.File('../dat/example.hdf5', 'w')
    # Store stations
    fh.create_dataset('stations', data=pairs.stations)
    # Store discretization
    fh.create_dataset('points', data=points)
    # Store prior covariance matrix
    fh.create_dataset('cov_CC_pri', data=cov_CC, dtype=np.float32)
    # Create datasets for mean, standard deviation and misfit
    dset_mu = fh.create_dataset('mu', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)
    dset_sd = fh.create_dataset('sd', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)
    dset_misfit = fh.create_dataset('misfit', (len(pairs) + 1, ), dtype=np.float32)
    # Save prior mean and standard deviation
    dset_mu[0,:] = mu_C
    dset_sd[0,:] = np.sqrt(cov_CC.diagonal())
    # Calculate and save prior misfit
    #mu_T = f_mu_T(pairs, mu_C)
    #cov_TT = f_cov_TT(pairs, mu_C, cov_CC)
    #dset_misfit[0] = misfit(d, mu_T, cov_TT)

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
        # Screen output
        print 'Combination', pair, '%3i/%3i' % (i, len(pairs))
        # Save mean and standard deviation
        dset_mu[i+1,:] = mu_C
        dset_sd[i+1,:] = np.sqrt(cov_CC.diagonal())
        # Calculate and save misfit
        #mu_T = f_mu_T(pairs, mu_C)
        #cov_TT = f_cov_TT(pairs, mu_C, cov_CC)
        #dset_misfit[i+1] = misfit(d, mu_T, cov_TT)

    # Save posterior covariance matrix
    fh.create_dataset('cov_CC_pst', data=cov_CC)
    # Close HDF5 file
    fh.close()

    # Write parameters for being used in the LaTeX document
    with open('../def_example.tex', 'w') as fh:
        fh.write(r'\def\SFWnst{%i}' % pairs.stations.size + '\n')
        fh.write(r'\def\SFWnobs{%i}' % len(pairs) + '\n')
        fh.write(r'\def\SFWminsamples{%i}' % pairs.min_samples + '\n')
        fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(pairs.ds) + '\n')
        fh.write(r'\def\SFWtau{%i}' % tau + '\n')
        fh.write(r'\def\SFWell{%i}' % ell + '\n')
        fh.write(r'\def\SFWepsilon{%.1f}' % pairs[0].error + '\n')
        fh.write(r'\def\SFWmuCpri{%i}' % mu_C_pri(1)  + '\n')
        fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')


