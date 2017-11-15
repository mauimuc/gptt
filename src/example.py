#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance, cos_central_angle, r_E, StationPair, read_station_file
from scipy.integrate import simps

def c_act(crd):
    ''' Toy model for the surface wave velocity to be recovered. '''
    c = np.full_like(crd, 4000., dtype=np.float)
    x1 = np.array((66,14.5), dtype=dt_latlon)
    gcd_x1 = great_circle_distance(crd, x1)
    c += 60*np.exp(-gcd_x1/40000)
    x2 = np.array((67.5,20), dtype=dt_latlon)
    gcd_x2 = great_circle_distance(crd, x2)
    c -= 40*np.exp(-gcd_x2/65000)
    return c

def mu_C_pri(crd):
    return np.full_like(crd, 4000, dtype=np.float)


# Read station coordinates
stations = read_station_file('../dat/stations.dat')[::2]

# Read pseudo data
# XXX Have that dtype ready
pseudo_data = np.genfromtxt('../dat/pseudo_data.dat', \
    dtype=[('stnm1', 'S5'), ('stnm2', 'S5'), ('tt', '<f4'), ('err', '<f4')])

# Determine how fine great circle segments are going to be sampled
# FIXME shall be derived from records
# Indices for all combinations of stations with duplicates dropped
idx, idy = np.tril_indices(stations.size, -1)
central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))
min_samples = 2
ds = central_angle.min()/min_samples # Spacing in [rad]

# TODO Estimate hyper-parameters
# Measurement noise; standard deviation
epsilon = 0.02 # Separate the residual term from measurement noise
# Kernel parameters
ell = 16000 # Characteristic length
tau = 40  # A priori uncertainty; standard deviation

pairs = list() # Empty list to store station pairs
# An index keeping track how many sampling points we have
index = stations.size # The first entries are reserved for station coordinates
for stnm1, stnm2, tt, err in pseudo_data:
    mask = np.logical_or(stations['stnm'] == stnm1, stations['stnm'] == stnm2)
    st1, st2 = stations[mask]
    idx1, idx2 = np.where(mask)[0]
    n = np.round(np.arccos(cos_central_angle(st1, st2))/ds, 0).astype(np.int)
    indices = np.array( [idx1, ] + range(index, index+n-2) + [idx2, ] )
    pair = StationPair(st1=st1, st2=st2, indices=indices, error=epsilon)
    # Pseudo observations
    pair.d = tt + err
    # True travel time
    pair.T_act = tt
    # Append to list of station pairs
    pairs.append(pair)
    # Increment index
    index += n-2

# Count number of sampling points
N_points = stations.size + np.sum([pair.npts - 2 for pair in pairs])
# Allocate memory for sampling points
points = np.empty( N_points, dtype=dt_latlon)
# Assign station coordinates
points[:stations.size] = stations.astype(dt_latlon)
# TODO Add design points; e.g. corners of the plot
# Fill array with sampling points
for pair in pairs:
    points[pair.indices] = pair.great_circle_path

# Observations
d = [pair.d for pair in pairs]

if __name__ == '__main__':
    from gptt import gauss_kernel, f_mu_T, f_cov_TT, misfit
    import h5py

    # A priori assumptions
    mu_C = mu_C_pri(points) # The velocity models a priori mean
    # A priori covariance
    cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

    # Open HDF5 file handle
    fh = h5py.File('../dat/example.hdf5', 'w')
    # Store stations
    fh.create_dataset('stations', data=stations)
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
        # Save mean and standard deviation
        dset_mu[i+1,:] = mu_C
        dset_sd[i+1,:] = np.sqrt(cov_CC.diagonal())
        # Calculate and save misfit
        mu_T = f_mu_T(pairs, mu_C)
        cov_TT = f_cov_TT(pairs, mu_C, cov_CC)
        dset_misfit[i+1] = misfit(d, mu_T, cov_TT)

    # Save posterior covariance matrix
    fh.create_dataset('cov_CC_pst', data=cov_CC)
    # Close HDF5 file
    fh.close()

    # Write parameters for being used in the LaTeX document
    with open('../def_example.tex', 'w') as fh:
        fh.write(r'\def\SFWnst{%i}' % stations.size + '\n')
        fh.write(r'\def\SFWnobs{%i}' % len(pairs) + '\n')
        fh.write(r'\def\SFWminsamples{%i}' % min_samples + '\n')
        fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(ds) + '\n')
        fh.write(r'\def\SFWtau{%i}' % tau + '\n')
        fh.write(r'\def\SFWell{%i}' % ell + '\n')
        fh.write(r'\def\SFWepsilon{%.2f}' % epsilon + '\n')
        fh.write(r'\def\SFWmuCpri{%i}' % mu_C_pri(1)  + '\n')
        fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')


