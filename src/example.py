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

# Indices for all combinations of stations with duplicates dropped
idx, idy = np.tril_indices(stations.size, -1)

# Determine how fine great circle segments are going to be sampled
central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))
min_samples = 2
ds = central_angle.min()/min_samples # Spacing in [rad]
# Number of samples per path; to suppress duplicates subtract two
npts = np.round(central_angle/ds, 0).astype(int) - 2

# Measurement noise; standard deviation
epsilon = 0.01

# Allocate memory for sampling points
points = np.empty(npts.sum() + stations.size, dtype=dt_latlon)
# An index keeping track where we are at the array of sampling points
index = stations.size # The first entries are reserved for station coordinates
# TODO Add design points; e.g. corners of the plot
pairs = list() # Empty list to store station pairs
for i, j, n in np.nditer( (idx, idy, npts) ):
    indices = np.array( [i, ] + range(index, index+n) + [j, ] )
    st_i = stations[i]
    st_j = stations[j]
    pair_ij = StationPair(indices=indices, error=epsilon, \
                          st1=stations[i], st2=stations[j])

    # Fill array of sampling points
    points[indices] = pair_ij.great_circle_path

    # Pseudo observations
    pair_ij.T_act = simps(r_E/c_act(points[indices]), dx=pair_ij.spacing)
    pair_ij.d = pair_ij.T_act + np.random.normal(loc=0, scale=epsilon)

    # Append to list of station pairs
    pairs.append(pair_ij)
    # Increment index
    index += n

ell = 16000 # Characteristic length
tau = 40  # A priori uncertainty; standard deviation

if __name__ == '__main__':
    from gptt import gauss_kernel
    import h5py

    # A priori assumptions
    mu_C = mu_C_pri(points) # The velocity models a priori mean
    # A priori covariance
    cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

    fh = h5py.File('../dat/example.hdf5', 'w')
    dst = fh.create_dataset('stations', stations.shape, dtype=stations.dtype)
    dst[:] = stations
    pst = fh.create_dataset('points', points.shape, dtype=dt_latlon)
    pst[:] = points
    dst = fh.create_dataset('d', len(pairs) )
    dst[:] = np.array( [pair.d for pair in pairs] )

    must = fh.create_dataset('mean', (len(pairs) + 1, ) + mu_C.shape, dtype=np.float32)
    covst = fh.create_dataset('cov', (len(pairs) + 1, ) + cov_CC.shape, dtype=np.float32)

    must[0,:] = mu_C
    covst[0,:,:] = cov_CC
    # Successively consider evidence
    for i in range(len(pairs)):
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

        must[i+1,:] = mu_C
        covst[i+1,:,:] = cov_CC

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


