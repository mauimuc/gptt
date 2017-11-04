#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance

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

def misfit():
    # XXX Does no longer work
    # TODO Adopt to OO approach
    cov_DD = np.zeros( (190,190) )
    mu_D = np.zeros( 190 )
    for i in range(index.size):
        slc_i = slice(index[i], index[i] + npts[i], 1) # Slice
        t_i = ts[slc_i] # Discretization
        mu_D[i] = simps(r_E/mu_C[slc_i], t_i) # travel time
        cor = -simps(cov_CC[:-N**2,slc_i]*r_E/mu_C[slc_i]**2, t_i, axis=-1)
        for j in range(i, index.size):
            slc_j = slice(index[j], index[j] + npts[j], 1) # Slice
            t_j = ts[slc_j] # Discretization
            cov = -simps(cor[slc_j]*r_E/mu_C[slc_j]**2, t_j, axis=-1)
            cov_DD[i,j] = cov
            if i!=j:
                cov_DD[j,i] = cov
    L = np.linalg.cholesky(cov_DD)
    return (np.linalg.solve(L, D - mu_D)**2).sum()


if __name__ == '__main__':
    from file_IO import read_station_file2
    from gptt import cos_central_angle, gauss_kernel, r_E, StationPair
    from scipy.integrate import simps


    # Read station coordinates
    stations = read_station_file2('../dat/stations.dat')[:30]

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


    # A priori assumptions
    ell = 11000 # Characteristic length
    tau = 40  # A priori uncertainty; standard deviation
    mu_C = mu_C_pri(points) # The velocity models a priori mean
    # A priori covariance
    cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

    # Successively consider evidence
    for i, pair in enumerate(pairs):
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
        print 'Combination %4s -- %-4s %3i/%3i' % ( pair.st1['stnm'], pair.st2['stnm'], i, len(pairs) )


    # Write parameters for being used in the LaTeX document
    with open('../def_example.tex', 'w') as fh:
        fh.write(r'\def\SFWnobs{%i}' % stations.size + '\n')
        fh.write(r'\def\SFWminsamples{%i}' % min_samples + '\n')
        fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(ds) + '\n')
        fh.write(r'\def\SFWtau{%i}' % tau + '\n')
        fh.write(r'\def\SFWell{%i}' % ell + '\n')
        fh.write(r'\def\SFWepsilon{%.2f}' % epsilon + '\n')
        fh.write(r'\def\SFWmuCpri{%i}' % mu_C_pri(1)  + '\n')
        fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')


    from plotting import plt, prepare_map


    m = prepare_map()
    x, y = m(points['lon'], points['lat'])
    pcol = plt.tripcolor(x, y, mu_C, vmin=3940, vmax=4060, cmap='seismic', rasterized=True)
    cbar = m.colorbar(pcol, location='right', pad="5%")
    ticks = np.linspace(3950, 4050, 5)
    cbar.set_ticks(ticks)
    cbar.set_label(r'$\frac ms$', rotation='horizontal')
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', s=2)
    plt.savefig('../fig_example_mu.pgf', bbox_inches='tight')
    plt.close()

    m = prepare_map()
    var_C = np.sqrt(cov_CC.diagonal())
    pcol = plt.tripcolor(x, y, var_C, cmap='Reds', rasterized=True)
    cbar = m.colorbar(location='right', pad="5%")
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', s=2)
    plt.savefig('../fig_example_sd.pgf', bbox_inches='tight')


