#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

import numpy as np
from gptt import dt_latlon, great_circle_distance, dt_float

def c_act(crd):
    ''' Toy model for the surface wave velocity to be recovered. '''
    c = np.full_like(crd, 4000, dtype=dt_float)
    x1 = np.array((66,14.5), dtype=dt_latlon)
    gcd_x1 = great_circle_distance(crd, x1)
    c += 60*np.exp(-gcd_x1/40000)
    x2 = np.array((67.5,20), dtype=dt_latlon)
    gcd_x2 = great_circle_distance(crd, x2)
    c -= 40*np.exp(-gcd_x2/65000)
    return c

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
    from file_IO import read_station_file
    from gptt import cos_central_angle, gauss_kernel, great_circle_path, line_element, dt_xyz, to_xyz, r_E
    from gptt import StationPair
    from scipy.integrate import simps


    # Read coordinates of the NORSAR Array
    stations = read_station_file('../dat/stations.dat')

    # Indices for all combinations of stations with duplicates dropped
    idx, idy = np.tril_indices(stations.size, -1)

    # Determine how fine great circle segments are going to be sampled
    central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))
    min_samples = 2
    ds = central_angle.min()/min_samples # Spacing in [rad]
    # Number of samples per path; to suppress duplicates subtract two
    npts = np.round(central_angle/ds, 0).astype(int) - 2

    # Standard deviation measurement noise
    epsilon = 0.01

    # Sampling points
    points = np.empty(npts.sum() + 20, dtype=dt_xyz)
    index = stations.size # An index keeping track where we are at the points array
    points[0:index] = stations
    # TODO Add design points; the plots corners
    pairs = list()
    for i, j, n in np.nditer( (idx, idy, npts) ):
        indices = np.array( [i, ] + range(index, index+n) + [j, ] )
        st_i = stations[i]
        st_j = stations[j]
        pair_ij = StationPair(indices=indices, \
                              lat1=st_i['lat'], lon1=st_i['lon'], \
                              lat2=st_j['lat'], lon2=st_j['lon'])
        t = np.linspace(0, pair_ij.central_angle, pair_ij.npts)
        path = great_circle_path(st_i, st_j, t)
        points[indices] = path
        pair_ij.T_act = simps(r_E/c_act(path), dx=pair_ij.spacing)
        pair_ij.d = pair_ij.T_act + np.random.normal(loc=0, scale=epsilon)
        pairs.append(pair_ij)
        # Increment index
        index += n


    # A priori assumptions
    mu_C_pri = 4000
    ell = 11000
    tau = 40
    mu_C = np.full_like(points, mu_C_pri, dtype=dt_float)
    cov_CC = tau**2*gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], ell)

    # Successively consider evidence
    a = 0
    for pair in pairs:
        a+=1
        # A prior travel time
        mu_T12 = pair.T(mu_C)
        # Correlations amongst model and travel time
        cor_CT = pair.cor_CT(mean=mu_C, cov=cov_CC)
        # Prior variance
        var_TT = pair.var_DD(mean=mu_C, cov=cov_CC)
        var_DD = var_TT + epsilon**2 # Noise level

        # Update posterior mean
        mu_C += cor_CT/var_DD*(pair.d - mu_T12)
        # Update posterior co-variance
        cov_CC -= np.dot(cor_CT[:,np.newaxis], cor_CT[np.newaxis,:])/var_DD
        print 'Combination %i ' % a # TODO plot station name




    with open('../def_example.tex', 'w') as fh:
        fh.write(r'\def\SFWnobs{%i}' % stations.size + '\n')
        fh.write(r'\def\SFWminsamples{%i}' % min_samples + '\n')
        fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(ds) + '\n')
        fh.write(r'\def\SFWtau{%i}' % tau + '\n')
        fh.write(r'\def\SFWell{%i}' % ell + '\n')
        fh.write(r'\def\SFWepsilon{%.2f}' % epsilon + '\n')
        fh.write(r'\def\SFWmuCpri{%i}' % mu_C_pri  + '\n')
        fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')
        fh.write(r'\def\SFWngrid{%i}' % 0 + '\n')


    from plotting import plt, m, lllat, lllon, urlat, urlon
    from gptt import to_latlon

    points = to_latlon(points)

    x, y = m(points['lon'], points['lat'])
    pcol = plt.tripcolor(x, y, mu_C, vmin=3940, vmax=4060, cmap='seismic', rasterized=True)
    cbar = m.colorbar(pcol, location='right', pad="5%")
    ticks = np.linspace(3950, 4050, 5)
    cbar.set_ticks(ticks)
    cbar.set_label(r'$\frac ms$', rotation='horizontal')
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', s=2)
    plt.savefig('../fig_example.pgf', bbox_inches='tight')

    pcol.remove()
    var_C = np.sqrt(cov_CC.diagonal())
    pcol = plt.tripcolor(x, y, var_C, cmap='Reds', rasterized=True)
    cbar = m.colorbar(location='right', pad="5%")
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True, marker='.', s=2)
    plt.savefig('../fig_example_var.pgf', bbox_inches='tight')


