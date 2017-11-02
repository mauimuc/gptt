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

if __name__ == '__main__':
    from file_IO import read_station_file
    from gptt import cos_central_angle, gauss_kernel, great_circle_path, line_element, dt_xyz, to_xyz, r_E
    from scipy.integrate import simps


    # Read coordinates of the NORSAR Array
    stations = read_station_file('../dat/stations.dat')

    # Indices for all combinations of stations with duplicates dropped
    idx, idy = np.tril_indices(stations.size, -1)

    # Determine how fine great circle segments are going to be sampled
    central_angle = np.arccos(cos_central_angle(stations[idx], stations[idy]))
    min_samples = 2
    ds = central_angle.min()/min_samples # Spacing in [rad]
    # Number of samples per path
    npts = np.round(central_angle/ds, 0).astype(int)
    # Indices for array slicing
    index = npts.cumsum() - npts

    # Allocate array for sampling points
    # FIXME There is quite a bunch of duplicates
    points = np.empty(npts.sum(), dtype=dt_xyz)
    # Allocate memory for path parametrization
    ts = np.empty_like(points, dtype=dt_float)

    # Calculate sampling points and parametrization
    it = np.nditer( (stations[idx], stations[idy], central_angle, npts, index) )
    for st1, st2, ca, n, i in it:
        slc = slice(i,i+n,1)
        t = np.linspace(0, ca, n)
        ts[slc] = t
        points[slc] = great_circle_path(st1, st2, t)

    # Actual velocity model
    c = c_act(points)
    # A priori assumptions
    # TODO add a grid of design points
    N = 35
    lat, lon = np.mgrid[64.5:69.5:N*1j, 11.5:23:N*1j]
    design_points = np.rec.fromarrays( (lat, lon), dtype=dt_latlon)
    points = np.concatenate( (points, to_xyz(design_points).flatten()) )
    mu_C_pri = 4000
    mu_C = np.full_like(points, mu_C_pri, dtype=dt_float)

    ell = 11000
    tau = 40
    cov_CC = tau**2*gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], ell)

    # Calculate actual and prior travel times
    it = np.nditer( (stations[idx], stations[idy], index, npts, None, None), \
                    op_dtypes=(None, None, None, None, dt_float, dt_float) )
    for st1, st2, i, n, T12, mu_T12_pri in it:
        slc = slice(i,i+n,1) # Slice
        c12 = c[slc] # Velocity along the path
        mu_C12 = mu_C[slc] # Prior velocity
        t12 = ts[slc] # Discretization
        T12[...] = simps(r_E/c12, t12) # Actual travel time
        mu_T12_pri[...] = simps(r_E/mu_C12, t12) # Prior travel time

    epsilon = 0.01

    T_act, mu_T_pri  = it.operands[-2:]
    # Pseudo travel time observations
    D = T_act + np.random.normal(loc=0, scale=epsilon, size=T_act.size).astype(dt_float)

    def misfit():
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


    it = np.nditer((stations[idx], stations[idy], D, index, npts))
    a = 0
    #m = np.empty(191)
    #m[0] = misfit()
    for st1, st2, D12, i, n in it:
        a+=1
        slc = slice(i,i+n,1) # Slice
        t12 = ts[slc] # Discretization
        mu_T12 = simps(r_E/mu_C[slc], t12) # Prior travel time
        # Correlations amongst model and travel times
        cor_CT = -simps(cov_CC[:,slc]*r_E/mu_C[slc]**2, t12, axis=-1).astype(dt_float)
        # Prior variance
        var_TT = -simps(cor_CT[slc]*r_E/mu_C[slc]**2, t12, axis=-1)
        var_DD = var_TT + epsilon**2 # Noise level

        # Update posterior mean
        mu_C += cor_CT/var_DD*(D12 - mu_T12)
        # Update posterior co-variance
        cov_CC -= np.dot(cor_CT[:,np.newaxis], cor_CT[np.newaxis,:])/var_DD
        #m[a] = misfit()
        print 'Combination %i ' % a




    with open('../def_example.tex', 'w') as fh:
        fh.write(r'\def\SFWnobs{%i}' % stations.size + '\n')
        fh.write(r'\def\SFWminsamples{%i}' % min_samples + '\n')
        fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(ds) + '\n')
        fh.write(r'\def\SFWtau{%i}' % tau + '\n')
        fh.write(r'\def\SFWell{%i}' % ell + '\n')
        fh.write(r'\def\SFWepsilon{%.2f}' % epsilon + '\n')
        fh.write(r'\def\SFWmuCpri{%i}' % mu_C_pri  + '\n')
        fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')
        fh.write(r'\def\SFWngrid{%i}' % N + '\n')


    from plotting import plt, m, lllat, lllon, urlat, urlon

    lat, lon = np.mgrid[lat.min():lat.max():(N+1)*1j, lon.min():lon.max():(N+1)*1j]
    C = mu_C[-N**2:].reshape( (N,N) )
    pcol = m.pcolormesh(lon, lat, C, vmin=3940, vmax=4060, cmap='seismic', rasterized=True, latlon=True)
    cbar = m.colorbar(location='right', pad="5%")
    ticks = np.linspace(3950, 4050, 5)
    cbar.set_ticks(ticks)
    cbar.set_label(r'$\frac ms$', rotation='horizontal')
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True)
    plt.savefig('../fig_example.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)

    pcol.remove()

    var_C = np.sqrt(cov_CC.diagonal()[-N**2:].reshape( (N,N) ))
    pcol = m.pcolormesh(lon, lat, var_C, rasterized=True, latlon=True, vmin=25, vmax=40, cmap='Reds')
    cbar = m.colorbar(location='right', pad="5%")
    cbar.solids.set_edgecolor("face")
    m.scatter(stations['lon'], stations['lat'], latlon=True)
    plt.savefig('../fig_example_var.pgf', transparent=True, bbox_inches='tight', pad_inches=0.01)


