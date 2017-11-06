#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from example import pairs, mu_C_pri, points, stations
from gptt import gauss_kernel
from plotting import prepare_map

# A priori assumptions
ell = 11000 # Characteristic length
tau = 40  # A priori uncertainty; standard deviation
mu_C = mu_C_pri(points).astype('float32') # The velocity models a priori mean
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

fig = plt.figure(figsize=(8,4.5))
fig.subplots_adjust(left=0.07, right=0.97, top=0.96)
ax_mu = fig.add_subplot(121)
ax_sd = fig.add_subplot(122)

m = prepare_map(ax_mu)
x, y = m(points['lon'], points['lat'])
tpc_mu = ax_mu.tripcolor(x, y, mu_C, \
    vmin=3940, vmax=4060, cmap='seismic', shading='gouraud')
cbar = m.colorbar(tpc_mu, location='bottom')
ticks = np.linspace(3950, 4050, 3)
cbar.set_ticks(ticks)
cbar.set_label('mean')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0)

m = prepare_map(ax_sd)
tpc_sd = ax_sd.tripcolor(x, y, np.sqrt(cov_CC.diagonal()), \
    vmin=20, vmax=40, cmap='Reds', shading='gouraud')
cbar = m.colorbar(tpc_sd, location='bottom')
cbar.set_label('standard deviation')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0)

def init():
    return tpc_mu, tpc_sd

def animate(i):
    global mu_C, cov_CC
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
    print 'Combination %5s -- %-5s %3i/%3i' % ( pair.st1['stnm'], pair.st2['stnm'], i, len(pairs) )
    tpc_mu.set_array(mu_C)
    tpc_sd.set_array(np.sqrt(cov_CC.diagonal()))
    return tpc_mu, tpc_sd

anim = animation.FuncAnimation(fig, animate, save_count=0, \
                               frames=len(pairs), interval=100, blit=False)

anim.save('../example.mp4', dpi=150)

plt.close()


