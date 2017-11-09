#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from example import pairs, mu_C_pri, points, stations, ell, tau
from gptt import gauss_kernel
from plotting import prepare_map

# The velocity models a priori mean
mu_C = mu_C_pri(points).astype('float32')
# A priori covariance
cov_CC = gauss_kernel(points[:,np.newaxis], points[np.newaxis,:], tau, ell).astype('float32')

# Ratio 16:9
fig = plt.figure(figsize=(8,4.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.95, wspace=0.02, bottom=0.05)
ax_mu = fig.add_subplot(121)
ax_sd = fig.add_subplot(122)

# Subplot on the left
m = prepare_map(ax_mu)
x, y = m(points['lon'], points['lat'])
tpc_mu = ax_mu.tripcolor(x, y, mu_C, \
    vmin=3940, vmax=4060, cmap='seismic', shading='gouraud')
cbar = m.colorbar(tpc_mu, location='bottom')
ticks = np.linspace(3950, 4050, 3)
cbar.set_ticks(ticks)
#cbar.set_label('mean')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# Subplot right
m = prepare_map(ax_sd, pls=[0,0,0,0])
tpc_sd = ax_sd.tripcolor(x, y, np.sqrt(cov_CC.diagonal()), \
    vmin=15, vmax=40, cmap='Purples', shading='gouraud')
cbar = m.colorbar(tpc_sd, location='bottom')
cbar.set_ticks([15, 20, 25, 30, 35, 40])
#cbar.set_label('standard deviation')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# First frame; Necessary for LaTeX beamer
plt.savefig('../animation_pri.png', dpi=150)

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

# Save video
anim.save('../animation.mp4', dpi=150)

# Last frame; Necessary for LaTeX beamer
plt.savefig('../animation_pst.png', dpi=150)


#plt.close()

