#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from sys import stdout
from matplotlib import pyplot as plt
from matplotlib import animation
from plotting import prepare_map, lllat, lllon, urlat, urlon, cmap_mu, cmap_sd
from reference import c_act, dt_latlon
import h5py

dpi=150

fh = h5py.File('../dat/example.hdf5', 'r')

points = fh['points']
stations = fh['stations']
mu_C = fh['mu']
sd_C = fh['sd']


# Ratio 16:9
fig = plt.figure(figsize=(8,4.5))
fig.subplots_adjust(left=0.06, right=0.97, top=0.95, wspace=0.02, bottom=0.05)
ax_mu = fig.add_subplot(121)
ax_sd = fig.add_subplot(122)

# Subplot on the left
mu_delta = max(c_act._v0 - c_act.min, c_act.max - c_act._v0)
mu_vmax = (c_act._v0 + mu_delta).round(0)
mu_vmin = (c_act._v0 - mu_delta).round(0)
m = prepare_map(ax_mu)
x, y = m(points['lon'], points['lat'])
tpc_mu = ax_mu.tripcolor(x, y, mu_C[0,:], \
    vmin=mu_vmin, vmax=mu_vmax, cmap=cmap_mu, shading='gouraud')
cbar = m.colorbar(tpc_mu, location='bottom')
cbar.set_ticks( range(mu_vmin.astype(np.int), mu_vmax.astype(np.int), 40)[1:])
#cbar.set_label('mean')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# Make a lat, lon grid with extent of the map
N = 60j
grid = np.rec.fromarrays(np.mgrid[lllat:urlat:N, lllon:urlon:N], dtype=dt_latlon)
c = c_act(grid) # Actual velocity model
# Contour lines
cnt = m.contour(grid['lon'], grid['lat'], c, levels=c_act.levels(20), latlon=True, colors='k', linewidths=0.5)



# Subplot right
m = prepare_map(ax_sd, pls=[0,0,0,0])
tpc_sd = ax_sd.tripcolor(x, y, sd_C[0,:], \
    vmin=np.min(sd_C), vmax=np.max(sd_C), cmap=cmap_sd, shading='gouraud')
cbar = m.colorbar(tpc_sd, location='bottom')
vmin_sd = np.min(sd_C).round().astype(np.integer)
vmax_sd = np.max(sd_C).round().astype(np.integer)
cbar.set_ticks(range(vmin_sd, vmax_sd, 5))
#cbar.set_label('standard deviation')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# First frame; Necessary for LaTeX beamer
plt.savefig('../animation_pri.png', dpi=dpi)

def animate(i):
    global mu_C, cov_CC

    tpc_mu.set_array(mu_C[i,:])
    tpc_sd.set_array(sd_C[i,:])
    # Screen output; a very basic progress bar
    p = int(100.*(i+1)/mu_C.shape[0]) # Progress
    stdout.write('\r[' + p*'#' + (100-p)*'-' + '] %3i' % p + '%' )
    if (i+1) == mu_C.shape[0]:
        stdout.write('\n')

    return tpc_mu, tpc_sd


frames = mu_C.shape[0]
duration = 30. # s
interval = 1000.*duration/frames # ms

anim = animation.FuncAnimation(fig, animate, save_count=0, \
                               frames=frames, interval=interval, blit=False)

# Save video
anim.save('../animation.avi', dpi=dpi, extra_args=['-vcodec', 'msmpeg4v2'])

# Last frame; Necessary for LaTeX beamer
plt.savefig('../animation_pst.png', dpi=dpi)


#plt.close()

