#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from plotting import prepare_map
import h5py


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
m = prepare_map(ax_mu)
x, y = m(points['lon'], points['lat'])
tpc_mu = ax_mu.tripcolor(x, y, mu_C[0,:], \
    vmin=3940, vmax=4060, cmap='seismic', shading='gouraud')
cbar = m.colorbar(tpc_mu, location='bottom')
ticks = np.linspace(3950, 4050, 3)
cbar.set_ticks(ticks)
#cbar.set_label('mean')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# Subplot right
m = prepare_map(ax_sd, pls=[0,0,0,0])
tpc_sd = ax_sd.tripcolor(x, y, sd_C[0,:], \
    vmin=15, vmax=40, cmap='Purples', shading='gouraud')
cbar = m.colorbar(tpc_sd, location='bottom')
cbar.set_ticks([15, 20, 25, 30, 35, 40])
#cbar.set_label('standard deviation')
m.scatter(stations['lon'], stations['lat'], latlon=True, lw=0, color='g')

# First frame; Necessary for LaTeX beamer
plt.savefig('../animation_pri.png', dpi=150)

def animate(i):
    global mu_C, cov_CC

    tpc_mu.set_array(mu_C[i,:])
    tpc_sd.set_array(sd_C[i,:])
    print('Frame %i of %i' % (i+1, mu_C.shape[0]))

    return tpc_mu, tpc_sd


anim = animation.FuncAnimation(fig, animate, save_count=0, \
                               frames=mu_C.shape[0], interval=100, blit=False)

# Save video
anim.save('../animation.mp4', dpi=150, extra_args=['-vcodec', 'libx264'])

# Last frame; Necessary for LaTeX beamer
plt.savefig('../animation_pst.png', dpi=150)


#plt.close()

