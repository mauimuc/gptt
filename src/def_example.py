#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Example script of a synthetic test for Bayesian travel time tomography '''

from sys import argv
import numpy as np
from gptt import read_station_file, ListPairs
from reference import dt_obs
from ConfigParser import ConfigParser

# Read parameter file
config = ConfigParser()
with open(argv[1]) as fh:
    config.readfp(fh)


# Kernel Parameters
# TODO Estimate hyper-parameters
tau = config.getfloat('Prior', 'tau') # A priori uncertainty; standard deviation
ell = config.getfloat('Prior', 'ell') # Characteristic length
mu  = config.getfloat('Prior', 'mu') # Constant a priori velocity model

# Read station coordinates
station_file = config.get('Observations', 'station_file')
all_stations = read_station_file(station_file)

# Read pseudo data
data_file = config.get('Observations', 'data')
pseudo_data = np.genfromtxt(data_file, dtype=dt_obs)

# Observations
pairs = ListPairs(pseudo_data, all_stations)

# Sampling points
points = pairs.points

# Write parameters for being used in the LaTeX document
with open('../def_example.tex', 'w') as fh:
    fh.write(r'\def\SFWnst{%i}' % pairs.stations.size + '\n')
    fh.write(r'\def\SFWnobs{%i}' % len(pairs) + '\n')
    fh.write(r'\def\SFWminsamples{%i}' % pairs.min_samples + '\n')
    fh.write(r'\def\SFWdeltaangle{%.3f}' % np.rad2deg(pairs.ds) + '\n')
    fh.write(r'\def\SFWtau{%i}' % tau + '\n')
    fh.write(r'\def\SFWell{%i}' % ell + '\n')
    fh.write(r'\def\SFWepsilon{%.1f}' % pairs[0].error + '\n')
    fh.write(r'\def\SFWmuCpri{%i}' % mu  + '\n')
    fh.write(r'\def\SFWnpts{%i}' % points.size + '\n')


