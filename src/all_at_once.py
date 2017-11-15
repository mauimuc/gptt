#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__    = "Stefan Mauerberger"
__copyright__ = "Copyright (C) 2017 Stefan Mauerberger"
__license__   = "GPLv3"

''' Calculate the Bayesian posterior considering data all at once. '''

import numpy as np
from gptt import f_mu_T, f_cov_TT, gauss_kernel, misfit
from example import mu_C_pri, tau, ell, points, pairs, d


# A priori mean and assumed covariance
mu_C = mu_C_pri(points)
cov_CC = gauss_kernel(points.reshape(1,-1),points.reshape(-1,1), ell=ell, tau=tau)

# Prior predictive mean and covariance
mu_D_pri = f_mu_T(pairs, mu_C)
cov_DD_pri = f_cov_TT(pairs, mu_C, cov_CC)

# Correlations amongst data and model
cor_CD = np.empty( mu_C.shape + np.shape(d) )
for i in range(len(pairs)):
    cor_CD[:,i] = pairs[i].cor_CT(mean=mu_C, cov=cov_CC)

# Cholesky factorization
L = np.linalg.cholesky(cov_DD_pri)
M = np.linalg.solve(L, cor_CD.T)
# Posterior mean an covariance
mu_C += np.dot(M.T, np.linalg.solve(L, d - mu_D_pri))
cov_CC -= np.dot(M.T, M)

# Posterior mean and covariance
mu_D_pst = f_mu_T(pairs, mu_C)
cov_DD_pst = f_cov_TT(pairs, mu_C, cov_CC)

# Prior and posterior misfit
misfit = (misfit(d, mu_D_pri, cov_DD_pri), \
          misfit(d, mu_D_pst, cov_DD_pst))

