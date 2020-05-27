import numpy as np
import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven import tgedmd as tgedmd
from examples.lemon_slice import LemonSlice
import time


""" System Settings:"""
# Number of dimensions:
d = 4
# Diffusion constant:
beta = 1.0
# Spring constant for harmonic parts of the potential
alpha = 10.0
# Pre-factor for Lemon Slice:
c = 1.0
# Number of minima for Lemon Slice:
k = 4

""" Simulation settings:"""
# Integration time step:
dt = 1e-3
# Number of time steps:
m = 300  # max = 3000
# Initial position:
x0 = np.ones(d)

""" Run Simulation"""
LS = LemonSlice(k, beta, c=c, d=d, alpha=alpha)
data = LS.Simulate(x0, m, dt)  # data.shape = (k, m)

""" Define basis functions"""
# Monomials
basis_list = []
for i in range(d):
    basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(2, 6)])

eigvals, eigtensors = tgedmd.amuset_hosvd(data, basis_list, LS.drift, LS.diffusion, return_option='eigenfunctionevals')
eigvals = eigvals.real
# print(eigvals)
print(eigtensors.shape)

# calculate implied timescales
num_timescales = 4
eigvals_sorted = np.sort(eigvals)
time_scales = [np.exp(kappa * dt) for kappa in eigvals_sorted[:num_timescales]]
time_scales.reverse()
print(time_scales)

# calculate and plot eigenfunctions
num_eigenfuns = 4
ind = np.argpartition(-eigvals, -num_eigenfuns)[-num_eigenfuns:]
ind = ind[np.argsort(eigvals[ind])]  # indices of num_eigenfuns leading eigvals





