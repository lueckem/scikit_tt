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
m = 500
# Initial position:
x0 = np.ones(d)

""" Run Simulation"""
LS = LemonSlice(k, beta, c=c, d=d, alpha=alpha)
data = LS.Simulate(x0, m, dt)  # data.shape = (k, m)

""" Define basis functions"""
# Monomials
basis_list = []
for i in range(d):
    basis_list.append([tdt.Identity(i)] + [tdt.Monomial(i, j) for j in range(1, 5)])

eigvals, eigtensors = tgedmd.amuset_reversible_exact(data, basis_list, LS.diffusion)
print(eigvals)


# eigenvalues, eigentensors = tedmd.amuset_hosvd(data, range(0, m - 1), range(1, m), basis_list)
# print(eigenvalues)
