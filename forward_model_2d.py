#!/bin/bash

import time

t69 = time.time()

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from functions import *

data = np.loadtxt('./marmousi.dat')
v1 = data.copy()

nz, nx = v1.shape

dx, dz = 20, 40

n_pml = 30
nz, nx = nz + n_pml, nx + 2*n_pml

z = np.arange(nz) * dz
x = np.arange(nx) * dx

vel = np.zeros((nz, nx))
vel[:-n_pml, n_pml:-n_pml] = v1

vel[:-n_pml, :n_pml] = np.repeat(np.vstack(v1[:, 0]), n_pml, axis = 1)
vel[:-n_pml, -n_pml:] = np.repeat(np.vstack(v1[:, -1]), n_pml, axis = 1)

vel1 = vel[-n_pml-1, :]
vel1.shape = (1, vel1.size)

vel[-n_pml:, :] = np.repeat(vel1, n_pml, axis = 0)

v = vel
d = np.power(vel, .25) * 310 

Fp = 3
df = .1
fmax = 12
F = np.arange(df, fmax+df, df)
nf = len(F)
src = (2/np.sqrt(np.pi))  *  (F**2/Fp**3)  *  np.exp(-((F**2/Fp**2)))

x_src = [int((nx-2*n_pml)*dx//2)]
z_src = 5

x_rec = np.arange(12, int((nx-2*n_pml)*dx), 12)
z_rec = 10

Ts, Tr = src_rec(x_src, z_src, x_rec, z_rec, dx, dz, nx, nz, n_pml)

P13, data13 = forward_solver_13(F, v, d, src, Ts, Tr, dx, dz, n_pml)
P5, data5 = forward_solver_5(F, v, d, src, Ts, Tr, dx, dz, n_pml)

fname13 = './results/wavefields13.png'
fname5 = './results/wavefields5.png'

plot_wavefields(P13, x_src, z_src, n_pml, dx, dz, F, fname13)
plot_wavefields(P5, x_src, z_src, n_pml, dx, dz, F, fname5)

t, seis13 = freq2time(data13, F)
t, seis5 = freq2time(data5, F)

fname13 = './results/seismogram13.png'
fname5 = './results/seismogram5.png'

plot_seismogram(seis13, t, nx, n_pml, dx, z_rec, fname13)
plot_seismogram(seis5, t, nx, n_pml, dx, z_rec, fname5)



t96 = time.time()
print(f'The wave propagation modeling was completed in {(int(t96-t69)//60)} min, {int((t96-t69)%60)} sec')





