import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve, eigs


"""
  ================================================================================== FORWARD WAVE PROPAGATION (FDFD) 13-POINT STENCIL =================================================================================================
"""

def forward_solver_13(F, v, d, src, Ts, Tr, dx, dz, n_pml):
    
    nz, nx = v.shape
    
    P = np.empty((nz, nx))
    P = np.atleast_3d(P)

    n_rec, temp = Tr.shape
    data = np.empty((1, n_rec))
    data = np.atleast_3d(data)

    K = np.power(v, 2) * d

    d1 = np.ones((nz+4, nx+4))
    d1[:-4, 2:-2] = d 

    d2 = d1[-5, 2:-2]
    d2.shape = (1, d2.size)

    d1[-4:, 2:-2] = np.repeat(d2, 4, axis = 0)
    d1[:, :2] = np.repeat(np.vstack(d1[:, 3]), 2, axis = 1)
    d1[:, -2:] = np.repeat(np.vstack(d1[:, -3]), 2, axis = 1)

    b = 1/d1 

    bi = np.zeros((nz+4, nx+3))
    bj = np.zeros((nz+3, nx+4))

    for i in range(nz+3):
        bj[i, :] = 1/2 * (b[i+1, :] + b[i, :]) 

    for i in range(nx+3):
        bi[:, i] = 1/2 * (b[:, i+1] + b[:, i]) 

    bj = bj[:, 2:-2]
    bi = bi[2:-2, :]

    
    for k in range(len(F)):
        w = (2 * np.pi * F[k]) 

        z1 = np.zeros(nz+2-n_pml)
        z1 = np.append(z1, np.arange(n_pml-1, -3, -1)) * dz
        x1 = np.arange(-2, n_pml)
        x1 = np.append(x1, np.zeros(nx-2*n_pml))
        x1 = np.append(x1, np.arange(n_pml-1, -3, -1)) * dx

        c_pml_x = np.zeros(nx+4)
        c_pml_z = np.zeros(nz+4)

        c_pml1 = 20

        c_pml_x[:n_pml+2] = c_pml1
        c_pml_x[-n_pml-2:] = c_pml1
        c_pml_z[-n_pml+2:] = c_pml1


        Lx = n_pml * dx
        Lz = n_pml * dz

        gx = 1 + (1j * c_pml_x * np.cos(np.pi/2 * x1/Lx) / w)
        gz = 1 + (1j * c_pml_z * np.cos(np.pi/2 * z1/Lz) / w)
        gz.shape = (nz+4, 1)


        gxi = 1/2 * (gx[1:] + gx[:-1])
        gzj = 1/2 * (gz[1:] + gz[:-1])
        
        C1 = (w**2/K) - (1/(gx[2:-2] * dx**2)) * ((1/24 * 1/24 * (bi[:, :-3]/gxi[:-3] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 2:-1]/gxi[2:-1]))) -\
              (1/(gz[2:-2] * dz**2)) * ((1/24 * 1/24 * (bj[:-3, :]/gzj[:-3] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * (bj[1:-2, :]/gzj[1:-2] + bj[2:-1, :]/gzj[2:-1])))

        C2 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, :-3]/gxi[:-3])
        C3 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, :-3]/gxi[:-3]))
        C4 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, :-3]/gxi[:-3])) + (9/8 * 9/8 * bi[:, 1:-2]/gxi[1:-2]))
        C5 = (1/(gx[2:-2] * dx**2)) * ((9/8 * 1/24 * (bi[:, 1:-2]/gxi[1:-2] + bi[:, 3:]/gxi[3:])) + (9/8 * 9/8 * bi[:, 2:-1]/gxi[2:-1]))
        C6 = (-1/(gx[2:-2] * dx**2)) * (9/8 * 1/24 * (bi[:, 2:-1]/gxi[2:-1] + bi[:, 3:]/gxi[3:]))
        C7 = (1/(gx[2:-2] * dx**2)) * (1/24 * 1/24 * bi[:, 3:]/gxi[3:])
        C8 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[:-3, :]/gzj[:-3])
        C9 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[:-3, :]/gzj[:-3]))
        C10 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[:-3, :]/gzj[:-3])) + (9/8 * 9/8 * bj[1:-2, :]/gzj[1:-2]))
        C11 = (1/(gz[2:-2] * dz**2)) * ((9/8 * 1/24 * (bj[1:-2, :]/gzj[1:-2] + bj[3:, :]/gzj[3:])) + (9/8 * 9/8 * bj[2:-1, :]/gzj[2:-1]))
        C12 = (-1/(gz[2:-2] * dz**2)) * (9/8 * 1/24 * (bj[2:-1, :]/gzj[2:-1] + bj[3:, :]/gzj[3:]))
        C13 = (1/(gz[2:-2] * dz**2)) * (1/24 * 1/24 * bj[3:, :]/gzj[3:])

        C1 = C1.flatten()   # (i, j)
        C2 = C2.flatten()   # (i-3, j)
        C3 = C3.flatten()   # (i-2, j)
        C4 = C4.flatten()   # (i-1, j)
        C5 = C5.flatten()   # (i+1, j)
        C6 = C6.flatten()   # (i+2, j)
        C7 = C7.flatten()   # (i+3, j)
        C8 = C8.flatten()   # (i, j-3)
        C9 = C9.flatten()   # (i, j-2)
        C10 = C10.flatten() # (i, j-1)
        C11 = C11.flatten() # (i, j+1)
        C12 = C12.flatten() # (i, j+2)
        C13 = C13.flatten() # (i, 2j+3)


        M = diags([C8[3*nx:], C9[2*nx:], C10[nx:], C2[3:], C3[2:], C4[1:], C1, C5, C6, C7, C11, C12, C13], [-3*nx, -2*nx, -nx, -3, -2, -1, 0, 1, 2, 3, nx, 2*nx, 3*nx], shape = (nx*nz, nx*nz), format = 'csc')


        s = Ts * src[k]
        s = s.flatten()

        # solving the matrix equation

        p1 = spsolve(M, s)
        p2 = p1.copy()
        p2.shape = (nx*nz, 1)
        
        n_rec, temp = Tr.shape

        data1 = np.matmul(Tr, p1)
        data1.shape = (1, data1.size, 1)
        data = np.append(data, data1, axis = 2)

        p = p1.reshape(nz, nx)
        P = np.append(P, np.atleast_3d(p), axis = 2)

    P = np.delete(P, 0, 2)
    data = np.delete(data, 0, 2)
    
    data.shape = (1, n_rec, len(F))
    return P, data


"""
  ================================================================================== FORWARD WAVE PROPAGATION (FDFD) 5-POINT STENCIL =================================================================================================
"""

def forward_solver_5(F, v, d, src, Ts, Tr, dx, dz, n_pml):
    
    nz, nx = v.shape
    
    P = np.empty((nz, nx))
    P = np.atleast_3d(P)
    n_rec, temp = Tr.shape

    data = np.empty((1, n_rec))
    data = np.atleast_3d(data)

    K = np.power(v, 2) * d 

    d1 = np.ones((nz+2, nx+2))
    d1[:-2, 1:-1] = d 

    d2 = d1[-3, 1:-1]
    d2.shape = (1, d2.size)

    d1[-2:, 1:-1] = np.repeat(d2, 2, axis = 0)
    d1[:, :1] = np.repeat(np.vstack(d1[:, 1]), 1, axis = 1)
    d1[:, -1:] = np.repeat(np.vstack(d1[:, -2]), 1, axis = 1)

    b = 1/d1 

    bi = np.zeros((nz+2, nx+1))
    bj = np.zeros((nz+1, nx+2))

    for i in range(nz+1):
        bj[i, :] = 1/2 * (b[i+1, :] + b[i, :]) 

    for i in range(nx+1):
        bi[:, i] = 1/2 * (b[:, i+1] + b[:, i]) 

    bj = bj[:, 1:-1]
    bi = bi[1:-1, :]

    
    for k in range(len(F)):
        w = (2 * np.pi * F[k]) 

        z1 = np.zeros(nz+1-n_pml)
        z1 = np.append(z1, np.arange(n_pml-1, -2, -1)) * dz
        x1 = np.arange(-1, n_pml)
        x1 = np.append(x1, np.zeros(nx-2*n_pml))
        x1 = np.append(x1, np.arange(n_pml-1, -2, -1)) * dx

        c_pml_x = np.zeros(nx+2)
        c_pml_z = np.zeros(nz+2)

        c_pml1 = 20

        c_pml_x[:n_pml+1] = c_pml1
        c_pml_x[-n_pml-1:] = c_pml1
        c_pml_z[-n_pml-1:] = c_pml1


        Lx = n_pml * dx
        Lz = n_pml * dz

        gx = 1 + (1j * c_pml_x * np.cos(np.pi/2 * x1/Lx) / w)
        gz = 1 + (1j * c_pml_z * np.cos(np.pi/2 * z1/Lz) / w)
        gz.shape = (nz+2, 1)


        gxi = 1/2 * (gx[1:] + gx[:-1])
        gzj = 1/2 * (gz[1:] + gz[:-1])
    
        C1 = (w**2/K) - (1/(gx[1:-1] * dx**2) * (bi[:, 1:]/gxi[1:] + bi[:, :-1]/gxi[:-1])) - (1/(gz[1:-1] * dz**2) * (bj[1:, :]/gzj[1:] + bj[:-1, :]/gzj[:-1]))
        C2 = 1/(gx[1:-1] * dx**2) * (bi[:, :-1]/gxi[:-1])
        C3 = 1/(gx[1:-1] * dx**2) * (bi[:, 1:]/gxi[1:])
        C4 = 1/(gz[1:-1] * dz**2) * (bj[:-1, :]/gzj[:-1])
        C5 = 1/(gz[1:-1] * dz**2) * (bj[1:, :]/gzj[1:])

        C1 = C1.flatten()
        C2 = C2.flatten()
        C3 = C3.flatten()
        C4 = C4.flatten()
        C5 = C5.flatten()

        M = diags([C4[nx:], C2[1:], C1, C3, C5], [-nx, -1, 0, 1, nx], shape = (nx*nz, nx*nz), format = 'csc')
	
        s = Ts * src[k]
        s = s.flatten()

        p1 = spsolve(M, s)
        p2 = p1.copy()
        p2.shape = (nx*nz, 1)

        n_rec, temp = Tr.shape

        data1 = np.matmul(Tr, p1)
        data1.shape = (1, data1.size, 1)
        data = np.append(data, data1, axis = 2)

        p = p1.reshape(nz, nx)
        P = np.append(P, np.atleast_3d(p), axis = 2)

    P = np.delete(P, 0, 2)
    data = np.delete(data, 0, 2)

    data.shape = (1, n_rec, len(F))
    return P, data




"""
  ================================================================================== SOURCE AND RECEIVER LOCATION =================================================================================================
"""


def src_rec(x_src, z_src, x_rec, z_rec, dx, dz, nx, nz, n_pml):
    
    Ts = np.zeros((nz, nx))

    for i in range(len(x_src)):
        si = x_src[i]//dx + n_pml
        sj = z_src//dz

        if x_src[i]%dx == 0 and z_src%dz == 0:
            Ts[sj, si] = 1
            Ts[sj+1, si] = 0
            Ts[sj, si+1] = 0
            Ts[sj+1, si+1] = 0

        if x_src[i]%dx == 0 and z_src%dz != 0:
            Ts[sj, si] = (dz-z_src%dz)/dz
            Ts[sj+1, si] = (z_src%dz)/dz
            Ts[sj, si+1] = 0
            Ts[sj+1, si+1] = 0

        if x_src[i]%dx != 0 and z_src%dz == 0:
            Ts[sj, si] = (dx-x_src[i]%dx)/dx
            Ts[sj+1, si] = 0
            Ts[sj, si+1] = (x_src[i]%dx)/dx
            Ts[sj+1, si+1] = 0

        if x_src[i]%dx != 0 and z_src%dz != 0:
            Ts[sj, si] =  ((dx - x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
            Ts[sj, si+1] = ((x_src[i]%dx)/dx + (dz - z_src%dz)/dz)/2
            Ts[sj+1, si] =  ((dx - x_src[i]%dx)/dx + (z_src%dz)/dz)/2
            Ts[sj+1, si+1] =  ((x_src[i]%dx)/dx + (z_src%dz)/dz)/2
                
    n_rec = len(x_rec)
    Tr = np.zeros((n_rec, nz*nx))
    
    for i in range(len(x_rec)):
        ri = int(x_rec[i]//dx) + n_pml
        rj = z_src//dz
        
        Tr1 = np.zeros((nz, nx))

        if x_rec[i]%dx == 0 and z_rec%dz == 0:
            Tr1[rj, ri] = 1
            Tr1[rj+1, ri] = 0
            Tr1[rj, ri+1] = 0
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx == 0 and z_rec%dz != 0:
            Tr1[rj, ri] = (dz-z_rec%dz)/dz
            Tr1[rj+1, ri] = (z_rec%dz)/dz
            Tr1[rj, ri+1] = 0
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx != 0 and z_rec%dz == 0:
            Tr1[rj, ri] = (dx-x_rec[i]%dx)/dx
            Tr1[rj+1, ri] = 0
            Tr1[rj, ri+1] = (x_rec[i]%dx)/dx
            Tr1[rj+1, ri+1] = 0

        if x_rec[i]%dx != 0 and z_rec%dz != 0:
            Tr1[rj, ri] =  ((dx - x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
            Tr1[rj, ri+1] = ((x_rec[i]%dx)/dx + (dz - z_rec%dz)/dz)/2
            Tr1[rj+1, ri] =  ((dx - x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
            Tr1[rj+1, ri+1] =  ((x_rec[i]%dx)/dx + (z_rec%dz)/dz)/2
            
        Tr2 = Tr1.flatten()
        Tr2.shape = (1, Tr2.size)
        
        Tr[i, :] = Tr2
                
    return Ts, Tr



"""
  ================================================================================== FOURIER TRANSFORM ===============================================================================================================
"""

def freq2time(data, F):

     temp1, n_rec, nf = data.shape

     t_max = 5
     dt = .001
     t = np.arange(dt, t_max+dt, dt)
     nt = len(t)
     df = F[1] - F[0]

     seis = np.empty((1, n_rec))
     W = 2 * np.pi * F
     W.shape = (1, 1, W.size)

     for i in range(nt):
        seis1 = data * np.exp(-1j * W * t[i]) * 1/(np.sqrt(2 * np.pi)) * df
        seis2 = np.sum(seis1, axis = 2)
        seis = np.append(seis, seis2, axis = 0)

     seis = np.delete(seis, 0, 0)


     return t, seis



"""
  ================================================================================== PLOTTING FINITE FREQUENCY WAVEFIELDS ========================================================================================================
"""

def plot_wavefields(P, x_src, z_src, n_pml, dx, dz, F, fname):

    nz, nx, nf = P.shape
    nrows, ncols = 4, 5
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(20, 12))
    plt.suptitle('Source x: '+str(x_src)+'m'+'\n Source z: '+str(z_src)+'m', fontweight="bold")

    num = int(nf/(nrows*ncols-1))

    F1 = np.round(F, 1)
    i = 0

    for ax in axes.flat:
        im_data = P[:-n_pml, n_pml:-n_pml, int(num*i)].real
        pmax = np.sqrt(np.mean(im_data**2))
        im = ax.imshow(im_data.real, cmap='seismic', vmin = -pmax, vmax = pmax, extent = [0, (nx-2*n_pml) * dx, (nz-n_pml)*dz, 0]) 
        ax.set_title(f'{F1[num*i]} Hz')
        if i%5 != 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel('Z  (m)')
        if i <= 14: 
            ax.set_xticks([])
        else:
            ax.set_xlabel('X  (m)')
        i += 1


    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(fname, dpi = 100)


"""
  ================================================================================== PLOTTING SEISMOGRAM =================================================================================================
"""

def plot_seismogram(seis, t, nx, n_pml, dx, z_rec, fname):

    vmax = np.sqrt(np.mean(seis.real**2))

    fig, ax = plt.subplots(figsize = (12, 6))
    fig1 = ax.imshow(seis.real, cmap = 'gray', aspect = 'auto', extent = [0, (nx-2*n_pml)*dx, t[-1], t[0]], vmin = -vmax, vmax = vmax)
    fig.colorbar(fig1)
    ax.set_title('seismogram recorded at: '+str(z_rec)+'m')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Time (s)')

    plt.savefig(fname, dpi = 100)
