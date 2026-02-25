import numpy as np
from numba import njit, prange, complex128

#def hopping_prep(site, nn, R, k, t):
#    s = np.asarray()
@njit#(fastmath=True)
def hopping_kernel(kx, ky, dnx, dny, N, sx, sy, nnx, nny, R, r0, r1, t):

    kx/=N
    ky/=N

    drkx = 1j * (nnx - sx)
    drky = 1j * (nny - sy)
    f0 =  np.exp(drkx * kx + drky * ky)
    
    farr = 0.0+0.0j
    dfarr = 0.0+0.0j
    for r in range(r0, r1):
        dRkx = 1j  * N * R[r,0]
        dRky = 1j  * N * R[r,1] 
        farr += np.exp(dRkx * kx + dRky * ky)
        dfarr += np.exp(dRkx * kx + dRky * ky)*(dRkx**dnx) * (dRky**dny)

    if dnx == 0 and dny == 0:
        res = -t * f0 * farr 
    else:
        res = - 1/N*t * (drkx**dnx) * (drky**dny) * f0 * farr # f'(sublattice)*g(uc)
        res += - 1/N*t*dfarr*f0   # f(sublattice)*g'(uc)
    return res

@njit(parallel=True)#, fastmath=True)
def H_kin(H, kx, ky, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t):
    """
    Evaluate the hopping Hamiltonian at given kx, ky.

    This function essentially only serves to calculate the SFW.
    The hole sector doesn't have the minus sign for this reason!
    """

    H0k = np.zeros((n,n), dtype=complex128)
    H0kh = np.zeros((n,n), dtype=complex128)
    H_kin = np.zeros((2*n, 2*n), dtype=complex128)
    
    for e in prange(len(s_idx)):
        j=s_idx[e]
        i=n_idx[e]
        
        H0k[i,j] += hopping_kernel(kx, ky, dnx, dny, N, sx[e], sy[e], nx[e], ny[e], R_flat, R_ptr[e], R_ptr[e+1], t)
        H0kh[i,j] += -hopping_kernel(-kx, -ky, dnx, dny, N, sx[e], sy[e], nx[e], ny[e], R_flat, R_ptr[e], R_ptr[e+1], t)
    
    H_kin[:n, :n] = H0k
    H_kin[n:, n:] = -np.conjugate(H0kh)

    return H_kin

@njit(parallel=True)#, fastmath=True)
def H_0(H, kx, ky, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu):
    """
    Evaluate the hopping Hamiltonian at given kx, ky.
    This function essentially only serves to calculate the SFW.
    The hole sector doesn't have the minus sign for this reason!
    """
    
    for e in range(len(s_idx)):
        j=s_idx[e]
        i=n_idx[e]
        H[i,j] += hopping_kernel(kx, ky, dnx, dny, N, sx[e], sy[e], nx[e], ny[e], R_flat, R_ptr[e], R_ptr[e+1], t)
    for j in prange(n):
        H[j,j] += -mu[j] - 0.5*U[j]* ns[j]    
            
    return H

@njit(parallel=True)#, fastmath=True)
def HBdG(H, kx, ky, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta): 
    """Evaluate the Hamiltonian at given kx, ky."""

    H0kp = np.zeros_like(H, dtype=complex128)
    H0kh = np.zeros_like(H, dtype=complex128)
    HDk = np.zeros_like(H, dtype=complex128)

    H_BdG = np.zeros((4*n, 4*n), dtype=complex128) 
    H_reduced = np.zeros((2*n, 2*n), dtype=complex128)

    H0kp = H_0(H0kp, kx, ky, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu)                   # particle sector
    H0kh = -np.conjugate(H_0(H0kh, -kx, -ky, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu))  # hole sector

    for j in range(n):
        HDk[j,j] += np.abs(U[j])*delta[j]
    
    for i in prange(n):
        for j in range(n):
            H_BdG[i,j] = H0kp[i,j]
            H_reduced[i,j] = H0kp[i,j]
            H_BdG[n+i, n+j] = H0kp[i,j]
            H_BdG[2*n+i, 2*n+j] = H0kh[i,j]
            H_BdG[3*n+i, 3*n+j] = H0kh[i,j]
            H_reduced[n+i, n+j] = H0kh[i,j]
            H_BdG[i,3*n+j] = HDk[i,j]
            H_reduced[i,n+j] = HDk[i,j]
            H_BdG[n+i,2*n+j] = -HDk[i,j]
            H_BdG[2*n+i,n+j] = -HDk[i,j]
            H_BdG[3*n+i,j] = HDk[i,j]
            H_reduced[n+i,j] = HDk[i,j]
    
    return H_BdG, H_reduced