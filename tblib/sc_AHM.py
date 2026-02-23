import numpy as np
from . import hamiltonian_opti
from importlib import reload
reload(hamiltonian_opti)
from numba import njit, jit, prange, complex128


@njit(fastmath=True)
def matmul(A, B):
    return np.dot(A, B)

@njit#(parallel=True)
def cooper(u,v,ubar,vbar,evals,T=0.0):
    if np.abs(T)<1e-10:#
        return  matmul(ubar.T,np.conjugate(vbar))
    else:
        A = np.zeros_like(u, dtype=complex128)
        B = np.zeros_like(u, dtype=complex128)
        
        for j in prange(B.shape[1]):
            if np.abs(evals[j])<1e-5:
                A[j,j] = 0.5+0.0j
                B[j,j] = 0.5+0.0j
            elif np.real(evals[j])<0.0 and np.abs(evals[j])>T*30:
                A[j,j] = 0.0+0.0j
                B[j,j] = 1.0+0.0j
            elif np.real(evals[j])>0.0 and np.abs(evals[j])>T*30:
                A[j,j] = 1.0+0.0j
                B[j,j] = 0.0+0.0j
            else:
                A[j,j] = 1/(1+np.exp(-evals[j]/T))
                B[j,j] = 1/(1+np.exp(evals[j]/T))

        el = matmul(ubar.T, matmul(A, np.conjugate(vbar)))
        el += matmul(v.T, matmul(B,np.conjugate(u)))
        return el

@njit#(parallel=True) 
def hatree(u,v,ubar,vbar,evals,T=0.0):
    if np.abs(T)<1e-10:
        return matmul(vbar.T, np.conjugate(vbar))
    else:
        A = np.zeros_like(u, dtype=complex128)
        B = np.zeros_like(u, dtype=complex128)
        
        for j in prange(B.shape[1]):
            if np.abs(evals[j])<1e-5:
                A[j,j] = 0.5+0.0j
                B[j,j] = 0.5+0.0j
            elif np.real(evals[j])<0.0 and np.abs(evals[j])>T*30:
                A[j,j] = 0.0+0.0j
                B[j,j] = 1.0+0.0j
            elif np.real(evals[j])>0.0 and np.abs(evals[j])>T*30:
                A[j,j] = 1.0+0.0j
                B[j,j] = 0.0+0.0j
            else:
                A[j,j] = 1/(1+np.exp(-evals[j]/T))
                B[j,j] = 1/(1+np.exp(evals[j]/T))

        el = matmul(vbar.T,matmul(A, np.conjugate(vbar)))
        el += matmul(u.T,matmul(B, np.conjugate(u)))
        return el 

@njit
def get_mean_fields(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr, HF=True):
    
    nk = np.shape(karr)[0]

    dnx = 0
    dny = 0
    
    Pairing=np.zeros((n,n), dtype=complex128)
    Occupation=np.zeros((n,n), dtype=complex128)
    deltas = np.zeros((n,), dtype=complex128)
    nsarr = np.zeros((n,), dtype=complex128)
    H = np.zeros((n,n), dtype=complex128)
    evals = np.zeros((n,), dtype=complex128)

    u = np.zeros((n,n), dtype=complex128)
    v = np.zeros((n,n), dtype=complex128)
    vbar = np.zeros((n,n), dtype=complex128)
    ubar = np.zeros((n,n), dtype=complex128)

    c=0
    for ix in range(nk):
        x = karr[ix]
        for iy in range(nk):
            y = karr[iy]
            
            c+=1

            evals1, Evec = np.linalg.eigh(hamiltonian_opti.HBdG(H, x, y, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta)[1])
            Evec = Evec.T
            Evec2 = np.empty_like(Evec)
            for i in range(Evec.shape[0]):
                Evec2[i,:] = Evec[Evec.shape[0]-1-i,:]
            for i in range(n):
                evals[i] = evals1[2*n-1-i]

            for i in range(n):
                for j in range(n):
                    u[i,j] = Evec2[i, j]
                    v[i,j] = Evec2[i, n+j]
                    vbar[i,j] = Evec[i,j] #this is conjugate(v-k)
                    ubar[i,j] = Evec[i, n+j]  #this is conjugate(u-k)
            
            Pairing+=cooper(u,v,ubar,vbar,evals,T)
            if HF:
                Occupation += hatree(u,v,ubar,vbar,evals,T)         
    
    Nmat = np.diag(Occupation)/nk**2
    for i in prange(n):
        deltas[i] = -Pairing[i,i]/nk**2
        nsarr[i] = Nmat[i]*2

    return deltas,nsarr

@njit
def self_consistency_loop(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, g=1e-6, HF=True, Nmax=100, Nmin=10, alpha=0.3):
    
    dnx=0
    dny=0
    delarr = delta.copy()
    narr = ns.copy()
    muarr = mu.copy()

    save_dels = np.zeros((n, Nmax+1), dtype=complex128)
    save_ns = np.zeros((n, Nmax+1), dtype=complex128)
    save_mus = np.zeros((n, Nmax+1), dtype=complex128)

    H0 = np.zeros((n,n), dtype=complex128)

    c=0
    for i in range(3):
        save_dels[:,c+i]=delarr
        save_ns[:,c+i]=narr
        save_mus[:,c+i]=muarr

    limit1 = False
    limit2 = False
    
    while True:
        c+=1

        if c >= Nmax:
            break
        if c >= Nmin and not (limit1 or limit2):
            break

        limit1 = False
        limit2 = False

        work_delta = delta.copy()
        work_ns = ns.copy()
        work_mu = mu.copy()

        save_dels[:,c+2]=delarr
        save_ns[:,c+2]=narr
        save_mus[:,c+2]=muarr

        Vals = get_mean_fields(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, work_ns, work_mu, work_delta, karr, HF)

        delarro = delarr.copy()
        narro = narr.copy()

        delta = Vals[0]

        delarr = delta.copy()
        delarr = alpha*delarro+(1-alpha)*delarr

        delta[:] = delarr

        if HF:
            ns = Vals[1]
            
            narr = ns.copy()
            narr = alpha*narro+(1-alpha)*narr

            ns[:] = narr

            v=0.0
            for i in range(n):
                v = np.abs(save_ns[i,c+2]-save_ns[i,c+1])
                if v*v > g:
                    limit2 = True
                    break

        v=0.0
        for i in range(n):
            v = np.abs(save_dels[i,c+2]-save_dels[i,c+1])
            if v*v > g:
                limit1 = True
                break

        if nu!=0.0:
            muarro = muarr.copy()
            
            en=0.0+0j
            H0[:]=0.0+0j
            H = hamiltonian_opti.HBdG(H0, 0.0,0.0, dnx, dny, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, narro, muarro, delarro)[1]
            #print(H)
            for i in range(n):
                en += H[i,i]
                

            mun = 1/(n)*(U[0]/2*(nu-n*2)+en)

            for i in range(n):
                muarr[i] = mun          
            for i in range(n):
                muarr[i] = alpha*muarro[i] + (1.0-alpha)*muarr[i] 

            mu[:] = muarr
                    
    return save_dels[:,:c+2], save_ns[:,:c+2], save_mus[:,:c+2]
    