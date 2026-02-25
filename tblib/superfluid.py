import numpy as np
from . import hamiltonian_opti
from importlib import reload
reload(hamiltonian_opti)
from numba import njit, jit, prange, complex128


@njit(fastmath=True)
def matmul(A, B):
    return np.dot(A, B)

@njit(parallel=True, fastmath=True)
def transpose(A):
    '''A must be a 2D array'''
    B = np.zeros_like(A)
    l1,l2 = np.shape(A)
    for i in prange(l1):
        for j in range(l2):
            B[i,j] = A[j,i]
    return B

@njit(fastmath=True)
def fermidirac(E,T,o=0):
    
    nE=0.0+0.0j
    if o==0:
        if T>=1e-2 or (np.real(E)<T*50 and T!=0.0):
            nE = 1/(1+np.exp(E/T))
        elif T<1e-2:
            if np.real(E)>0:
                nE = 0.0+0.0j
            elif np.abs(E)<1e-10:
                nE = 1/2+0.0j
            else:
                nE = 1+0.0j
        
    elif o==1:
        if T>=1e-2 or (np.real(E)<T*50 and T!=0.0):
            nE = -(1/(1+np.exp(E/T))**2)*np.exp(E/T)/T
        elif T<1e-2:
            nE = 0.0 + 0.0j     
    return nE

@njit(parallel=True)
def eigen_Hred(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr, dmy= (0,0)):
    '''
    Calculate all eigenvalues and eigenvectors of Hkin, Hreduced and their derivatives
    return as matrices respectively 
    '''
    H = np.zeros((n, n), dtype=complex128)
    l = len(karr)
    eval_arr = np.zeros((l**2, 2*n), dtype=complex128)
    evec_arr = np.zeros((l**2, 2*n, 2*n), dtype=complex128)
    
    for i in prange(l): 
        ky = karr[i]
        for j in range(l):
            kx = karr[j]

            Hred = hamiltonian_opti.HBdG(H.copy(), kx, ky, dmy[0], dmy[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta)[1]
            evals, evec = np.linalg.eigh(Hred)

            eval_arr[i*l+j] = evals.copy()
            evec_arr[i*l+j] = evec.copy()
        
    return eval_arr, evec_arr


@njit
def SFW(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy= (1,0), dny=(1,0)):
    
    nk = len(karr)
    H = np.zeros((n, n), dtype=complex128)
    Hdmy = np.zeros((2*n,2*n), dtype=complex128)
    Hdny = np.zeros((2*n,2*n), dtype=complex128)

    gammaz = np.diag(np.ones(2*n, dtype=complex128))
    gammaz[n:, n:] = -gammaz[n:, n:] 
    term_array = np.zeros((nk**2, 3,int((n*2)**2)), dtype=complex128)

    pflist = np.zeros((2*n)**2, dtype=complex128)
    parli = np.zeros((2*n)**2, dtype=complex128)
    diali = np.zeros((2*n)**2, dtype=complex128)

    evals = np.zeros(2*n, dtype=complex128)
    Evec = np.zeros((2*n, 2*n), dtype=complex128)
    M1 = np.zeros((2*n, 2*n),dtype=complex128)
    M2 = np.zeros((2*n, 2*n),dtype=complex128)
    nE = np.zeros(2*n, dtype=complex128)
    dnE = np.zeros(2*n, dtype=complex128)

    eval_arr, evec_arr = eigen_Hred(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy, dny)

    summe = 0.0+0.0j
    counter =0

    for yi in range(nk): 
        ky = karr[yi]
        for xj in range(nk):
            kx = karr[xj]

            Hdmy[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dmy[0], dmy[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)
            Hdny[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dny[0], dny[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)

            evals[:] = eval_arr[counter]
            Evec[:] = evec_arr[counter]

            M1[:] = matmul(Hdmy,gammaz)
            M2[:] = matmul(Hdny,gammaz)         

            for ei,E in enumerate(evals):

                nE[ei] = fermidirac(E,T,o=0)
                dnE[ei] = fermidirac(E,T,o=1)

            scounter = 0
            for k in range(2*n):
                i = evals[k]
                for l in range(2*n):
                    j = evals[l]
                    if np.abs(i-j)<1e-6 or k==l:
                        pf = -dnE[l]
                    else:
                        pf = (nE[l]-nE[k])/(i-j)

                    if np.real(pf)==0.0:
                        summe+=0.0+0.0j
                        f1, f2, f3, f4 = (0.0+0.0j,0.0+0.0j,0.0+0.0j,0.0+0.0j)

                    else:
                        vk = Evec[:,k]
                        vl = Evec[:,l]
                        
                        T1 = matmul(Hdmy,vl)
                        T2 = matmul(Hdny,vk)
                        T3 = matmul(M1,vl)
                        T4 = matmul(M2,vk)

                        f1 = matmul(np.conjugate(vk),T1)
                        f2 = matmul(np.conjugate(vl),T2)
                        f3 = matmul(np.conjugate(vk),T3)
                        f4 = matmul(np.conjugate(vl),T4)

                        s = pf*(f1*f2-f3*f4)

                        summe+=s

                    pflist[scounter] = pf/nk**2
                    diali[scounter] = f1*f2
                    parli[scounter] = f3*f4
                    scounter+=1
                
            term_array[counter,0]= pflist
            term_array[counter,1]= diali
            term_array[counter,2]= parli

            counter +=1
    
    return summe/nk**2, term_array

@njit
def detSFW(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr):
    xx = SFW(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy= (1,0), dny=(1,0))[0]
    xy = SFW(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy= (1,0), dny=(0,1))[0]
   
    ten = np.array([[xx,xy],[xy,xx]], dtype=complex128)

    return ten, np.sqrt(xx*xx-xy*xy)

@njit
def SFWconv(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy= (1,0), dny=(1,0)):
    
    nk = len(karr)
    
    H = np.zeros((n, n), dtype=complex128)
    evals = np.zeros(2*n, dtype=complex128)
    Evec = np.zeros((2*n, 2*n), dtype=complex128)
    evec_up = np.zeros((n, n), dtype=complex128)
    evec_down = np.zeros((n, n), dtype=complex128)
    evalsdmy = np.zeros(n, dtype=complex128)
    evalsdny = np.zeros(n, dtype=complex128)
    m_mat = np.zeros((2*n, 2*n), dtype=complex128)
    nE = np.zeros(2*n, dtype=complex128)
    dnE = np.zeros(2*n, dtype=complex128)

    eval_arr, evec_arr = eigen_Hred(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy, dny)

    term_array = np.zeros((nk**2, 3,int((n)**2)), dtype=complex128)
    pref = np.zeros(int(n**2), dtype=complex128)
    upcurr = np.zeros(int(n**2), dtype=complex128)
    downcurr = np.zeros(int(n**2), dtype=complex128)

    summe = 0.0 + 0.0j
    counter =0
    
    for yi in range(nk): 
        ky = karr[yi]
        for xj in range(nk):
            kx = karr[xj]

            dH_my = hamiltonian_opti.H_kin(H.copy(), kx, ky, dmy[0], dmy[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)
            dH_ny = hamiltonian_opti.H_kin(H.copy(), kx, ky, dny[0], dny[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)

            H_up = hamiltonian_opti.H_kin(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)[:n,:n]
            H_down = -hamiltonian_opti.H_kin(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)[n:,n:]
            
            evals[:] = eval_arr[counter]
            Evec[:] = evec_arr[counter]
            evec_up[:] = np.linalg.eigh(H_up)[1]
            evec_down[:] = np.linalg.eigh(H_down)[1]
            
            evalsdmy[:] = 0.0+0.0j
            evalsdny[:] = 0.0+0.0j

            for ei in range(n):
                evalsdmy[ei] = matmul(evec_up[:,ei], matmul(dH_my[:n,:n], evec_up[:,ei]))
                evalsdny[ei] = matmul(evec_down[:,ei], matmul(dH_ny[n:,n:], evec_down[:,ei]))

            m_mat[:] = 0.0+0.0j
            m_mat[:n,:n]=evec_up.T
            m_mat[n:,n:]=evec_down.T     

            s_array = np.zeros((2*n,2*n), dtype=complex128)
            for i in range(2*n):
                s_array[i]= np.linalg.solve(m_mat.T, Evec[:,i])

            for ei,E in enumerate(evals):
                nE[ei] = fermidirac(E,T,o=0)
                dnE[ei] = fermidirac(E,T,o=1)

            scount = 0
            for mi in range(n):
                for ni in range(n):
                    Cnn=0.0+0.0j
                    for k,i in enumerate(evals):
                        for l,j in enumerate(evals):
                            
                            if np.abs(i-j)<1e-6 or k==l:
                                pf = -dnE[l]
                            else:
                                pf = (nE[l]-nE[k])/(i-j)

                            if pf==0.0+0.0j:
                                Cnn+=0.0+0.0j

                            else:
                                
                                s_l = s_array[l]
                                s_k = s_array[k]
                                w1 = np.conjugate(s_l[mi])
                                w2 = s_k[mi]
                                w3 = np.conjugate(s_k[ni+n])
                                w4 = s_l[ni+n]

                                Cnn+=4*pf*w1*w2*w3*w4
                    
                    upc = evalsdmy[mi]
                    downc = evalsdny[ni]
                    summe+=Cnn/(nk**2)*upc*downc

                    pref[scount] = Cnn/(nk**2)
                    upcurr[scount] = upc
                    downcurr[scount] = downc     
                    scount+=1

            term_array[counter,0]= pref[:]
            term_array[counter,1]= upcurr[:]
            term_array[counter,2]= downcurr[:]
            counter+=1

    return summe, term_array

@njit
def SFW_complete(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, dmy= (1,0), dny=(1,0)):
    
    nk = len(karr)
    H = np.zeros((n, n), dtype=complex128)
    Hdmy = np.zeros((2*n,2*n), dtype=complex128)
    Hdny = np.zeros((2*n,2*n), dtype=complex128)

    gammaz = np.diag(np.ones(2*n, dtype=complex128))
    gammaz[n:, n:] = -gammaz[n:, n:] 
    
    evals = np.zeros(2*n, dtype=complex128)
    Evec = np.zeros((2*n, 2*n), dtype=complex128)
    evec_up = np.zeros((n, n), dtype=complex128)
    evec_down = np.zeros((n, n), dtype=complex128)
    evalsdmy = np.zeros(n, dtype=complex128)
    evalsdny = np.zeros(n, dtype=complex128)
    m_mat = np.zeros((2*n, 2*n), dtype=complex128)

    M1 = np.zeros((2*n, 2*n),dtype=complex128)
    M2 = np.zeros((2*n, 2*n),dtype=complex128)
    nE = np.zeros(2*n, dtype=complex128)
    dnE = np.zeros(2*n, dtype=complex128)

    eval_arr, evec_arr = eigen_Hred(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, 
                          karr, (0,0), (0,0))

    sfw_tot = 0.0+0.0j
    sfw_conv = 0.0+0.0j
    counter =0
    
    for yi in prange(nk): 
        ky = karr[yi]
        for xj in range(nk):
            kx = karr[xj]

            Hdmy[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dmy[0], dmy[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)
            Hdny[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dny[0], dny[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)

            H_up = hamiltonian_opti.H_kin(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)[:n,:n]
            H_down = -hamiltonian_opti.H_kin(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)[n:,n:]

            evals[:] = eval_arr[counter]
            Evec[:] = evec_arr[counter]
            evec_up[:] = np.linalg.eigh(H_up)[1]
            evec_down[:] = np.linalg.eigh(H_down)[1]

            M1[:] = matmul(Hdmy,gammaz)
            M2[:] = matmul(Hdny,gammaz)      

            evalsdmy[:] = 0.0+0.0j
            evalsdny[:] = 0.0+0.0j  
            for ei in range(n):
                evalsdmy[ei] = matmul(evec_up[:,ei], matmul(Hdmy[:n,:n],evec_up[:,ei]))
                evalsdny[ei] = matmul(evec_down[:,ei], matmul(Hdny[n:,n:],evec_down[:,ei]))

            m_mat[:] = 0.0+0.0j
            m_mat[:n,:n]=transpose(evec_up)
            m_mat[n:,n:]=transpose(evec_down)

            s_array = np.zeros((2*n,2*n), dtype=complex128)
            for i in range(2*n):
                s_array[i]= np.linalg.solve(transpose(m_mat), Evec[:,i])

            for ei,E in enumerate(evals):

                nE[ei] = fermidirac(E,T,o=0)
                dnE[ei] = fermidirac(E,T,o=1)

            scounter=0
            for mi in range(n):
                for ni in range(n):
                    Cnn=0.0+0.0j

                    for k in range(2*n):
                        i = evals[k]
                        for l in range(2*n):
                            j = evals[l]
                            if np.abs(i-j)<1e-6 or k==l:
                                pf = -dnE[l]
                            else:
                                pf = (nE[l]-nE[k])/(i-j)

                            if np.abs(pf)<1e-12:
                                sfw_tot+=0.0+0.0j
                                f1, f2, f3, f4 = (0.0+0.0j,0.0+0.0j,0.0+0.0j,0.0+0.0j)

                                Cnn+=0.0+0.0j

                            else:
                                if scounter==0:
                                    vk = Evec[:,k]
                                    vl = Evec[:,l]
                                    
                                    T1 = matmul(Hdmy,vl)
                                    T2 = matmul(Hdny,vk)
                                    T3 = matmul(M1,vl)
                                    T4 = matmul(M2,vk)

                                    f1 = matmul(np.conjugate(vk),T1)
                                    f2 = matmul(np.conjugate(vl),T2)
                                    f3 = matmul(np.conjugate(vk),T3)
                                    f4 = matmul(np.conjugate(vl),T4)

                                    s = pf*(f1*f2-f3*f4)
                                    sfw_tot+=s

                                s_l = s_array[l]
                                s_k = s_array[k]
                                w1 = np.conjugate(s_l[mi])
                                w2 = s_k[mi]
                                w3 = np.conjugate(s_k[ni+n])
                                w4 = s_l[ni+n]

                                Cnn+=4*pf*w1*w2*w3*w4
                    scounter+=1

                    upc = evalsdmy[mi]
                    downc = evalsdny[ni]
                    sfw_conv+=Cnn/(nk**2)*upc*downc
            counter +=1
    
    sfw_tot = sfw_tot/nk**2
    sfw_geom = sfw_tot - sfw_conv
    return sfw_tot, sfw_conv, sfw_geom

@njit
def det_SFWs(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr):
    xx_tot, xx_conv, xx_geom = SFW_complete(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr, dmy= (1,0), dny=(1,0))
    xy_tot, xy_conv, xy_geom = SFW_complete(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr, dmy= (1,0), dny=(0,1))
    
    ten_tot = np.array([[xx_tot,xy_tot],[xy_tot,xx_tot]])
    ten_conv = np.array([[xx_conv,xy_conv],[xy_conv,xx_conv]])
    ten_geom = np.array([[xx_geom,xy_geom],[xy_geom,xx_geom]])

    ds_tot = np.sqrt(xx_tot*xx_tot-xy_tot*xy_tot)
    ds_conv = np.sqrt(xx_conv*xx_conv-xy_conv*xy_conv)
    ds_geom = np.sqrt(xx_geom*xx_geom-xy_geom*xy_geom)

    return ds_tot, ds_conv, ds_geom, ten_tot, ten_conv, ten_geom
    
