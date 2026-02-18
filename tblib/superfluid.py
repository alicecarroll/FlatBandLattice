import numpy as np
from . import hamiltonian_opti
from importlib import reload
reload(hamiltonian_opti)
from numba import njit, jit, prange, complex128


@njit#(fastmath=True)
def matmul(A, B):
    return np.dot(A, B)

@njit#(fastmath=True)
def fermidirac(E,T,o=0):
    
    nE=0.0+0.0j
    if o==0:
        if T>=1e-2 or (np.real(E)<T*50 and T!=0.0):
            nE = 1/(1+np.exp(E/T))
        elif T<1e-2:
            if np.real(E)>0:
                nE = 0.0+0.0j
            elif np.abs(E)<1e-10:
                nE = 1/2
            else:
                nE = 1
        
    elif o==1:
        if T>=1e-2 or (np.real(E)<T*50 and T!=0.0):
            nE = -(1/(1+np.exp(E/T))**2)*np.exp(E/T)/T
        elif T<1e-2:
            nE = 0     
    return nE

@njit
def eigen_Hred(s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta, karr, dmy= (1,0), dny= (1,0)):
    '''
    Calculate all eigenvalues and eigenvectors of Hkin, Hreduced and their derivatives
    return as matrices respectively 
    '''
    H = np.zeros((n, n), dtype=complex128)
    l = len(karr)
    eval_arr = np.zeros((l**2, 2*n), dtype=complex128)
    evec_arr = np.zeros((l**2, 2*n, 2*n), dtype=complex128)
    
    count=0
    for i in range(l): 
        ky = karr[i]
        for j in range(l):
            kx = karr[j]

            Hred = hamiltonian_opti.HBdG(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t, nu, T, U, ns, mu, delta)[1]
            evals, evec = np.linalg.eigh(Hred)

            eval_arr[count] = evals.copy()
            evec_arr[count] = evec.copy()
            count+=1
            

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

            #Hred = hamiltonian_opti.HBdG(H.copy(), kx, ky, 0, 0, s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t,
            #                             nu, T, U, ns, mu, delta)[1]
            Hdmy[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dmy[0], dmy[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)
            Hdny[:] = hamiltonian_opti.H_kin(H.copy(), kx, ky, dny[0], dny[1], s_idx, n_idx, sx, sy, nx, ny, R_ptr, R_flat, n, N, t)

            #evals, evec = np.linalg.eigh(Hred)
            #Evec = evec.T 
            evals[:] = eval_arr[counter]
            Evec[:] = evec_arr[counter]

            M1[:] = matmul(Hdmy,gammaz)
            M2[:] = matmul(Hdny,gammaz)         

            if counter<3:
                print(H, '\n', Hdmy, '\n', Hdny, '\n')   

            for ei,E in enumerate(evals):

                nE[ei] = fermidirac(E,T,o=0)
                dnE[ei] = fermidirac(E,T,o=1)

            pflist[:] = 0.0+0.0j
            diali[:] = 0.0+0.0j
            parli[:] = 0.0+0.0j
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

                        f1 = np.vdot(vk, Hdmy @ vl)
                        f2 = np.vdot(vl, Hdny @ vk)
                        f3 = np.vdot(vk, M1 @ vl)
                        f4 = np.vdot(vl, M2 @ vk)
                        #f1 = matmul(np.conjugate(Evec[k]),matmul(Hdmy,Evec[l]))
                        #f2 = matmul(np.conjugate(Evec[l]),matmul(Hdny,Evec[k]))

                        #f3 = matmul(np.conjugate(Evec[k]),matmul(M1,Evec[l]))
                        #f4 = matmul(np.conjugate(Evec[l]),matmul(M2,Evec[k]))

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

def detSFW(model, nk=41):
    xx = SFW(model, nk, my=(1,0), ny=(1,0))[0]
    xy = SFW(model, nk, my=(1,0), ny=(0,1))[0]
    #yx = SFW(model, nk, my=(0,1), ny=(1,0))[0]
    #yy = SFW(model, nk, my=(0,1), ny=(0,1))[0]
    ten = np.array([[xx,xy],[xy,xx]])

    return ten, np.sqrt(np.linalg.det(ten))

def SFWconv(model, nk=41, my= (1,0), ny=(1,0)):
    
    T=model.T
    a=model.n

    karr = np.linspace(0,2*np.pi, nk, endpoint=False)
    summe = 0

    term_array = np.zeros((nk**2, 3,int((a)**2)), dtype=complex)
    counter =0

    HBdG = model.get_reducedH()
    kinH = model.get_kinH()
    kinHdmy = model.get_kinH(dnx=my[0], dny=my[1])
    kinHdny = model.get_kinH(dnx=ny[0], dny=ny[1])
    
    for kx in karr:
        for ky in karr:
            pref = []
            upcurr = []
            downcurr = []

            H = HBdG(kx,ky)
            dH_my = kinHdmy(kx,ky)
            dH_ny = kinHdny(kx,ky)

            H_up = kinH(kx,ky)[:a,:a]
            H_down = -kinH(kx,ky)[a:,a:]
            
            evals, evec = np.linalg.eigh(H)
            evals_up, evec_up = np.linalg.eigh(H_up)
            evals_down, evec_down = np.linalg.eigh(H_down)
            
            Evec = evec.T 
            Evec_up = evec_up.T 
            Evec_down = evec_down.T 

            evalsdmy = [np.matmul(e, np.matmul(dH_my[:a,:a], e))for e in Evec_up]
            evalsdny = [np.matmul(e, np.matmul(dH_ny[a:,a:], e))for e in Evec_down]

            m_mat=np.block([[Evec_up, np.zeros((a,a))], 
                            [np.zeros((a,a)), Evec_down]])
            s_array = np.zeros((2*a,2*a), dtype=complex)
            for i in range(2*a):
                s_array[i]= np.linalg.solve(m_mat.T, Evec[i])

            nE = [fermidirac(E,T,o=0) for E in evals]
            dnE = [fermidirac(E,T,o=1) for E in evals]

            for m in range(a):
                for n in range(a):
                    Cnn=0
                    for k,i in enumerate(evals):
                        for l,j in enumerate(evals):
                            
                            if np.abs(i-j)<1e-6 or k==l:
                                pf = -dnE[l]
                            else:
                                pf = (nE[l]-nE[k])/(i-j)

                            if pf==0:
                                Cnn+=0

                            else:
                                
                                s_l = s_array[l]
                                s_k = s_array[k]
                                w1 = np.conjugate(s_l[m])
                                w2 = s_k[m]
                                w3 = np.conjugate(s_k[n+a])
                                w4 = s_l[n+a]

                                Cnn+=4*pf*w1*w2*w3*w4
                    
                    upc = evalsdmy[m]
                    downc = evalsdny[n]
                    summe+=Cnn/(nk**2)*upc*downc

                            
                    pref.append(Cnn/(nk**2))
                    upcurr.append(upc)
                    downcurr.append(downc)

            term_array[counter]= np.array([pref, upcurr, downcurr])
            counter+=1

    return summe, term_array

def det_convSFW(model, nk=41):
    xx = SFWconv(model, nk, my=(1,0), ny=(1,0))[0]
    xy = SFWconv(model, nk, my=(1,0), ny=(0,1))[0]
    #yx = SFWconv(model, nk, my=(0,1), ny=(1,0))[0]
    #yy = SFWconv(model, nk, my=(0,1), ny=(0,1))[0]
    ten = np.array([[xx,xy],[xy,xx]])

    return ten, np.sqrt(np.linalg.det(ten))