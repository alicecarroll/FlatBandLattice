import numpy as np
from .fermi import fermi_dirac, fermi_dirac_prime

# Partially refactored: need check expressions and test

def cooper(u, v, uk, vk, evals, T=.0):
    if np.abs(T) < 1e-10: return np.matmul(np.conjugate(v.T), v)
    res = np.matmul(np.conjugate(u.T),np.matmul(np.diag(1/(1+np.exp(-evals/T))), v))
    res += np.matmul(vk.T,np.matmul(np.diag(1/(1+np.exp(evals/T))),np.conjugate(uk)))
    return res

def hartree(v, uk, evals, T=.0):
    if np.abs(T) < 1e-10: return np.matmul(np.conjugate(v.T), v)
    res = np.matmul(np.conjugate(v.T),np.matmul(np.diag(1/(1+np.exp(-evals/T))), v))
    res += np.matmul(uk.T,np.matmul(1/(1+np.exp(evals/T)), np.conjugate(uk)))
    return res

def get_mean_fields(model, k_grid, HF_Q=True, T=.0):
    """Compute Cooper and Hartree mean-field channels once."""

    sdim = int( 2 * model.n ) # Single-particle dimension
    PairingMatrix = np.zeros((sdim,sdim), dtype=complex)
    OccupationMatrix = np.zeros((sdim,sdim), dtype=complex)

    # Get updated Hamiltonian
    HBdG = model.get_HBdG()

    nk = len(k_grid)
    for kx, ky in k_grid:
        evals, evecs = np.linalg.eigh(HBdG(kx, ky))

        upk = evecs[sdim:, :sdim]
        vpk = evecs[:sdim, :sdim]
        vmk = evecs[sdim:, sdim:]
        umk = evecs[:sdim, sdim:]

        PairingMatrix += cooper(umk, vmk, upk, vpk, evals, T=.0)
        if HF_Q: OccupationMatrix += hartree(vmk, upk, evals, T=.0)
        
    OccArr = np.diag(OccupationMatrix) / nk
    Delta = [-np.abs(model.U[i]) / nk * PairingMatrix[i,int(i+sdim/2)] for i in range(int(a/2))]
    Nu = [OccArr[i]+OccArr[int(i+sdim/2)] for i in range(int(sdim/2))]

    Delta = np.array([Delta[model.lat.map_idx[(i,0)]] for i in range(model.lat.N)])
    Nu = np.array([Nu[model.lat.map_idx[(i,0)]] for i in range(model.lat.N)])

    return Delta, Nu


# Not refactored yet   
def Deltra(model, N, T=0, g=0.01, HF=False, Nmax=20, Nmin=10, alpha=0.5):
    
    delarr = np.array(model.delta)
    nuarr = np.array(model.ns)

    dels = delarr.reshape(model.N,1)
    nus = nuarr.reshape(model.N,1)

    nk = 40    
    k_arr = np.linspace(-np.pi, np.pi, nk)
    k_grid = np.array(np.meshgrid(k_arr, k_arr)).T.reshape(-1,2)

    c=0
    while (c<Nmax and (np.std(np.abs(dels), axis=1)>g).any()) or c<Nmin:
        c+=1

        get_mean_fields(model, k_grid, HF_Q=True, T=.0)

        delarro = delarr
        nuarro = nuarr

        delta = Vals[0]

        delarr = np.array(delta)
        delarr = alpha*delarro+(1-alpha)*delarr

        self.delta =delarr

        if HF:
            nu = Vals[1]
            
            nuarr = np.array(nu)
            nuarr = alpha*nuarro+(1-alpha)*nuarr

            self.ns = nuarr

        dels = np.concatenate((dels, delarr.reshape(self.N,1)), axis=1)
        nus = np.concatenate((nus, nuarr.reshape(self.N,1)), axis=1)


    avdel = np.average(dels[:,-3:], axis=1) 
    avnu = np.average(nus[:,-3:], axis=1)  
    dels = np.concatenate((dels, avdel.reshape(self.N,1)), axis=1)
    nus = np.concatenate((nus, avnu.reshape(self.N,1)), axis=1)

    self.delta = avdel
    self.ns = avnu
    
    return dels, nus


def SFW(self, N, T=0, my='x', ny='y'):

    gammaz = np.kron(np.diag([1,-1]), np.eye(int(self.dim/2)))
    sum = 0
    karr = np.linspace(-np.pi,np.pi,N)

    for kx in karr:
        for ky in karr:
            evals, evec = np.linalg.eigh(self.Hk(kx,ky)[0])
            Evec = evec.T 

            nE = np.array(fermi_dirac(evals,T))
            dnE = np.array(fermi_dirac_prime(evals,T))
            for k,i in enumerate(evals):
                for l,j in enumerate(evals):
                    if np.abs(i-j)<1e-10 or k==l:
                        pf = -dnE[l]
                    else:
                        pf = (nE[k]-nE[l])/(j-i)
                    
                    f1 = np.matmul(Evec[l],np.matmul(self.Hk(kx,ky,o=my)[0],Evec[k]))
                    f2 = np.matmul(Evec[k],np.matmul(self.Hk(kx,ky,o=ny)[0],Evec[l]))

                    M1 = np.matmul(self.Hk(kx,ky,o=my)[0],gammaz)
                    M2 = np.matmul(gammaz,self.Hk(kx,ky,o=ny)[0])

                    f3 = np.matmul(Evec[l],np.matmul(M1,Evec[k]))
                    f4 = np.matmul(Evec[k],np.matmul(M2,Evec[l]))

                    sum+=pf*(f1*f2-f3*f4)
    
    return sum

