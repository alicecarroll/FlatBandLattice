import numpy as np
from .fermi import fermi_dirac, fermi_dirac_prime

def DeltaN(model, HF, temperature=.0, nk=40):

    karr = np.linspace(-np.pi, np.pi, nk)
    k_grid = np.array(np.meshgrid(karr, karr)).T.reshape(-1,2)

    a = int(model.dim/2)   
    Delta=np.zeros((a,a), dtype=object)
    Nu=np.zeros((a,a), dtype=object)

    HBdG = model.get_HBdG()

    for kx, ky in k_grid:

        evals, evec = np.linalg.eigh(HBdG(kx, ky))

        evals = np.flip(evals)[0:a]

        Carr= np.flip(evec.T, axis=1)

        uk = Carr[0:a, 0:a]
        vk = Carr[0:a,a:]
        v=np.conjugate(evec.T[a:, 0:a])
        u=np.conjugate(evec.T[a:, a:])
        
        if T==0:
            el=np.matmul(np.conjugate(u.T),v)
            Delta+=el
            if HF:
                nu_el = np.matmul(np.conjugate(v.T), v)
                Nu+=nu_el

        else:
            el=np.matmul(np.conjugate(u.T),np.matmul(np.diag(1/(1+np.exp(-evals/T))), v))+np.matmul(vk.T,np.matmul(np.diag(1/(1+np.exp(evals/T))),np.conjugate(uk)))
            Delta+=el
            if HF:
                nu_el = np.matmul(np.conjugate(v.T),np.matmul(np.diag(1/(1+np.exp(-evals/T))), v))+np.matmul(uk.T,np.matmul(1/(1+np.exp(evals/T)), np.conjugate(uk)))
                Nu+=nu_el
        if np.isnan(el.any()):
            print(x, y, ': \n', el)
            print('u\n', u)
            print('evals\n', evals)          
    
    finNu = np.diag(Nu)/N**2
    nu = [finNu[i]+finNu[int(i+a/2)] for i in range(int(a/2))]
    delta = [-np.abs(Us[i])/N**2*Delta[i,int(i+a/2)] for i in range(int(a/2))]

    delta2 = [delta[self.map_idx[(i,0)]] for i in range(self.N)]
    nu2 = [nu[self.map_idx[(i,0)]] for i in range(self.N)]
    return delta2,nu2
        
def Deltra(self, N, T=0, g=0.01, HF=False, Nmax=20, Nmin=10, alpha=0.5):
    
    delarr = np.array(self.delta)
    nuarr = np.array(self.ns)

    dels = delarr.reshape(self.N,1)
    nus = nuarr.reshape(self.N,1)

    c=0
    while (c<Nmax and (np.std(np.abs(dels), axis=1)>g).any()) or c<Nmin:
        c+=1

        Vals = self.DeltaN(N, T, HF)

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

