from . import lattice
import numpy as np
import scipy.constants as sc


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.dim = self.N**2*4
        self.kind = kwargs.get('kind', 'DSL')

        self.t = kwargs.get('t', 1.0)
        self.mu = kwargs.get('mu', np.zeros(self.N))

        self.Hk = self.HBdG()



    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        if self.kind == 'DSL':
            lat = lattice.DiagonallyStripedLattice(N=self.N)
        if self.kind == 'dDSL':
            lat = lattice.dDiagonallyStripedLattice(N=self.N)

        
        mu_d = {i: [((i+j)%self.N, j) for j in range(self.N)] for i in range(self.N)} #associate each site to its group of stripes via the chemical potential index
        
        c=0
        map_site={}
        map_idx={}
        for el in mu_d:
            for site in mu_d[el]:
                map_site[c] = site
                map_idx[site] = c
                c+=1

        # print(mu_d)
        # print(map_idx)

        d1 = self.N**2 #for one spin direction
        H = np.zeros((d1, d1), dtype=complex)        
        
        def fact(i,j,kx,ky):
            site = map_site[j]
            nn = map_site[i]
            R = lat.nn[site][nn]
            f=0

            for v in R:
                f+=-self.t*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
            
            return f


        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
             
            hkp = np.zeros_like(H, dtype=complex)
            hkh = np.zeros_like(H, dtype=complex)

            eps = 1e-15
            for site in lat.nn:
                for nn in lat.nn[site]:
                    j=map_idx[site]
                    i=map_idx[nn]
                    hkp[i,j] = fact(i,j,kx, ky)
                    hkh[i,j] = -np.conjugate(fact(i,j,kx, ky))

            if self.kind == 'DSL':
                for os in range(d1):
                    site = map_site[os]
                    num = [key for key,val in mu_d.items() if site in val]
                    hkp[os,os] = self.mu[num[0]]
                    hkh[os,os] = -np.conjugate(self.mu[num[0]])
            elif self.kind == 'dDSL':
                for os in range(d1):
                    site = map_site[os]
                    if site in lat.nn:
                        num = [key for key,val in mu_d.items() if site in val]
                        hkp[os,os] = self.mu[num[0]]
                        hkh[os,os] = -np.conjugate(self.mu[num[0]])

            #hk[np.abs(hk) < eps] = 0
            A = np.zeros((self.dim,self.dim), dtype=complex)
            A[:d1, :d1] = hkp
            A[d1:2*d1, d1:2*d1] = hkp
            A[d1*2:d1*3, d1*2:d1*3]=hkh
            A[d1*3:, d1*3:]=hkh

            return A

        return Hk

    def solvHam(self, kx, ky):
            '''
            solves hamiltonian for each pair of coordinates
            '''
            eps = 1e-15
            n = np.shape(kx)[0]
            eig = np.zeros((n, self.dim))
        
            for i in range(n):
                e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])) #

                #e[np.abs(e)<eps]=0
                eig[i]=np.sort(e)
                
            return eig.T
    
    def Es(self, k):
        #E=np.array([[],[],[]])
        l = np.shape(k)[0]
        a1 = np.ones(l)
        E=np.empty((self.dim, l))
        eps = 1e-15

        for i in k:
            Erow = self.solvHam(i*a1, k)
            E = np.concatenate((E, Erow), axis=1)
            E[np.abs(E)<eps]=0
        return E

 


