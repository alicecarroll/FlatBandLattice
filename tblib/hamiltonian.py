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
        print(mu_d)
        c=0
        map={}
        map2={}
        for el in mu_d:
            for site in mu_d[el]:
                map[c] = site
                map2[site] = c
                c+=1
        print(map2)


        d1 = self.N**2 #for one spin direction
        H = np.zeros((d1, d1), dtype=complex)

        
        
        def fact(i,j,kx,ky):
            site = map[j]
            nn = map[i]
            R = lat.nn[site][nn]
            f=0

            for v in R:
                f+=-self.t*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
            
            return f


        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
             
            hkp = np.empty_like(H, dtype=complex)
            hkh = np.empty_like(H, dtype=complex)

            eps = 1e-15
            for site in lat.nn:
                for nn in lat.nn[site]:
                    j=map2[site]
                    i=map2[nn]
                    hkp[i,j] = fact(i,j,kx, ky)
                    hkh[i,j] = -np.conjugate(fact(i,j,kx, ky))
            if self.kind == 'DSL':
                for os in range(d1):
                    site = map[os]
                    num = [key for key,val in mu_d.items() if site in val]
                    hkp[os,os] = self.mu[num[0]]
                    hkh[os,os] = -np.conjugate(self.mu[num[0]])
            elif self.kind == 'dDSL':
                hkp[0,0]=self.mu[0]
                hkh[0,0]=-np.conjugate(self.mu[0])

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

 


