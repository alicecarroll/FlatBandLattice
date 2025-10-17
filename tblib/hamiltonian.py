from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.kind = kwargs.get('kind', 'DSL')
        if self.kind == 'DSL':
            self.dim = self.N**2*4
        elif self.kind == 'dDSL':
            self.dim = (self.N**2-self.N+1)*4

        self.t = kwargs.get('t', 1.0)
        self.mu = kwargs.get('mu', np.zeros(self.N))

        self.Hk = self.HBdG()



    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        if self.kind == 'DSL':
            lat = lattice.DiagonallyStripedLattice(N=self.N)
            n = self.N**2
        if self.kind == 'dDSL':
            lat = lattice.dDiagonallyStripedLattice(N=self.N)
            n = self.N**2-self.N+1

        mu_d = {i: [site for site in lat.nn if site in [((i+j)%self.N, j) for j in range(self.N)]] for i in range(self.N)} #associate each site to its group of stripes via the chemical potential index

        c=0
        map_site={}
        map_idx={}
        for el in mu_d:
            for site in mu_d[el]:
                map_site[c] = site
                map_idx[site] = c
                c+=1

        # print(mu_d)

        H = np.zeros((n, n), dtype=complex)        

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

            for site in lat.nn:
                for nn in lat.nn[site]:
                    j=map_idx[site]
                    i=map_idx[nn]
                    hkp[i,j] = fact(i,j,kx, ky)
                    hkh[i,j] = -np.conjugate(fact(i,j,kx, ky))

            for os in range(n):
                site = map_site[os]
                if site in lat.nn:
                    num = [key for key,val in mu_d.items() if site in val]
                    hkp[os,os] = self.mu[num[0]]
                    hkh[os,os] = -np.conjugate(self.mu[num[0]])
            
            #hk[np.abs(hk) < eps] = 0
            A = np.zeros((self.dim,self.dim), dtype=complex)
            A[:n, :n] = hkp
            A[n:2*n, n:2*n] = hkp
            A[n*2:n*3, n*2:n*3]=hkh
            A[n*3:, n*3:]=hkh

            return A, hkp

        return Hk

    def solvHam(self, kx, ky):
            '''
            solves hamiltonian for each pair of coordinates
            '''
            eps = 1e-15
            n = np.shape(kx)[0]
            eig = np.zeros((n, self.dim))
        
            for i in range(n):
                e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[0]) #

                #e[np.abs(e)<eps]=0
                eig[i]=np.sort(e)
                
            return eig.T

    def solvParticleHam(self, kx, ky):
            '''
            solves hamiltonian for each pair of coordinates
            '''
            eps = 1e-15
            n = np.shape(kx)[0]
            eig = np.zeros((n, int(self.dim/4)))
        
            for i in range(n):
                e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[1]) #

                #e[np.abs(e)<eps]=0
                eig[i]=np.sort(e)
                
            return eig.T
    
    def Es(self, k):
        '''
        returns an array of the energies for each k-point in the BZ 
        for the normal state particle Hamiltonian 
        Array should have dimension of (N_bands,N_kpoints**2)
        
        '''
        l = np.shape(k)[0]
        a1 = np.ones(l)
        E=np.empty((int(self.dim/4), l))
        eps = 1e-15

        for i in k:
            Erow = self.solvParticleHam(i*a1, k)
            E = np.concatenate((E, Erow), axis=1)
            E[np.abs(E)<eps]=0
        return E
    
    def plot_particle_bands(self,k):

        k1 = np.ones(100)
        k0 = np.zeros(100)
        path = np.concatenate((k, k, k*np.sqrt(2)))
        kx = np.concatenate((k,k[-1]*k1, k[::-1]))
        ky = np.concatenate((k0, k, k[::-1]))

        p = [i for i in range(np.shape(path)[0])]
        np.shape(p)
        energies = self.solvParticleHam(kx, ky)

        emax = np.amax(energies)
        emax = emax+0.1*emax

        plt.figure(figsize=(8,6))
        plt.xlabel("$(k_x a,k_y a)$", size='x-large')
        plt.ylabel("E", size='x-large')
        plt.yticks(size='x-large')
        plt.xticks(ticks= [0, 100, 200, 299], labels=[r"$\Gamma$",r"X",r"M", r"$\Gamma$"], size='x-large')

        for i in energies:
            plt.plot(p, i, color='black')

        plt.vlines([0, 99, 199, 299], [-emax, -emax, -emax, -emax], [emax, emax, emax, emax], colors= 'grey', linestyles='--')
        plt.show()
        return
    
    def DOS(self, E, k, b=0, sig=5e-2):
        ''' 
        returns an array with the density of state values to each E

        E is an array (against which DOS is plotted)
        En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
        sigma is the width of the gauss function
        '''
        def Gauss(E, En, sig):
            return np.exp(-(E-En)**2/(2*sig**2))
            
        En = self.Es(k)
        l = np.shape(En)[1]
        arr1 = np.ones(l)
        s1 = 0
        res = np.ones(np.shape(E)[0])
        c=0
        for j in E:
            s1 += np.sum(Gauss(j*arr1,En[b], sig))
            res[c] = s1
            s1=0
            c+=1
        return res
 


