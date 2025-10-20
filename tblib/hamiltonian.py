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

        self.mu = kwargs.get('mu', np.zeros(self.N)) #N different kinds of atoms arranged in diagonal stripes
        self.delta = kwargs.get('delta', np.zeros(self.N)) #N different kinds of atoms arranged in diagonal stripes
        self.ns = kwargs.get('ns', np.zeros(self.N))
        self.U = kwargs.get('U', np.ones(self.N))

        self.Hk = self.HBdG()



    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        if self.kind == 'DSL':
            lat = lattice.DiagonallyStripedLattice(N=self.N)
            n = self.N**2
        if self.kind == 'dDSL':
            lat = lattice.dDiagonallyStripedLattice(N=self.N)
            n = self.N**2-self.N+1

        as_d = {i: [site for site in lat.nn if site in [((i+j)%self.N, j) for j in range(self.N)]] for i in range(self.N)} # atomic species dictionary associate each site to its group of stripes
        c=0
        map_site={}
        map_idx={}
        for el in as_d:
            for site in as_d[el]:
                map_site[c] = site
                map_idx[site] = c
                c+=1

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
             
            hkp = np.zeros_like(H, dtype=complex) #normal state hamiltonian particles
            hkh = np.zeros_like(H, dtype=complex) #normal state hamiltonian holes
            

            for site in lat.nn:
                for nn in lat.nn[site]:
                    j=map_idx[site]
                    i=map_idx[nn]
                    hkp[i,j] = fact(i,j,kx, ky)
                    hkh[i,j] = -np.conjugate(fact(i,j,-kx, -ky))

            for os in range(n):
                site = map_site[os]
                if site in lat.nn:
                    num = [key for key,val in as_d.items() if site in val]
                    hkp[os,os] = -self.mu[num[0]]-self.U[num[0]]/2*self.ns[num[0]]
                    hkh[os,os] = np.conjugate(self.mu[num[0]]+self.U[num[0]]/2*self.ns[num[0]])
            
            #interaction according to AHM
            darr = np.zeros((2*n, 2*n), dtype=complex)  #order parameter
            #ddarr = np.zeros_like(H, dtype=complex) # order parameter dagger
            for i in map_site:
                site = map_site[i]
                num = [key for key,val in as_d.items() if site in val]
                delta = self.delta[num[0]]
                darr[i,n+i] = -delta
                darr[n+i,i] = delta
                    

            #hk[np.abs(hk) < eps] = 0
            A = np.zeros((self.dim,self.dim), dtype=complex)
            A[:n, :n] = hkp
            A[n:2*n, n:2*n] = hkp
            A[n*2:n*3, n*2:n*3]=hkh
            A[n*3:, n*3:]=hkh

            A[:2*n, 2*n:] = darr
            A[2*n:, :2*n] = np.conjugate(darr.T)

            return A, hkp

        return Hk

    def solvHam(self, kx, ky, p='all'):
            '''
            solves hamiltonian for each pair of coordinates
            '''

            eps = 1e-15
            n = np.shape(kx)[0]
            if p=='all':
                eig = np.zeros((n, self.dim))
            elif p=='part':
                eig = np.zeros((n, int(self.dim/4)))

        
            for i in range(n):
                if p == 'all':
                    e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[0]) #solves full BdG-Hamiltonian
                elif p == 'part':
                    e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[1]) #solves only particle Hamiltonian
                #e[np.abs(e)<eps]=0
                eig[i]=np.sort(e)
                
            return eig.T
    
    def plot_bands(self,k, p='all'):

        k1 = np.ones(100)
        k0 = np.zeros(100)
        path = np.concatenate((k, k, k*np.sqrt(2)))
        kx = np.concatenate((k,k[-1]*k1, k[::-1]))
        ky = np.concatenate((k0, k, k[::-1]))

        pl = [i for i in range(np.shape(path)[0])]
        
        energies = self.solvHam(kx, ky, p)

        emax = np.amax(energies)
        emax = emax+0.1*emax

        plt.figure(figsize=(8,6))
        plt.xlabel("$(k_x a,k_y a)$", size='x-large')
        plt.ylabel("E", size='x-large')
        plt.yticks(size='x-large')
        plt.xticks(ticks= [0, 100, 200, 299], labels=[r"$\Gamma$",r"X",r"M", r"$\Gamma$"], size='x-large')

        for i in energies:
            plt.plot(pl, i, color='black')

        plt.vlines([0, 99, 199, 299], [-emax, -emax, -emax, -emax], [emax, emax, emax, emax], colors= 'grey', linestyles='--')
        plt.show()
        return

    def Es(self, k, p='part'):
        '''
        returns an array of the energies for each k-point in the BZ 
        for the normal state particle Hamiltonian (if p='part') or whole BdG-Hamiltonian (if p='all')
        Array should have dimension of (N_bands,N_kpoints**2)
        
        '''
        l = np.shape(k)[0]
        a1 = np.ones(l)
        if p=='all':
            E=np.empty((self.dim, l))
        elif p=='part':
            E=np.empty((int(self.dim/4), l))
        eps = 1e-15

        for i in k:
            Erow = self.solvHam(i*a1, k, p)
            E = np.concatenate((E, Erow), axis=1)
            E[np.abs(E)<eps]=0
        return E
    
    
    
    def DOS(self, E, k, b=0, sig=5e-2, p='part'):
        ''' 
        returns an array with the density of state values to each E

        E is an array (against which DOS is plotted)
        En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
        sigma is the width of the gauss function
        '''
        def Gauss(E, En, sig):
            return np.exp(-(E-En)**2/(2*sig**2))
            
        En = self.Es(k, p)
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
 


