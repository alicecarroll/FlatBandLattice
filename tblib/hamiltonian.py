from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.T = kwargs.get('T', 0)
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

        if self.kind == 'DSL':
            self.lat = lattice.DiagonallyStripedLattice(N=self.N)
        if self.kind == 'dDSL':
            self.lat = lattice.dDiagonallyStripedLattice(N=self.N)

        self.as_d = {i: [site for site in self.lat.nn if site in [((i+j)%self.N, j) for j in range(self.N)]] for i in range(self.N)} # atomic species dictionary associate each site to its group of stripes
        c=0
        self.map_site={}
        self.map_idx={}
        for el in self.as_d:
            for site in self.as_d[el]:
                self.map_site[c] = site
                self.map_idx[site] = c
                c+=1

        self.Hk = self.HBdG()

        
    def striped_props(self):
        deltalist = []
        mulist = []
        nslist = []
        Ulist = [] 
        
        for i in self.map_site:
            site = self.map_site[i]
            num = [key for key,val in self.as_d.items() if site in val]
            
            deltalist.append(self.delta[num[0]])
            mulist.append(self.mu[num[0]])
            nslist.append(self.ns[num[0]])
            Ulist.append(self.U[num[0]])

        return deltalist, mulist, nslist, Ulist
            
        
    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        n=int(self.dim/4)

        H = np.zeros((n, n), dtype=complex)        

        def fact(i,j,kx,ky,o=0):
            '''
            returns the Hamiltonian entry for a hopping process from site j to i

            o = 0,'x' or 'y' determines whether the normal Hamiltonian is calculated (0)
            or a derivative to kx ('x') or ky ('y')
            '''
            site = self.map_site[j]
            nn = self.map_site[i]
            R = self.lat.nn[site][nn]
            f=0

            for v in R:
                if o==0:
                    f+=-self.t*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
                elif o=='x':
                    f+=-self.t*(-(0+1j)*(nn[0]-site[0])/self.N)*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
                    f+=-self.t*(-(0+1j)*v[0])*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
                elif o=='y':
                    f+=-self.t*(-(0+1j)*(nn[1]-site[1])/self.N)*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
                    f+=-self.t*(-(0+1j)*v[1])*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*v[0]+ky*v[1]))
            
            return f
        
        


        def Hk(kx, ky, o=0): 
            """Evaluate the Hamiltonian at given kx, ky."""
            deltas, mus, nss, Us = self.striped_props()

            hkp = np.zeros_like(H, dtype=complex) #normal state hamiltonian particles
            hkh = np.zeros_like(H, dtype=complex) #normal state hamiltonian holes
            

            for site in self.lat.nn:
                for nn in self.lat.nn[site]:
                    j=self.map_idx[site]
                    i=self.map_idx[nn]
                    hkp[i,j] = fact(i,j,kx, ky,o)
                    hkh[i,j] = -np.conjugate(fact(i,j,-kx, -ky,o))

            for i in self.map_site:
                if o ==0: 
                    hkp[i,i] = -mus[i]-Us[i]/2*nss[i]
                    hkh[i,i] = mus[i]+Us[i]/2*nss[i]
            
            #interaction according to AHM
            darr = np.zeros((2*n, 2*n), dtype=complex)  #order parameter
            #ddarr = np.zeros_like(H, dtype=complex) # order parameter dagger
            for i in self.map_site:
                if o==0:
                    site = self.map_site[i]
                    delta = deltas[i]
                    darr[i,n+i] = delta
                    darr[n+i,i] = -delta
                    

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
    
    def DeltaN(self, N, HF):
            Us = self.striped_props()[3]

            karr = np.linspace(-np.pi, np.pi, N)
            a = int(self.dim/2)            

            Delta=np.zeros((a,a), dtype=object)
            Nu=np.zeros((a,a), dtype=object)

            c=0
            for x in karr:
                for y in karr:
                    c+=1

                    evals1, Evec = np.linalg.eigh(self.Hk(x, y)[0])
                    Evec = Evec.T

                    evals = np.flip(evals1)[0:a]

                    Carr= np.flip(Evec, axis=1)

                    uk = Carr[0:a, 0:a]
                    vk = Carr[0:a,a:]
                    v=np.conjugate(Evec[a:, 0:a])
                    u=np.conjugate(Evec[a:, a:])
                    
                    if self.T==0:
                        el=np.matmul(np.conjugate(u.T),v)
                        Delta+=el
                        if HF:
                            nu_el = np.matmul(np.conjugate(v.T), v)
                            Nu+=nu_el

                    else:
                        el=np.matmul(np.conjugate(u.T),np.matmul(np.diag(1/(1+np.exp(-evals/self.T))), v))+np.matmul(vk.T,np.matmul(np.diag(1/(1+np.exp(evals/self.T))),np.conjugate(uk)))
                        Delta+=el
                        if HF:
                            nu_el = np.matmul(np.conjugate(v.T),np.matmul(np.diag(1/(1+np.exp(-evals/self.T))), v))+np.matmul(uk.T,np.matmul(1/(1+np.exp(evals/self.T)), np.conjugate(uk)))
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
        
    def Deltra(self, N, g=0.01, HF=False, Nmax=20, Nmin=10, alpha=0.5):
        
        delarr = np.array(self.delta)
        nuarr = np.array(self.ns)

        dels = delarr.reshape(self.N,1)
        nus = nuarr.reshape(self.N,1)

        c=0
        while (c<Nmax and (np.std(np.abs(dels), axis=1)>g).any()) or c<Nmin:
            c+=1

            Vals = self.DeltaN(N, HF)

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



    def fermidirac(self,E,o=0):
        
        nE=0
        if o==0:
            if np.abs(E)<1e-14 and self.T!=0:
                nE = 1
            elif self.T==0:
                if E>0:
                    nE = 0
                else:
                    nE = 1
            else:
                nE = 1/(1+np.exp(E/self.T))
        elif o==1:
            if np.abs(E)<1e-14 and self.T!=0:
                nE = 1/(4*self.T)
            elif self.T==0:
                nE = 0
            else:
                nE = 1/((1+np.exp(E/self.T))**2)*np.exp(E)/self.T
        
        return nE
    

    def SFW(self, N, my='x', ny='y'):

        gammaz = np.kron(np.diag([1,-1]), np.eye(int(self.dim/2)))
        sum = 0
        karr = np.linspace(-np.pi,np.pi*0.1,N)

        for kx in karr:
            for ky in karr:

                M1 = np.matmul(self.Hk(kx,ky,o=my)[0],gammaz)
                M2 = np.matmul(gammaz,self.Hk(kx,ky,o=ny)[0])
                
                evals, evec = np.linalg.eigh(self.Hk(kx,ky)[0])
                Evec = evec.T 

                nE = [self.fermidirac(E,o=0) for E in evals]
                dnE = [self.fermidirac(E,o=1) for E in evals]
                for k,i in enumerate(evals):
                    for l,j in enumerate(evals):
                        if np.abs(i-j)<1e-10 or k==l:
                            pf = -dnE[l]
                        else:
                            pf = (nE[k]-nE[l])/(j-i)

                        if pf==0:
                            sum+=0
                        else:
                            f1 = np.matmul(np.conjugate(Evec[l]),np.matmul(self.Hk(kx,ky,o=my)[0],Evec[k]))
                            f2 = np.matmul(np.conjugate(Evec[k]),np.matmul(self.Hk(kx,ky,o=ny)[0],Evec[l]))

                            f3 = np.matmul(np.conjugate(Evec[l]),np.matmul(M1,Evec[k]))
                            f4 = np.matmul(np.conjugate(Evec[k]),np.matmul(M2,Evec[l]))

                            sum+=pf*(f1*f2-f3*f4)
        
        return sum
    
    def detSFW(self, N):
        xx = self.SFW(N, my='x', ny='x')
        xy = self.SFW(N, my='x', ny='y')
        yx = self.SFW(N, my='y', ny='x')
        yy = self.SFW(N, my='y', ny='y')
        ten = np.array([[xx,xy],[yx,yy]])
        return np.sqrt(np.linalg.det(ten))

