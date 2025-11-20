from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.T = kwargs.get('T', 0)
        self.nu = kwargs.get('nu', 0.0)
        self.kind = kwargs.get('kind', 'DSL')
        if self.kind == 'DSL':
            self.site_num = self.N**2
        elif self.kind == 'dDSL':
            self.site_num = (self.N**2-self.N+1)

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

        def fact(i,j,kx,ky,dnx=0, dny=0):
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
                sublattice_f = np.exp((0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))
                uc_f = np.exp((0+1j)*(kx*v[0]+ky*v[1]))

                if dnx == 0 and dny ==0:
                    f+=-self.t*sublattice_f*uc_f
                else:
                    dx_sublattice_f = (0+1j)*(nn[0]-site[0])/self.N
                    dy_sublattice_f = (0+1j)*(nn[1]-site[1])/self.N

                    dx_uc_f = (0+1j)*v[0]
                    dy_uc_f = (0+1j)*v[1]

                    f+=-self.t*(dx_sublattice_f)**dnx*(dy_sublattice_f)**dny*sublattice_f*uc_f # f'(sublattice)*g(uc)
                    f+=-self.t*(dx_uc_f)**dnx*(dy_uc_f)**dny*sublattice_f*uc_f # f(sublattice)*g'(uc)
                
            return f
        
        def H_variety(kx, ky, dnx=0, dny=0):
            """Evaluate the Hamiltonian at given kx, ky."""
            deltas, mus, nss, Us = self.striped_props()

            n=int(self.site_num)
            H = np.zeros((n, n), dtype=complex) 

            hkp = np.zeros_like(H, dtype=complex) #normal state hamiltonian particles
            hkh = np.zeros_like(H, dtype=complex) #normal state hamiltonian holes
            

            for site in self.lat.nn:
                for nn in self.lat.nn[site]:
                    j=self.map_idx[site]
                    i=self.map_idx[nn]
                    hkp[i,j] = fact(i,j,kx, ky,dnx, dny)
                    hkh[i,j] = -np.conjugate(fact(i,j,-kx, -ky,dnx, dny))

            for i in self.map_site:
                if dnx==0 and dny==0: 
                    hkp[i,i] = -mus[i]-Us[i]/2*nss[i]
                    hkh[i,i] = mus[i]+Us[i]/2*nss[i]
            
            #interaction according to AHM
            HDk = np.zeros((n, n), dtype=complex)  #order parameter
            
            for i in self.map_site:
                if dnx==0 and dny==0:
                    site = self.map_site[i]
                    delta = deltas[i]
                    
                    HDk[i,i] = np.abs(Us[i])*delta
                    

            H0k_spinfull = np.block([[hkp, np.zeros_like(hkp)],
                                     [np.zeros_like(hkp), hkp]])
            H0kh_spinfull = np.block([[hkh, np.zeros_like(hkh)],
                                     [np.zeros_like(hkh), hkh]])
            HDk_spinfull = np.block([[np.zeros_like(HDk), HDk],
                                     [-HDk, np.zeros_like(HDk)]])
            
            HBdGk = np.block([[H0k_spinfull, HDk_spinfull],
                             [np.conjugate(HDk_spinfull.T), H0kh_spinfull]])
            
            HBdGk_up = np.block([[hkp, HDk],
                                 [np.conjugate(HDk.T),hkh]])
            

            return HBdGk, HBdGk_up, hkp


        def Hk(kx, ky, reduce=False, dnx=0, dny=0): 

            if not reduce:
                HBdGk = H_variety(kx, ky, dnx, dny)[0] #get full spin hamiltonian
            else:
                HBdGk = H_variety(kx, ky, dnx, dny)[1] #get only spin up BdG hamiltonian


            return HBdGk
            
        return Hk

    def solvHam(self, kx, ky, p='all'):
            '''
            solves hamiltonian for each pair of coordinates
            '''

            n = np.shape(kx)[0]
            if p=='all':
                eig = np.zeros((n, self.site_num*4))
            elif p=='part':
                eig = np.zeros((n, int(self.site_num)))

        
            for i in range(n):
                if p == 'all':
                    e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])) #solves full BdG-Hamiltonian
                elif p == 'part':
                    e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[:self.site_num, :self.site_num]) #solves only particle Hamiltonian
                eig[i]=np.sort(e)
                
            return eig.T

    def Es(self, p='all'):
        '''
        returns an array of the energies for each k-point in the BZ 
        for the normal state particle Hamiltonian (if p='part') or whole BdG-Hamiltonian (if p='all')
        Array should have dimension of (N_bands,N_kpoints**2)
        
        '''
        k=np.linspace(-np.pi,np.pi, 100)

        l = np.shape(k)[0]
        a1 = np.ones(l)
        if p=='all':
            E=np.empty((self.site_num*4, l))
        elif p=='part':
            E=np.empty((int(self.site_num), l))
        eps = 1e-15

        for i in k:
            Erow = self.solvHam(i*a1, k,p)
            E = np.concatenate((E, Erow), axis=1)
            E[np.abs(E)<eps]=0
        return E
    
    def DOS(self, N, E, b=0, sig=5e-2, p='part'):
        ''' 
        returns an array with the density of state values to each E

        E is an array (against which DOS is plotted)
        En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
        sigma is the width of the gauss function
        '''
        k = np.linspace(-np.pi, np.pi, N)

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
       
    def fermidirac(self,E,o=0):
        
        nE=0
        if o==0:
            if np.abs(E)<1e-6 and self.T!=0:
                nE = 1
            elif self.T==0:
                if E>0:
                    nE = 0
                else:
                    nE = 1
            else:
                nE = 1/(1+np.exp(E/self.T))
        elif o==1:
            if np.abs(E)<1e-6 and self.T!=0:
                nE = -1/(4*self.T)
            elif np.abs(E)<1e-6 and self.T==0:
                nE = -1/4
            elif self.T==0:
                nE = 0
            else:
                nE = -1/((1+np.exp(E/self.T))**2)*np.exp(E/self.T)/self.T
            
        return nE
    
    
    def SFW(self, N, my='x', ny='x'):

        gammaz = np.kron(np.diag([1,-1]), np.eye(int(self.site_num*4)))
        
        term_array = np.zeros((N**2, 3,int((self.site_num*2)**2)))

        summe = 0
        
        karr = np.linspace(0, 2*np.pi, N, endpoint=False)
        counter =0

        for kx in karr:
            for ky in karr:
                pflist = []
                parli = []
                diali = []

                H = self.Hk(kx,ky)[2]
                dHdmy = self.Hk(kx,ky,o=my)[2]
                dHdny = self.Hk(kx,ky,o=ny)[2]

                H[np.abs(H)<1e-14] =0
                dHdmy[np.abs(dHdmy)<1e-14] =0
                dHdny[np.abs(dHdny)<1e-14] =0
                
                M1 = np.matmul(dHdmy,gammaz)
                M2 = np.matmul(dHdny,gammaz)
                
                evals, evec = np.linalg.eigh(H)
                Evec = evec.T 

                nE = [self.fermidirac(E,o=0) for E in evals]
                dnE = [self.fermidirac(E,o=1) for E in evals]
                for k,i in enumerate(evals):
                    for l,j in enumerate(evals):
                        if np.abs(i-j)<1e-6 or k==l:
                            pf = -dnE[l]
                        else:
                            pf = (nE[l]-nE[k])/(i-j)

                        if pf==0:
                            summe+=0
                            f1, f2, f3, f4 = (0,0,0,0)

                        else:
                            f1 = np.matmul(np.conjugate(Evec[k].T),np.matmul(dHdmy,Evec[l]))
                            f2 = np.matmul(np.conjugate(Evec[l].T),np.matmul(dHdny,Evec[k]))

                            f3 = np.matmul(np.conjugate(Evec[k].T),np.matmul(M1,Evec[l]))
                            f4 = np.matmul(np.conjugate(Evec[l].T),np.matmul(M2,Evec[k]))

                            s = pf*(f1*f2-f3*f4)

                            summe+=s

                        pflist.append(pf/N)
                        diali.append(f1*f2/N)
                        parli.append(f3*f4/N)
                    
                term_array[counter]= np.array([pflist, diali, parli])

                counter +=1
        
        return summe/N**2, term_array
    
    def detSFW(self, N):
        xx = self.SFW(N, my='x', ny='x')[0]
        xy = self.SFW(N, my='x', ny='y')[0]
        yx = self.SFW(N, my='y', ny='x')[0]
        yy = self.SFW(N, my='y', ny='y')[0]
        ten = np.array([[xx,xy],[yx,yy]])
        return ten, np.sqrt(np.linalg.det(ten))

