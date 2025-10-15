from . import lattice
import numpy as np
import scipy.constants as sc


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.kind = kwargs.get('kind', 'DLS')

        self.t = kwargs.get('t', 1.0)
        self.mu = kwargs.get('mu', np.zeros(self.N**2))

        self.Hk = self.HBdG()



    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        if self.kind == 'DLS':
            lat = lattice.DiagonallyStripedLattice(N=self.N)
        
        map = {site: i for i,site in enumerate(lat.sites)}
        d1 = self.N**2 #for one spin direction
        H = np.zeros((d1, d1), dtype=object)

        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0

        for i in range(d1): 
            H[i,i] = lambda kx,ky: -self.mu[i]
        H[1,1] = lambda kx,ky: -30#self.mu[1]

        for site in lat.nn:
            for delta in lat.nn[site]:
                for nn in lat.nn[site][delta]:
                    j=map[site]
                    i=map[nn]
                    H[i,j] = lambda kx,ky: -self.t*np.exp(-(0+1j)*(kx*(nn[0]-site[0])/self.N+ky*(nn[1]-site[1])/self.N))*np.exp((0+1j)*(kx*delta[0]+ky*delta[1]))

        

        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
             
            hk = np.empty_like(H, dtype=complex)
            eps = 1e-15
            for index in np.ndindex(H.shape): hk[index] = H[index](kx, ky)
            #hk[np.abs(hk) < eps] = 0
            A = np.zeros((d1*4,d1*4), dtype=complex)
            A[:d1, :d1] = hk
            A[d1:2*d1, d1:2*d1] = hk
            A[d1*2:, d1*2:]=-np.conjugate(A[:2*d1, :2*d1])

            return A

        return Hk




 


