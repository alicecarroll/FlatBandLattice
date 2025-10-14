from . import lattice
import numpy as np
import scipy.constants as sc


class Model:
    def __init__(self, **kwargs):

        self.N = kwargs.get('N', 2)
        self.kind = kwargs.get('kind', 'DLS')

        self.t = kwargs.get('t', 1.0)
        self.mu = kwargs.get('mu', [m for m in range(self.N)])

        self.Hk = self.HBdG()



    def HBdG(self):
        """Construct the k-space Hamiltonian function."""

        if self.kind == 'DLS':
            lat = lattice.dDiagonallyStripedLattice(N=self.N)
        
        map = {site: i for i,site in enumerate(lat.sites)}

        H = np.zeros((self.N*4, self.N*4), dtype=object)

        for index, _ in np.ndenumerate(H): H[index] = lambda kx, ky: 0



        def Hk(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""
             
            hk = np.empty_like(H, dtype=complex)
            eps = 1e-15
            for index in np.ndindex(H.shape): hk[index] = H[index](kx, ky)
            #hk[np.abs(hk) < eps] = 0

            return hk

        return Hk




 


