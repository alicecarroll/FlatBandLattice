from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

class Model:
    def __init__(self, lat=lattice.Lattice(sites=[]), **kwargs):

        # Lattice and dimensional parameters
        self.lat = lat
        self.dim = 0    # BdG Hamiltonian dimension

        # Physical parameters
        self.t = kwargs.get('t', 1.0)
        self.mu = None
        self.delta = None
        self.ns = None
        self.U = None

        self.H0 = None
        self.HD = None
        self.HBdG = None

    @property
    def nsites(self): return self.lat.nsites

    def get_onsite_energy(self, i):
        """Default onsite energy function."""
        return 0.0

    def get_hopping(self, i, j, k, dnx=0, dny=0):
        """Default hopping function."""
        return 0.0

    def get_onsite_pairing(self, i):
        """Default onsite energy function."""
        return 0.0
        
    def get_H0(self, dnx=0, dny=0):
        """Construct the (dnx, dny) derivative of the 
        k-space normal state Hamiltonian function."""

        def H0(kx, ky):
            """Evaluate the normal state Hamiltonian at given kx, ky."""

            H0k = np.zeros( ( self.nsites, self.nsites ), dtype=complex)
            for site, nns in self.lat.nn.items():
                j = self.lat.map_indices[site]
                H0k[j,j] += self.get_onsite_energy(j)
                for nn in nns:
                    i = self.lat.map_indices[nn]
                    H0k[i,j] += self.get_hopping(i, j, np.asarray((kx, ky)), dnx=dnx, dny=dny)
            return H0k

        return H0            
        
    def get_HBdG(self, dnx=0, dny=0):
        """Construct the (dnx, dny) derivative of the 
        k-space BdG Hamiltonian function."""

        H0 = self.get_H0(dnx=dnx, dny=dny)

        def HBdG(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""

            H0kp = H0(kx, ky)                   # particle sector
            H0kh = -np.conjugate(H0(-kx, -ky))  # hole sector
            HDk = np.zeros_like(H0kp)           # pairing matrix

            for site, _ in self.lat.nn.items():
                j = self.lat.map_indices[site]
                HDk[j,j] += self.get_onsite_pairing(j)

            H0kp_spinfull = np.block([[H0kp, np.zeros_like(H0kp)],
                                  [np.zeros_like(H0kp), H0kp]])
            H0kh_spinfull = np.block([[H0kh, np.zeros_like(H0kh)],
                                  [np.zeros_like(H0kh), H0kh]])
            HDk_spinfull = np.block([[np.zeros_like(HDk), HDk],
                                  [-HDk, np.zeros_like(HDk)]])

            # Compute tensor product to form full BdG Hamiltonian
            HBdGk = np.block([[H0kp_spinfull, HDk_spinfull], 
                              [np.conjugate(HDk_spinfull.T), H0kh_spinfull]])

            return HBdGk
        
        return HBdG    

### Model Initializations ###
    
def _init_square_base(self, N=1, **kwargs):
    """Base initialization for square-lattice-type models"""

    self.t = kwargs.get('t', 1.0)

    for param in ['mu', 'delta', 'ns', 'U']:
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(self.nsites)
        assert value.shape == (self.nsites,), f"Parameter {param} must be of shape (n,)"
        setattr(self, param, value)

    def get_onsite_energy(i):
        return -self.mu[i] - self.U[i]/2 * self.ns[i]
    setattr(self, 'get_onsite_energy', get_onsite_energy)

    def get_hopping(i, j, k, dnx=0, dny=0):
        site = self.lat.map_sites[j]
        nn = self.lat.map_sites[i]
        dR = N * np.stack(self.lat.nn[site][nn])
        dr = -1j * (self.lat.mapback(np.asarray(nn) - np.asarray(site)) + dR)
        farr = (dr[:,0]**dnx) * (dr[:,1]**dny) * np.exp(np.dot(dr, k))
        return self.t * np.sum(farr)
    setattr(self, 'get_hopping', get_hopping)

    def get_onsite_pairing(i):
        return self.delta[i]
    setattr(self, 'get_onsite_pairing', get_onsite_pairing)

def _init_DSLmodel_base(self, N=1, **kwargs):
    """Base initialization for DSL-type models"""

    self.t = kwargs.get('t', 1.0)
    
    for param in ['mu', 'delta', 'ns', 'U']:
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(N)
        assert value.shape == (N,), f"Parameter {param} must be of shape (N,)"
        setattr(self, param, value)

    def get_onsite_energy(i):
        idx = self.lat.map_diag[i]
        return -self.mu[idx] - self.U[idx]/2 * self.ns[idx]
    setattr(self, 'get_onsite_energy', get_onsite_energy)

    def get_hopping(i, j, k, dnx=0, dny=0):
        site = self.lat.map_sites[j]
        nn = self.lat.map_sites[i]
        dR = N * np.stack(self.lat.nn[site][nn])
        dr = -1j * (self.lat.mapback(np.asarray(nn) - np.asarray(site)) + dR)
        farr = (dr[:,0]**dnx) * (dr[:,1]**dny) * np.exp(np.dot(dr, k))
        return self.t * np.sum(farr)
    setattr(self, 'get_hopping', get_hopping)

    def get_onsite_pairing(i):
        return self.delta[self.lat.map_diag[i]]
    setattr(self, 'get_onsite_pairing', get_onsite_pairing)   

### Specific Models ###

class SquareLatticeModel(Model):
    def __init__(self, **kwargs):

        self.lat = lattice.SquareLattice()
        super().__init__(lat=self.lat)

        _init_square_base(self, 1, **kwargs)
        self.dim = 4 * self.nsites

class DSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.DiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        _init_DSLmodel_base(self, N, **kwargs)
        self.dim = 4 * self.nsites

class dDSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.dDiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        _init_square_base(self, N, **kwargs)
        self.dim = 4 * self.nsites

class LiebNmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.LiebNLattice(N=N)
        super().__init__(lat=self.lat)

        _init_square_base(self, N, **kwargs)
        self.dim = 4 * self.nsites