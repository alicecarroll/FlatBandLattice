from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

import numbers


class Model:
    def __init__(self, lat, **kwargs):

        # Lattice and dimensional parameters
        self.lat = lat

        # Physical parameters
        self.t = kwargs.get('t', 1.0)
        self.mu = kwargs.get('mu', np.zeros(self.lat.nsites))
        self.delta = kwargs.get('delta', np.zeros(self.lat.nsites))
        self.ns = kwargs.get('ns', np.zeros(self.lat.nsites))
        self.U = kwargs.get('U', np.zeros(self.lat.nsites))

    @property # Number of sites
    def nsites(self): return self.lat.nsites

    @property # BdG dimension
    def dim(self): return 4 * self.nsites

    def get_onsite_energy(self, i):
        return -self.mu[i] - self.U[i]/2 * self.ns[i]

    def get_onsite_pairing(self, i):
        return self.delta[i]

    def get_hopping(self, i, j, k, dnx=0, dny=0):
        site = self.lat.map_sites[j]
        nn = self.lat.map_sites[i]
        dR = np.matmul(np.stack(self.lat.nn[site][nn]), self.lat.lattice_vecs)
        dr = -1j * (np.asarray(nn) - np.asarray(site) + dR)
        farr = (dr[:,0]**dnx) * (dr[:,1]**dny) * np.exp(np.dot(dr, k))
        return -self.t * np.sum(farr)
        
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

    # Inputs accepted: None, float/int, list/array of size (nsites,)
    for param in ['mu', 'delta', 'ns', 'U']: 
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(self.nsites)
        elif isinstance(value, float) or isinstance(value, int): value = value * np.ones(self.nsites)
        else: assert  np.array(value).shape == (self.nsites,), f"Parameter {param}={value} of shape {np.array(value).shape} must be of shape ({self.nsites},)"
        setattr(self, param, value)


def _init_DSLmodel_base(self, N=1, **kwargs):
    """Base initialization for DSL-type models"""

    # Map that takes site index to diagonal index
    map_diag = {self.map_indices[site]: (site[0]+site[1])%self.N for site in self.sites}

    # Inputs accepted: None, float/int, list/array of size (nsites,) or (N,)
    for param in ['mu', 'delta', 'ns', 'U']: 
        value = kwargs.get(param, None)

        if value is None: value = np.zeros(self.nsites)
        elif np.array(value).shape == (N,): 
            value = np.array([value[map_diag[i]] for i in range(self.nsites)])

        elif isinstance(value, float) or isinstance(value, int): value = value * np.ones(self.nsites)
        else: assert  np.array(value).shape == (self.nsites,), f"Parameter {param}={value} must be of shape ({self.nsites},)"
        setattr(self, param, value)

### Specific Models ###

class SquareLatticeModel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.SquareLattice(N=N)
        super().__init__(lat=self.lat)

        _init_square_base(self, **kwargs)

class DSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.DiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        _init_DSLmodel_base(self, N, **kwargs)

class dDSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.dDiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        _init_square_base(self, N, **kwargs)

class LiebNmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.LiebNLattice(N=N)
        super().__init__(lat=self.lat)

        _init_square_base(self, N, **kwargs)