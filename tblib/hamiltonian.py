from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt


class Model:
    def __init__(self, lat=lattice.Lattice(sites=[]), **kwargs):
        
        # Lattice and dimensional parameters
        self.lat = lat
        self.dim = 0    # BdG Hamiltonian dimension
        self.n = 0      # Number of sites per u.c.
        self.N = 0 # number of atoms until next u.c. along x-axis for example


        # Physical parameters
        self.t = kwargs.get('t', 1.0)

        self.mu = None
        self.delta = None
        self.ns = None
        self.U = None
        self.T = kwargs.get('T', 0)
        self.nu = kwargs.get('nu', 0.0)

        self.H0 = None
        self.HD = None
        self.HBdG = None


    def get_onsite_energy(self, i):
        """Default onsite energy function."""
        return 0.0

    def get_hopping(self, i, j, k, dnx=0, dny=0):
        """Default hopping function."""
        return 0.0

    def get_onsite_pairing(self, i):
        """Default onsite energy function."""
        return 0.0
    
    def get_kinH(self, dnx=0, dny=0):
        """Construct the (dnx, dny) derivative of the 
        k-space normal state Hamiltonian function."""

        def H0(kx, ky):
            """
            Evaluate the hopping Hamiltonian at given kx, ky.
            This function essentially only serves to calculate the SFW.
            The hole sector doesn't have the minus sign for this reason!
            """

            H0k = np.zeros( ( self.n, self.n ), dtype=complex)
            H0kh = np.zeros( ( self.n, self.n ), dtype=complex)
            for site, nns in self.lat.nn.items():
                j = self.lat.map_indices[site]
                for nn in nns:
                    i = self.lat.map_indices[nn]
                    H0k[i,j] += self.get_hopping(i, j, np.asarray((kx, ky)), dnx=dnx, dny=dny)
                    H0kh[i,j] += -self.get_hopping(i, j, np.asarray((-kx, -ky)), dnx=dnx, dny=dny)
            
            H0k_reduced = np.block([[H0k, np.zeros_like(H0k)],
                                  [np.zeros_like(H0k), -np.conjugate(H0kh)]])
            
            return H0k_reduced

        return H0           


    def get_H0(self, dnx=0, dny=0):
        """Construct the (dnx, dny) derivative of the 
        k-space normal state Hamiltonian function."""

        def H0(kx, ky):
            """Evaluate the normal state Hamiltonian at given kx, ky."""

            H0k = np.zeros( ( self.n, self.n ), dtype=complex)
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
            zmat = np.zeros_like(H0kp, dtype=complex)
            H0kh = -np.conjugate(H0(-kx, -ky))  # hole sector
            HDk = np.zeros_like(H0kp, dtype=complex)           # pairing matrix

            for site, _ in self.lat.nn.items():
                j = self.lat.map_indices[site]
                HDk[j,j] += self.get_onsite_pairing(j)

            H0kp_spinfull = np.block([[H0kp, zmat],
                                  [zmat, H0kp]])
            H0kh_spinfull = np.block([[H0kh, zmat],
                                  [zmat, H0kh]])
            HDk_spinfull = np.block([[np.zeros_like(HDk), HDk],
                                  [-HDk, np.zeros_like(HDk)]])

            # Compute tensor product to form full BdG Hamiltonian
            HBdGk = np.block([[H0kp_spinfull, HDk_spinfull], 
                              [np.conjugate(HDk_spinfull.T), H0kh_spinfull]])

            return HBdGk
        
        return HBdG

    def get_reducedH(self, dnx=0, dny=0):
        """Construct the (dnx, dny) derivative of the 
        k-space BdG Hamiltonian function. 
        Returns only upspin particle and downspin hole part of BdG Hamiltonian"""

        H0 = self.get_H0(dnx=dnx, dny=dny)

        def HBdG(kx, ky): 
            """Evaluate the Hamiltonian at given kx, ky."""

            H0kp = H0(kx, ky)                   # particle sector
            H0kh = -np.conjugate(H0(-kx, -ky))  # hole sector
            HDk = np.zeros_like(H0kp)           # pairing matrix

            for site, _ in self.lat.nn.items():
                j = self.lat.map_indices[site]
                HDk[j,j] += self.get_onsite_pairing(j)

            # Compute tensor product to form full BdG Hamiltonian
            HBdGk = np.block([[H0kp, HDk], 
                              [np.conjugate(HDk), H0kh]])

            return HBdGk
        
        return HBdG      

### Model Initializations ###

def _init_square_base(self, N=1, **kwargs):
    """Base initialization for square-lattice-type models"""

    self.t = kwargs.get('t', 1.0)
    self.nu = kwargs.get('nu', 0.5)
    self.T = kwargs.get('T', 0.0)
    
    for param in ['mu', 'delta', 'ns', 'U']:
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(self.n)
        assert np.asarray(value).shape == (self.n,), f"Parameter {param} must be of shape (n,)"
        setattr(self, param, value)

    def get_onsite_energy(i):
        #idx = self.lat.map_sites[i]
        return -self.mu[i] - self.U[i]/2 * self.ns[i]
    setattr(self, 'get_onsite_energy', get_onsite_energy)

    def get_hopping(i, j, k, dnx=0, dny=0):
        N = self.lat.N
        k[0] /= N
        k[1] /= N
        site = self.lat.map_sites[j]
        nn = self.lat.map_sites[i]
        R = np.stack(self.lat.nn[site][nn]).T
        drkx, drky = 1j * self.lat.mapback(np.asarray(nn) - np.asarray(site))
        f0 =  np.exp(drkx * k[0] + drky * k[1])
        dRkx, dRky = 1j  * N * R
        farr = np.exp(dRkx * k[0] + dRky * k[1])
        if dnx == 0 and dny == 0:
            res = -self.t * f0 * np.sum(farr) 
        else:
            res = - 1/N*self.t * (drkx**dnx) * (drky**dny) * f0 * np.sum(farr) # f'(sublattice)*g(uc)
            res += - 1/N*self.t*np.sum((dRkx**dnx) * (dRky**dny) * farr) *f0   # f(sublattice)*g'(uc)
        return res
    setattr(self, 'get_hopping', get_hopping)
    
    def get_onsite_pairing(i):
        #idx = self.lat.map_sites[i]
        return np.abs(self.U[i])*self.delta[i]
    setattr(self, 'get_onsite_pairing', get_onsite_pairing)  

def _init_DSLmodel_base(self, N=1, **kwargs):
    """Base initialization for DSL-type models"""

    self.t = kwargs.get('t', 1.0)
    self.nu = kwargs.get('nu', 3.0)
    self.T = kwargs.get('T', 0.0)
    
    for param in ['mu', 'delta', 'ns', 'U']:
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(self.n)
        assert np.asarray(value).shape == (self.n,), f"Parameter {param} must be of shape (n,)"
        setattr(self, param, value)

    def get_onsite_energy(i):
        #idx = self.lat.map_sites[i]
        return -self.mu[i] - self.U[i]/2 * self.ns[i]
    setattr(self, 'get_onsite_energy', get_onsite_energy)

    def get_hopping(i, j, k, dnx=0, dny=0):
        N = self.lat.N
        k[0] /= N
        k[1] /= N
        site = self.lat.map_sites[j]
        nn = self.lat.map_sites[i]
        R = np.stack(self.lat.nn[site][nn]).T
        drkx, drky = 1j * self.lat.mapback(np.asarray(nn) - np.asarray(site))
        f0 =  np.exp(drkx * k[0] + drky * k[1])
        dRkx, dRky = 1j  * N * R
        farr = np.exp(dRkx * k[0] + dRky * k[1])
        if dnx == 0 and dny == 0:
            res = -self.t * f0 * np.sum(farr) 
        else:
            res = - 1/N*self.t * (drkx**dnx) * (drky**dny) * f0 * np.sum(farr) # f'(sublattice)*g(uc)
            res += - 1/N*self.t*np.sum((dRkx**dnx) * (dRky**dny) * farr) *f0   # f(sublattice)*g'(uc)
        return res
    setattr(self, 'get_hopping', get_hopping)

    def get_onsite_pairing(i):
        #idx = self.lat.map_sites[i]
        return np.abs(self.U[i])*self.delta[i]
    setattr(self, 'get_onsite_pairing', get_onsite_pairing)   

### Specific Models ###

class SquareLatticeModel(Model):
    def __init__(self, **kwargs):

        self.lat = lattice.SquareLattice()
        super().__init__(lat=self.lat)

        self.n = self.lat.N**2
        self.dim = 4 * self.n
        _init_square_base(self, 1, **kwargs)


class DSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.DiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        self.n = self.lat.N**2
        self.dim = 4 * self.n
        _init_DSLmodel_base(self, N, **kwargs)

class dDSLmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.dDiagonallyStripedLattice(N=N)
        super().__init__(lat=self.lat)

        self.n = self.lat.N**2 - self.lat.N + 1
        self.dim = 4 * self.n
        _init_DSLmodel_base(self, N, **kwargs)

class LiebNmodel(Model):
    def __init__(self, N=1, **kwargs):

        self.lat = lattice.LiebNLattice(N=N)
        super().__init__(lat=self.lat)

        self.n = 2 * self.lat.N - 1
        self.dim = 4 * self.n
        _init_square_base(self, N, **kwargs)





