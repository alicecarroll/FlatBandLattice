from . import lattice
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from numba import jit, prange, int64, complex128, float64
from . import hamiltonian_opti
from . import sc_AHM
from importlib import reload
reload(hamiltonian_opti)


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
        self.T = None
        self.nu = None


    
    def prep_lat_params(self):
        s_idx = []
        n_idx = []
        sx = []
        sy = []
        nx = []
        ny = []
        R_flat = []
        R_ptr = [0]

        for site, nns in self.lat.nn.items():
            j = self.lat.map_indices[site]
            for nn in nns:
                i = self.lat.map_indices[nn]
                R = self.lat.nn[site][nn]

                for Rx, Ry in R:
                    R_flat.append((Rx, Ry))
                R_ptr.append(len(R_flat))

                s_idx.append(i)
                n_idx.append(j)
                sx.append(site[0])
                sy.append(site[1])
                nx.append(nn[0])
                ny.append(nn[1])
        
        return [
            np.array(s_idx, dtype=np.int64),
            np.array(n_idx, dtype=np.int64),
            np.array(sx, dtype=np.int64),
            np.array(sy, dtype=np.int64),
            np.array(nx, dtype=np.int64),
            np.array(ny, dtype=np.int64),
            np.array(R_ptr,   dtype=np.int64),
            np.array(R_flat,  dtype=np.int64),
        ]
    
    def prep_en_params(self):
        small = [self.n, self.lat.N, self.t, self.nu, self.T]
        big = np.array([self.U, self.ns, self.mu, self.delta], dtype=complex)
        return small, big
    
    def get_SFW(self):
        """Default hopping function."""
        return 0.0

    def get_H0(self, dnx=0, dny=0):
        """Get basic normal state Hamiltonian."""
        
        lparams = self.prep_lat_params()
        sparams, bparams = self.prep_en_params()
        U, ns, mu = bparams[:-1]

        def H0(kx, ky):
            H = np.zeros((self.n, self.n), dtype=complex)
            H_0 = hamiltonian_opti.H_0(H, kx, ky, dnx, dny, *lparams, *sparams, U, ns, mu)
            return H_0
        
        return H0
    
    def get_Hkin(self, dnx=0, dny=0):
        """Get basic normal state Hamiltonian."""
        
        lparams = self.prep_lat_params()
        sparams, bparams = self.prep_en_params()

        def H0(kx, ky):
            H = np.zeros((self.n, self.n), dtype=complex)
            H_0 = hamiltonian_opti.H_kin(H, kx, ky, dnx, dny, *lparams, *sparams[:-2])
            return H_0
        
        return H0
    
    def get_HBdG(self, dnx=0, dny=0):
        """Get BdG Hamiltonian."""
        lparams = self.prep_lat_params()
        sparams, bparams = self.prep_en_params()
        U, ns, mu, delta= bparams

        def H_B(kx, ky):
            H = np.zeros((self.n, self.n), dtype=complex)
            H_bdg = hamiltonian_opti.HBdG(H, kx, ky, dnx, dny, *lparams, *sparams, U, ns, mu, delta)
            return H_bdg[0]
        
        return H_B

    def get_Hreduced(self, dnx=0, dny=0):
        """Get BdG Hamiltonian for one spin orientation."""
        lparams = self.prep_lat_params()
        sparams, bparams = self.prep_en_params()
        U, ns, mu, delta= bparams

        def H_B(kx, ky):
            H = np.zeros((self.n, self.n), dtype=complex)
            H_bdg = hamiltonian_opti.HBdG(H, kx, ky, dnx, dny, *lparams, *sparams, U, ns, mu, delta)
            return H_bdg[1]
        
        return H_B
    
    def get_sc_params(self, g=1e-6, HF=True, Nmax=300, Nmin=10, alpha=0.3):
        """Get selfconsistent pairing strength delta, 
        occupation numbers n and on-site energy mu (if filling factor is defined)"""

        lparams = self.prep_lat_params()
        sparams, bparams = self.prep_en_params()
        U, ns, mu, delta= bparams
        karr = np.linspace(0,2*np.pi, 41,endpoint=False)

        dels, ons, mus = sc_AHM.self_consistency_loop(*lparams, *sparams, U, ns, mu, delta, 
                              karr, g, HF, Nmax, Nmin, alpha)
        
        self.delta = dels[:,-1]
        self.ns = ons[:,-1]
        self.mu = mus[:,-1]
        
        return dels, ons, mus

### Model Initializations ###

def _init_square_base(self, N=1, **kwargs):
    """Base initialization for square-lattice-type models"""

    self.t = kwargs.get('t', 1.0)
    self.nu = kwargs.get('nu', 3.0)
    self.T = kwargs.get('T', 0.0)
    
    for param in ['mu', 'delta', 'ns', 'U']:
        value = kwargs.get(param, None)
        if value is None: value = np.zeros(self.n)
        assert np.asarray(value).shape == (self.n,), f"Parameter {param} must be of shape (n,)"
        setattr(self, param, value)


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

### Specific Models ###

class SquareLatticeModel(Model):
    def __init__(self, **kwargs):

        self.lat = lattice.SquareLattice()
        super().__init__(lat=self.lat)

        self.n = 1
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
        self.N = self.lat.N
        self.dim = 4 * self.n
        
        _init_square_base(self, N, **kwargs)





