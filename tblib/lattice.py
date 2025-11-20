import matplotlib.pyplot as plt
import numpy as np
from .geometry import get_reciprocal_vectors

class Lattice:
    def __init__(self, sites, latvecs):
        
        self.sites = list(set({tuple(s) for s in sites}))
        self.nn = {site: {} for site in self.sites}
        self.nn_dir = {site: {} for site in self.sites} # For later

        # Lattice vectors
        self.a1 = latvecs[0]
        self.a2 = latvecs[1]
        self.lattice_vecs = np.stack(latvecs)
        self.reciprocal_vecs = np.stack(get_reciprocal_vectors(latvecs))

    @property
    def map_sites(self):
        return {i: site for i, site in enumerate(self.sites)}
    
    @property
    def map_indices(self):
        return {site: i for i, site in enumerate(self.sites)}

    def plot_lattice(self, ax=None, field=None, cmap='viridis'):
        if ax is None: _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        sites = np.array(self.sites)
        if field is not None: assert len(field) == len(sites)
        ax.scatter(*sites.T, c='k' if field is None else field, cmap=cmap, vmin=0, vmax=1)

        x0 = (-0.5, -0.5)
        ax.annotate('', x0, tuple(np.add(x0, self.a1)), 
                    va='center', ha='left', arrowprops=dict(arrowstyle='<|-'))
        ax.annotate('', x0, tuple(np.add(x0, self.a2)), 
                    va='bottom', ha='center', arrowprops=dict(arrowstyle='<|-'))
        # ax.annotate('$\\mathbf{a}_1$', x0, va='bottom', ha='right')
        # ax.annotate('$\\mathbf{a}_2$', x0, va='top', ha='left')
        
        # from matplotlib.patches import FancyArrow
        
        # arrow1 = FancyArrow(*x0, *self.a1, color='k', width=.1, length_includes_head=True)
        # arrow2 = FancyArrow(*x0, *self.a2, color='k', width=.1, length_includes_head=True)
        
        # ax.add_patch(arrow1)
        # ax.add_patch(arrow2)
    
        return ax
    
    def plot_nn(self, ax=None, field=None):

        ax = self.plot_lattice(ax=ax, field=field)
      
        for site in self.nn:
            for nn in self.nn[site]:

                for R in self.nn[site][nn]:
                    if R == (0,0):
                        x = [site[0], nn[0]]
                        y = [site[1], nn[1]]
                        ax.plot(x, y, c='blue', zorder=-1)
                    else:
                        x1 = [site[0], site[0]+R[0]/2]
                        y1 = [site[1], site[1]+R[1]/2]
                        x2 = [nn[0], nn[0]-R[0]/2]
                        y2 = [nn[1], nn[1]-R[1]/2]
                        ax.plot(x1, y1, c='blue', zorder=-1)
                        ax.plot(x2, y2, c='blue', zorder=-1)
        return ax
    
    @property
    def nsites(self): return len(self.sites)

def _init_square_base(self):

    nn_templates = [(1,0), (0,1), (-1,0), (0,-1)]

    for site in self.sites:
        nxi, nyi = site

        for dx, dy in nn_templates:

            Rxf, nxf = divmod(nxi + dx, self.N)
            Ryf, nyf = divmod(nyi + dy, self.N)
            if (nxf, nyf) in self.sites:
                if (nxf, nyf) in self.nn[site]:
                    self.nn[site][(nxf, nyf)].append( (Rxf, Ryf) )
                else: 
                    self.nn[site][(nxf, nyf)] = [(Rxf, Ryf),]

class SquareLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]
        a1, a2 = (N, 0), (0, N)

        super().__init__(sites, (a1, a2))
        _init_square_base(self)

class DiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]

        a1, a2 = (N, 0), (0, N)
        
        super().__init__(sites, (a1, a2))
        _init_square_base(self)

class dDiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N) if (nx!=ny or nx==0)]

        a1, a2 = (N, 0), (0, N)

        super().__init__(sites, (a1, a2))
        _init_square_base(self)

class LiebNLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(0, n) for n in range(self.N)]
        sites += [(n, 0) for n in range(1, self.N)]

        a1, a2 = (N, 0), (0, N)

        super().__init__(sites, (a1, a2))

        _init_square_base(self)