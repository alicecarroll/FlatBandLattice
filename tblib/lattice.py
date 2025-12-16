import matplotlib.pyplot as plt
import numpy as np

class Lattice:
    def __init__(self, sites):
        
        #self.sites = {tuple(s) for s in sites}
        self.sites = list(set({tuple(s) for s in sites}))

        self.nn = {site: {} for site in self.sites}
        self.nn_dir = {site: {} for site in self.sites} 

        self.map_sites = {i: site for i, site in enumerate(self.sites)}
        self.map_indices = {site: i for i, site in enumerate(self.sites)}

        self.mapback = lambda s: s # Default: identity mapping

    def update_maps(self):
        self.map_sites = {i: site for i, site in enumerate(self.sites)}
        self.map_indices = {site: i for i, site in enumerate(self.sites)}

    def plot_lattice(self, ax=None, field=None, cmap='viridis'):
        if ax is None: _, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        sites = np.array(self.sites)
        if field is not None: assert len(field) == len(sites)
        ax.scatter(*sites.T, c='k' if field is None else field, cmap=cmap, vmin=0, vmax=2)
        return ax
        #for site in self.sites:
        #    ax.scatter(site[0], site[1], c='k')
        #return ax
    
    def plot_nn(self, ax=None, field=None, cmap='viridis'):

        ax = self.plot_lattice(ax=ax, field=field, cmap=cmap)

        #if ax is None: _, ax = plt.subplots()
        #ax.set_aspect('equal')
        #ax.set_xticks([])
        #ax.set_yticks([])
        #for site in self.sites:
        #    ax.scatter(site[0], site[1], c='k')
      
        for site in self.nn:
            for nn in self.nn[site]:

                for R in self.nn[site][nn]:
                    if R == (0,0):
                        x = [site[0], nn[0]]
                        y = [site[1], nn[1]]
                        ax.plot(x, y, c='black', zorder=-1)
                    else:
                        x1 = [site[0], site[0]+R[0]/2]
                        y1 = [site[1], site[1]+R[1]/2]
                        x2 = [nn[0], nn[0]-R[0]/2]
                        y2 = [nn[1], nn[1]-R[1]/2]
                        ax.plot(x1, y1, c='black', zorder=-1)
                        ax.plot(x2, y2, c='black', zorder=-1)
        return ax

    def striped_props(self, props):
        
        arblist = []
        as_d = {i: [site for site in self.nn if site in [((i+j)%self.N, j) for j in range(self.N)]] for i in range(self.N)} # atomic species dictionary associate each site to its group of stripes

        for i in self.map_sites:
            site = self.map_sites[i]
            num = [key for key,val in as_d.items() if site in val]
            
            arblist.append(props[num[0]])
            
        return arblist

class SquareLattice(Lattice):
    def __init__(self):
        sites = {(0, 0)}
        super().__init__(sites)
        
        self.nn[(0, 0)] = {(0, 0): [(1, 0), (0, 1), (-1, 0), (0, -1)]}
        self.update_maps()

    
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

    # Use the raw displacement between sites (no modulo) so that forward and reverse
    # hops are exact negatives of each other. This ensures k-space phases are
    # complex conjugates and the Hamiltonian remains Hermitian.
    self.mapback = lambda s: s
    
    self.update_maps()

def _init_DSL_base(self):

    _init_square_base(self)
    setattr(self, 'map_diag', 
        {self.map_indices[site]: (site[0]+site[1])%self.N for site in self.sites})


class DiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]
        
        super().__init__(sites)
        _init_DSL_base(self)

        #nn_templates = [(1,0), (0,1), (-1,0), (0,-1)]
#
        #for site in self.sites:
        #    nxi, nyi = site
#
        #    for dx, dy in nn_templates:
#
        #        Rxf, nxf = divmod(nxi + dx, self.N)
        #        Ryf, nyf = divmod(nyi + dy, self.N)
        #        if (nxf, nyf) in sites:
        #            if (nxf, nyf) in self.nn[site]:
        #                self.nn[site][(nxf, nyf)].append( (Rxf, Ryf) )
        #            else: 
        #                self.nn[site][(nxf, nyf)] = [(Rxf, Ryf),]

class dDiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N) if (nx!=ny or nx==0)]

        super().__init__(sites)
        _init_DSL_base(self)
        

        #sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]
        #for site in sites[1:]:
        #    if site[0]==site[1]:
        #        sites.remove(site)


        #nn_templates = [(1,0), (0,1), (-1,0), (0,-1)]
        #self.sites.pop('(2,2)')
        #for site in self.sites:
        #    nxi, nyi = site
#
        #    for dx, dy in nn_templates:
#
        #        Rxf, nxf = divmod(nxi + dx, self.N)
        #        Ryf, nyf = divmod(nyi + dy, self.N)
        #        if (nxf, nyf) in sites:
        #            if (nxf, nyf) in self.nn[site]:
         #               self.nn[site][(nxf, nyf)].append( (Rxf, Ryf) )
        #            else: 
        #                self.nn[site][(nxf, nyf)] = [(Rxf, Ryf),]
    
class LiebNLattice(Lattice):
    def __init__(self, N=1):

        self.N = N
        sites = [(0, n) for n in range(self.N)]
        sites += [(n, 0) for n in range(1, self.N)]

        super().__init__(sites)

        _init_square_base(self)