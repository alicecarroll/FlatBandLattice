import matplotlib.pyplot as plt

class Lattice:
    def __init__(self, sites):
        
        self.sites = {tuple(s) for s in sites}
        self.nn = {site: {} for site in self.sites}
        self.nn_dir = {site: {} for site in self.sites} # For later

    def plot_lattice(self, ax=None):
        if ax is None: fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for site in self.sites:
            ax.scatter(site[0], site[1], c='k')
        return fig, ax
    
    def plot_nn(self, ax=None):
        if ax is None: fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for site in self.sites:
            ax.scatter(site[0], site[1], c='k')
      
        for site in self.nn:
            for R in self.nn[site]:

                for nn in self.nn[site][R]:
                    if R == (0,0):
                        x = [site[0], nn[0]]
                        y = [site[1], nn[1]]
                        ax.plot(x, y, c='blue')
                    else:
                        x1 = [site[0], site[0]+R[0]/2]
                        y1 = [site[1], site[1]+R[1]/2]
                        x2 = [nn[0], nn[0]-R[0]/2]
                        y2 = [nn[1], nn[1]-R[1]/2]
                        ax.plot(x1, y1, c='blue')
                        ax.plot(x2, y2, c='blue')
        return fig, ax


class SquareLattice(Lattice):
    def __init__(self):
        sites = {(0, 0)}
        super().__init__(sites)
        
        self.nn[(0, 0)] = {(0, 0): [(1, 0), (0, 1), (-1, 0), (0, -1)]}

class DiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N

        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]
        
        super().__init__(sites)

        nn_templates = [(1,0), (0,1), (-1,0), (0,-1)]

        for site in self.sites:
            nxi, nyi = site

            for dx, dy in nn_templates:

                Rxf, nxf = divmod(nxi + dx, self.N)
                Ryf, nyf = divmod(nyi + dy, self.N)
                if (nxf, nyf) in sites:
                    if (Rxf, Ryf) in self.nn[site]:
                        self.nn[site][(Rxf, Ryf)].append( (nxf, nyf) )
                    else: 
                        self.nn[site][(Rxf, Ryf)] = [(nxf, nyf),]

class dDiagonallyStripedLattice(Lattice):
    def __init__(self, N=1):

        self.N = N

        sites = [(nx, ny) for nx in range(self.N) for ny in range(self.N)]
        for site in sites[1:]:
            if site[0]==site[1]:
                sites.remove(site)

        super().__init__(sites)

        nn_templates = [(1,0), (0,1), (-1,0), (0,-1)]
        #self.sites.pop('(2,2)')
        for site in self.sites:
            nxi, nyi = site

            for dx, dy in nn_templates:

                Rxf, nxf = divmod(nxi + dx, self.N)
                Ryf, nyf = divmod(nyi + dy, self.N)
                if (nxf, nyf) in sites:
                    if (Rxf, Ryf) in self.nn[site]:
                        self.nn[site][(Rxf, Ryf)].append( (nxf, nyf) )
                    else: 
                        self.nn[site][(Rxf, Ryf)] = [(nxf, nyf),]