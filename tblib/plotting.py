import matplotlib.pyplot as plt
import numpy as np
from .geometry import get_k_grid, get_k_path
from .meanfield import get_mean_fields

hsp_labels = {
    'G': '$\\Gamma$',
    'X': '$X$',
    'M': '$M$',
    'Y': '$Y$',
    'H': '$\\Gamma^\\prime$'
}


def plot_bands(H, nk=200, hsp_path='GXM', ax=None, **kwargs):

    k_path, hsp_indices = get_k_path(nk, hsp_path)

    s = kwargs.get('s', (1,1))
    k_path[:,0] /= s[0]
    k_path[:,1] /= s[1]

    n = len(k_path)
    Hk = H(k_path[0][0], k_path[0][0])
    energies = np.zeros((n, Hk.shape[0]))
    for i in range(n):
        energies[i] = np.linalg.eigvalsh(H(*k_path[i]))

    if ax is None: _, ax = plt.subplots(figsize=(6,6))
    for band in energies.T:
        ax.plot(range(n), band, color='black')
        
    # ax.set_xlabel("$(k_x a,k_y a)$", size='x-large')
    ylabel = kwargs.get('ylabel', 'Energy (t)')
    if ylabel: ax.set_ylabel(ylabel, size='x-large')
    ax.set_xticks(ticks=hsp_indices, labels=[hsp_labels[hsp] for hsp in hsp_path])
    ax.tick_params(labelsize='x-large')
    for i in hsp_indices:
        ax.axvline(i, c= 'grey', ls='--')

    return ax   
    
def plot_DOS(H, s=(1,1), elim=(-1, 1), ne=200, nk=40, sig=5e-2, ax=None, **kwargs):

    dos = np.zeros(ne)
    e_array = np.linspace(elim[0], elim[1], ne)

    k_points = get_k_grid(nk, s=s)

    spectrum = np.empty((k_points.shape[0], H(0,0).shape[0]))

    for idx, kp in enumerate(k_points):
        eigenvalues = np.linalg.eigvalsh(H(*kp))
        spectrum[idx, :] = eigenvalues

    def Gaussian(E, En, sig):
        return np.exp(-(E-En)**2/(2*sig**2)) / np.sqrt(2*np.pi*sig**2)
    
    for i, e0 in enumerate(e_array):
        g_vec = np.vectorize(lambda e: Gaussian(e0, e, sig))
        dos[i] = np.sum(g_vec(spectrum)) / nk**2

    if ax is None: _, ax = plt.subplots(figsize=(6,6))

    ylabel = kwargs.get('ylabel', 'Energy (t)')
    xlabel = kwargs.get('xlabel', 'DOS')

    ax.plot(dos, e_array, color='black')
    if xlabel: ax.set_xlabel(xlabel, size='x-large')
    if ylabel: ax.set_ylabel(ylabel, size='x-large')
    ax.tick_params(labelsize='x-large')

    return ax

def plot_mean_fields(model, pvals, pname, s=(1,1), nk=40, ax=None, kBT=.0):

    hvals, cvals = np.zeros_like(pvals), np.zeros_like(pvals)
    k_grid = get_k_grid(nk, s=s)
    for i, p in enumerate(pvals):
        pvec = p * np.ones(model.nsites)
        setattr(model, pname, pvec)
        h, c = get_mean_fields(model, k_grid, kBT=kBT)
        hvals[i] = np.real( np.trace(h) )
        cvals[i] = np.real( np.trace(np.matmul(np.conjugate(c.T),c)) )

    if ax is None: _, ax = plt.subplots(figsize=(6,6))
    ax.plot(pvals, hvals, c='r')
    ax.plot(pvals, cvals, c='b')

    return ax


## temp

    # def solvHam(self, kx, ky, p='all'):
    #         '''
    #         solves model for each pair of coordinates
    #         '''

    #         eps = 1e-15
    #         n = np.shape(kx)[0]
    #         if p=='all':
    #             eig = np.zeros((n, self.dim))
    #         elif p=='part':
    #             eig = np.zeros((n, int(self.dim/4)))

        
    #         for i in range(n):
    #             if p == 'all':
    #                 e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[0]) #solves full BdG-Hamiltonian
    #             elif p == 'part':
    #                 e = np.linalg.eigvalsh(self.Hk(kx[i], ky[i])[1]) #solves only particle Hamiltonian
    #             #e[np.abs(e)<eps]=0
    #             eig[i]=np.sort(e)
                
    #         return eig.T
    
    # def plot_bands(self,k, p='all'):

    #     k1 = np.ones(100)
    #     k0 = np.zeros(100)
    #     path = np.concatenate((k, k, k*np.sqrt(2)))
    #     kx = np.concatenate((k,k[-1]*k1, k[::-1]))
    #     ky = np.concatenate((k0, k, k[::-1]))

    #     pl = [i for i in range(np.shape(path)[0])]
        
    #     energies = self.solvHam(kx, ky, p)

    #     emax = np.amax(energies)
    #     emax = emax+0.1*emax

    #     plt.figure(figsize=(8,6))
    #     plt.xlabel("$(k_x a,k_y a)$", size='x-large')
    #     plt.ylabel("E", size='x-large')
    #     plt.yticks(size='x-large')
    #     plt.xticks(ticks= [0, 100, 200, 299], labels=[r"$\Gamma$",r"X",r"M", r"$\Gamma$"], size='x-large')

    #     for i in energies:
    #         plt.plot(pl, i, color='black')

    #     plt.vlines([0, 99, 199, 299], [-emax, -emax, -emax, -emax], [emax, emax, emax, emax], colors= 'grey', linestyles='--')
    #     plt.show()
    #     return

    # def Es(self, k, p='part'):
    #     '''
    #     returns an array of the energies for each k-point in the BZ 
    #     for the normal state particle Hamiltonian (if p='part') or whole BdG-Hamiltonian (if p='all')
    #     Array should have dimension of (N_bands,N_kpoints**2)
        
    #     '''
    #     l = np.shape(k)[0]
    #     a1 = np.ones(l)
    #     if p=='all':
    #         E=np.empty((self.dim, l))
    #     elif p=='part':
    #         E=np.empty((int(self.dim/4), l))
    #     eps = 1e-15

    #     for i in k:
    #         Erow = self.solvHam(i*a1, k, p)
    #         E = np.concatenate((E, Erow), axis=1)
    #         E[np.abs(E)<eps]=0
    #     return E
    
    
    
    # def DOS(self, E, k, b=0, sig=5e-2, p='part'):
    #     ''' 
    #     returns an array with the density of state values to each E

    #     E is an array (against which DOS is plotted)
    #     En is an array with the E-values for all k grid points of shape (3, ???) because we currently have 3 bands
    #     sigma is the width of the gauss function
    #     '''
    #     def Gauss(E, En, sig):
    #         return np.exp(-(E-En)**2/(2*sig**2))
            
    #     En = self.Es(k, p)
    #     l = np.shape(En)[1]
    #     arr1 = np.ones(l)
    #     s1 = 0
    #     res = np.ones(np.shape(E)[0])
    #     c=0
    #     for j in E:
    #         s1 += np.sum(Gauss(j*arr1,En[b], sig))
    #         res[c] = s1
    #         s1=0
    #         c+=1
    #     return res