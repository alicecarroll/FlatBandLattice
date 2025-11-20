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


def plot_bands(model, bdg=True, nk=200, hsp_path='GXM', ax=None, **kwargs):

    if bdg: H = model.get_HBdG()
    else: H = model.get_H0()

    k_points, hsp_indices = get_k_path(nk, hsp_path, model.lat.reciprocal_vecs)

    spectrum = np.empty((k_points.shape[0], H(0,0).shape[0]))

    for idx, kp in enumerate(k_points):
        eigenvalues = np.linalg.eigvalsh(H(*kp))
        spectrum[idx, :] = eigenvalues

    if ax is None: _, ax = plt.subplots(figsize=(6,6))
    for band in spectrum.T:
        ax.plot(range(k_points.shape[0]), band, color='black')
        
    # ax.set_xlabel("$(k_x a,k_y a)$", size='x-large')
    ylabel = kwargs.get('ylabel', 'Energy (t)')
    if ylabel: ax.set_ylabel(ylabel, size='x-large')
    ax.set_xticks(ticks=hsp_indices, labels=[hsp_labels[hsp] for hsp in hsp_path])
    ax.tick_params(labelsize='x-large')
    for i in hsp_indices:
        ax.axvline(i, c= 'grey', ls='--')

    return ax   
    
def plot_DOS(model, bdg=True, elim=(-1, 1), ne=200, nk=40, sig=5e-2, ax=None, **kwargs):

    if bdg: H = model.get_HBdG()
    else: H = model.get_H0()

    dos = np.zeros(ne)
    e_array = np.linspace(elim[0], elim[1], ne)

    k_points = get_k_grid(nk, model.lat.reciprocal_vecs)

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
    ax.set_xticks(())

    return ax

def plot_mean_fields(model, pvals, pname, nk=40, ax=None, kBT=.0):

    hvals, cvals = np.zeros_like(pvals), np.zeros_like(pvals)
    k_grid = get_k_grid(nk, model.lat.reciprocal_vecs)
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