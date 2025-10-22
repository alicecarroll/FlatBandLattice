import matplotlib.pyplot as plt
import numpy as np


hsp_dict = {
    'G': (0, 0),
    'X': (np.pi, 0),
    'M': (np.pi, np.pi),
    'Y': (0, np.pi),
    'H': (2*np.pi, 0)
}

hsp_labels = {
    'G': '$\\Gamma$',
    'X': '$X$',
    'M': '$M$',
    'Y': '$Y$',
    'H': '$\\Gamma^\\prime$'
}

def get_k_path(nk, hsp_path):
    k_distances = []
    for i, hsp in enumerate(hsp_path[:-1]):
        k_start = np.array(hsp_dict[hsp])
        k_end = np.array(hsp_dict[hsp_path[i+1]])
        k_distances.append(np.linalg.norm(k_end - k_start))
    

    total_distance = sum(k_distances)
    segment_lengths = [int(nk * (d / total_distance)) for d in k_distances]

    k_path = []
    for i, hsp in enumerate(hsp_path[:-1]):
        k_start = np.array(hsp_dict[hsp])
        k_end = np.array(hsp_dict[hsp_path[i+1]])
        segment_length = segment_lengths[i]
        k_segment = np.linspace(k_start, k_end, segment_length, endpoint=False)
        k_path.extend(k_segment)
    k_path.append(hsp_dict[hsp_path[-1]])

    hsp_indices = np.cumsum([0] + segment_lengths)

    return np.array(k_path), hsp_indices


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

    k_lin = np.linspace(0, 2*np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(k_lin, k_lin)

    kx /= s[0]
    ky /= s[1]

    k_points = np.column_stack((kx.flatten(), ky.flatten()))

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