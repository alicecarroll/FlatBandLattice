import numpy as np

hsp_dict = {
    'G': (0, 0),
    'X': (np.pi, 0),
    'M': (np.pi, np.pi),
    'Y': (0, np.pi),
    'H': (2*np.pi, 0)
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

def get_k_grid(nk, s=(1,1)):

    k_lin = np.linspace(0, 2*np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(k_lin, k_lin)

    kx /= s[0]
    ky /= s[1]

    k_points = np.column_stack((kx.flatten(), ky.flatten()))

    return k_points