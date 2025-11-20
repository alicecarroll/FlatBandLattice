import numpy as np

hsp_dict = {
    'G': (0, 0),
    'X': (0.5, 0),
    'M': (0.5, 0.5),
    'Y': (0, 0.5),
    'H': (0.5, 0)
}

def get_reciprocal_vectors(latvecs):
    """
    Compute 2D reciprocal lattice vectors b1, b2 from
    real-space lattice vectors latvecs = (a1, a2),
    satisfying b_i · a_j = 2 pi delta_ij.
    """
    a1, a2 = (np.asarray(v, dtype=float) for v in latvecs)

    area = a1[0]*a2[1] - a1[1]*a2[0]
    if area == 0:
        raise ValueError("Lattice vectors are linearly dependent.")

    b1 = 2 * np.pi * np.array([ a2[1], -a2[0] ]) / area
    b2 = 2 * np.pi * np.array([-a1[1],  a1[0] ]) / area

    return b1, b2

def get_k_path(nk, hsp_path, reciprocal_vecs):
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
    k_path = np.matmul(k_path, reciprocal_vecs)

    hsp_indices = np.cumsum([0] + segment_lengths)

    return k_path, hsp_indices

def get_k_grid(nk, reciprocal_vecs):

    k_lin = np.linspace(0, 1, nk, endpoint=False)
    kx, ky = np.meshgrid(k_lin, k_lin)

    k_points = np.column_stack((kx.flatten(), ky.flatten()))
    k_points = np.matmul(k_points, reciprocal_vecs)

    return k_points