import numpy as np
    
def monkhorst_pack(self,nk, shift=None):
    """
    Generate a Monkhorst–Pack k-point grid.

    Parameters
    ----------
    nk : tuple/list of 3 ints
        Number of divisions along each reciprocal direction (N1, N2, N3).
    shift : tuple/list of 3 floats, optional
        Shift applied to the grid (in fractional coordinates). 
        Default is the standard Monkhorst-Pack shift: (0, 0, 0).

    Returns
    -------
    kpoints : ndarray of shape (N1*N2*N3, 3)
        Fractional k-point coordinates.
    """
    nk = np.asarray(nk, dtype=int)

    if shift is None:
        shift = np.zeros(2)
    else:
        shift = np.asarray(shift, dtype=float)

    # Generate coordinates along each direction
    grids = [(2 * np.arange(1, n + 1) - n - 1) / (2.0 * n) for n in nk]

    # Create full 3D grid
    kmesh = np.array(np.meshgrid(*grids, indexing='ij')).reshape(2, -1).T

    # Apply shift
    kmesh = (kmesh + shift) % 1.0

    return kmesh