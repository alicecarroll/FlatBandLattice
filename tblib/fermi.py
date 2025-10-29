import numpy as np

def fermi_dirac(energy, temperature, tol=1e-10):
    """Robust Fermi-Dirac distribution function."""
    # ensure arrays, at least 1D
    E = np.atleast_1d(energy)
    T = np.atleast_1d(temperature)

    # broadcast for all combinations if both are 1D
    if E.ndim == 1 and T.ndim == 1 and E.size > 1 and T.size > 1:
        E = E[:, None]
        T = T[None, :]

    T_zero = np.abs(T) < tol
    x = E / T

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        f = np.where(x > 0,
            np.exp(-x) / (1 + np.exp(-x)),
            1 / (1 + np.exp(x)) )

    f = np.where(T_zero, (E < 0).astype(float), f)
    f = np.nan_to_num(f)# handle T=0 limit

    if np.isscalar(energy) and np.isscalar(temperature): 
        return float(f)
    return f


def fermi_dirac_prime(energy, temperature, tol=1e-10):
    """Robust derivative of the Fermi-Dirac distribution function."""
    E = np.atleast_1d(energy)
    T = np.atleast_1d(temperature)

    if np.any(np.abs(T) < tol):
        raise UserWarning("Derivative undefined at T=0.")

    if E.ndim == 1 and T.ndim == 1 and E.size > 1 and T.size > 1:
        E = E[:, None]
        T = T[None, :]

    f = fermi_dirac(E, T, tol)
    df = -f * (1 - f) / T
    df = np.nan_to_num(df)

    if np.isscalar(energy) and np.isscalar(temperature):
        return float(df)
    return df