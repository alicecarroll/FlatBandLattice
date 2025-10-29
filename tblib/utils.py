import numpy as np

def diagnose(self):
    for key, val in self.lat.nn.items():
        print(f"Index {self.lat.map_indices[key]}: Site {key}: Neighbors {val}", flush=False)

def hermiticity_error(M: np.ndarray) -> float:
    """Frobenius norm of the anti-Hermitian part; zero means Hermitian."""
    return float(np.linalg.norm(M - M.conj().T, ord='fro'))

def quick_hermiticity_report(model, ks=((0.0, 0.0), (0.3, -0.7), (np.pi/3, np.pi/5))):
    """Print a quick hermiticity check for H0 and HBdG over a few k-points."""
    H0 = model.get_H0()
    HBdG = model.get_HBdG()
    for kx, ky in ks:
        Hk = H0(kx, ky)
        HBdGk = HBdG(kx, ky)
        print(
            f"k=({kx:.3f},{ky:.3f}) -> |H0-H0^†|_F={hermiticity_error(Hk):.2e}, "
            f"|HBdG-HBdG^†|_F={hermiticity_error(HBdGk):.2e}"
        )