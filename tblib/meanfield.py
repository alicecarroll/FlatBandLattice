import numpy as np
from .fermi import fermi_dirac

def cooper(umk, vmk, upk, vpk, evals, kBT=.0):
    # if kBT < 1e-10: return np.matmul(umk.T, vpk)
    f = np.diag(fermi_dirac(evals, kBT, tol=1e-10))
    res = np.matmul(umk.T,np.matmul(f, vpk))
    return res

def hartree(umk, vmk, upk, vpk, evals, kBT=.0):
    # if kBT < 1e-10: return np.matmul(np.conjugate(v.T), v)
    f = np.diag(fermi_dirac(evals, kBT, tol=1e-10))
    res = np.matmul(np.conjugate(vpk.T),np.matmul(f, vpk))
    res += np.matmul(np.conjugate(vpk.T),np.matmul(f, vpk))
    return res

def get_mean_fields(model, k_grid, kBT=.0):
    """Compute Cooper and Hartree mean-field channels once."""

    assert kBT >= 0, ValueError(f"Negative Temperature: kBT={kBT}")

    sdim = int( 2 * model.nsites ) # 1 spin per orbital
    PairingMatrix = np.zeros((sdim,sdim), dtype=complex)
    OccupationMatrix = np.zeros((sdim,sdim), dtype=complex)

    # Get updated Hamiltonian
    HBdG = model.get_HBdG()

    for kx, ky in k_grid:
        evals, evecs = np.linalg.eigh(HBdG(kx, ky))

        evals = np.flip(evals)
        evecs = np.flip(evecs, axis=1)

        upk = evecs[sdim:, :sdim]
        vpk = evecs[:sdim, :sdim]
        vmk = evecs[sdim:, sdim:]
        umk = evecs[:sdim, sdim:]

        epos = evals[:sdim]

        c = cooper(umk, vmk, upk, vpk, epos, kBT=kBT)
        h = hartree(umk, vmk, upk, vpk, epos, kBT=kBT)

        PairingMatrix += c
        OccupationMatrix += h
        
    nk = len(k_grid)
    PairingMatrix /= nk
    OccupationMatrix /= nk

    # OccArr = np.diag(OccupationMatrix) / nk
    # Delta = [-np.abs(model.U[i]) / nk * PairingMatrix[i,int(i+sdim/2)] for i in range(int(sdim/2))]
    # Nu = [OccArr[i]+OccArr[int(i+sdim/2)] for i in range(int(sdim/2))]

    # Delta = np.array([Delta[model.lat.map_idx[(i,0)]] for i in range(model.lat.N)])
    # Nu = np.array([Nu[model.lat.map_idx[(i,0)]] for i in range(model.lat.N)])

    return OccupationMatrix, PairingMatrix
