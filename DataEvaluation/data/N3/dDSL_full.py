import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from importlib import reload
from tblib import lattice
from tblib import hamiltonian
from tblib import sc_AHM
from tblib import plotting
from tblib import superfluid as sf

reload(lattice)
reload(hamiltonian)
reload(sf)
reload(sc_AHM)

Nv = 3
Uarr = np.linspace(0.00001, 7, 20)
Tarr = np.linspace(0,0.3*Nv,50)

cU =0
for Uv in Uarr:
    cU+=1
    cT = 0

    for Tv in Tarr:
        cT+=1
        N=Nv
        m = -Uv/2
        n = N**2-N+1
        U=Uv
        T=Tv
        lat = lattice.dDiagonallyStripedLattice(N=N)
        ham = hamiltonian.dDSLmodel(N=N, delta=np.ones(n), T=T, U=np.ones(n)*U, nu=n, ns=np.ones(n), mu=np.ones(n)*m)

        dels, ons, mus = sc_AHM.self_consistency_loop(ham, 41, HF=True, Nmax=300, alpha=0.3, g=1e-4)

        nk=41

        H = ham.get_HBdG()
        k=np.linspace(0,2*np.pi,nk)
        E2 = np.zeros((ham.dim,nk*nk))
        c = 0
        for kx in k:
            for ky in k:
                E2[:,c] = np.linalg.eigh(H(kx,ky))[0]
                c+=1

        s1=np.shape(E2)[0]
        E2=E2.reshape(ham.dim,nk,nk)

        #SFW

        ten, Ds, term = sf.detSFW(ham,nk) #total SFW
        tens, dcsfw, termc = sf.det_convSFW(ham, nk) #conventional contribution to SFW


        #HDF5 data storage
        name = f'dDSL_N{N}_n{n}_U{cU}_T{cT}.hdf5'
        f = h5py.File(name, 'w')

        sc_gr = f.create_group("hamiltonian/sc_params")
        sc_gr.attrs['initial_dnm']= [1,1,m]
        sc_gr.attrs['nk_Nmax_alpha_g']= [nk, 300, 0.3, 1e-4]

        dels_data = sc_gr.create_dataset("dels", data=dels)
        ons_data = sc_gr.create_dataset("ons", data=ons)
        mus_data = sc_gr.create_dataset("mus", data=mus)

        E_data = f.create_dataset("hamiltonian/Es", data=E2)
        E_data.attrs['nk']=nk
        E_data.attrs['U']=U
        E_data.attrs['T']=T
        E_data.attrs['N']=N
        E_data.attrs['n']=n


        sfw_gr = f.create_group("SFW")
        sfw_gr.attrs['nk']= nk

        sfw = sfw_gr.create_dataset("Ds_tot_conv", data=[Ds, dcsfw])
        sfw.attrs['names']=['sq(det(total SFWtens))', 'sq(det(convnetional SFWtens))']

        tot_sfwtens = sfw_gr.create_dataset("tot_sfwtens", data=ten)
        conv_sfwtens = sfw_gr.create_dataset("conv_sfwtens", data=tens)

        tot_sfwterms = sfw_gr.create_dataset("tot_sfwterms", data=term)
        conv_sfwterms = sfw_gr.create_dataset("conv_sfwterms", data=termc)
