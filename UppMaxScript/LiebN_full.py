import h5py

import numpy as np

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


U = 1
Tarr = np.linspace(0,0.5,50)
Narr = np.arange(2,10)

for Nv in Narr:
    cT = 0

    for Tv in Tarr:
        cT+=1
        N=Nv
        m = -U/2
        n = 2*N-1
        T=Tv
        lat = lattice.LiebNLattice(N=N)
        ham = hamiltonian.LiebNmodel(N=N, delta=np.ones(n), T=T, U=np.ones(n)*U, nu=n, ns=np.ones(n), mu=np.ones(n)*m)

        dels, ons, mus = sc_AHM.self_consistency_loop(ham, 41, HF=True, Nmax=300, alpha=0.3, g=1e-4)

        print(f'sc for N={N} is done')

        nk=41

        #SFW

        ten, Ds, term = sf.detSFW(ham,nk) #total SFW
        tens, dcsfw, termc = sf.det_convSFW(ham, nk) #conventional contribution to SFW

        print(f'SFW calculation is done for N={N}, T={T}')

        #HDF5 data storage
        name = f'LiebN_N{N}_n{n}_U{U}_T{cT}.hdf5'
        f = h5py.File(name, 'w')

        sc_gr = f.create_group("hamiltonian/sc_params")
        sc_gr.attrs['initial_dnm']= [1,1,m]
        sc_gr.attrs['nk_Nmax_alpha_g']= [41, 300, 0.3, 1e-4]

        dels_data = sc_gr.create_dataset("dels", data=dels)
        ons_data = sc_gr.create_dataset("ons", data=ons)
        mus_data = sc_gr.create_dataset("mus", data=mus)        

        sfw_gr = f.create_group("SFW")
        sfw_gr.attrs['nk']= nk
        sfw_gr.attrs['U']=U
        sfw_gr.attrs['T']=T
        sfw_gr.attrs['N']=N
        sfw_gr.attrs['n']=n

        sfw = sfw_gr.create_dataset("Ds_tot_conv", data=[Ds, dcsfw])
        sfw.attrs['names']=['sq(det(total SFWtens))', 'sq(det(convnetional SFWtens))']

        tot_sfwtens = sfw_gr.create_dataset("tot_sfwtens", data=ten)
        conv_sfwtens = sfw_gr.create_dataset("conv_sfwtens", data=tens)

        tot_sfwterms = sfw_gr.create_dataset("tot_sfwterms", data=term)
        conv_sfwterms = sfw_gr.create_dataset("conv_sfwterms", data=termc)

    print(f'N={N} completed \n\n---------------------------------------')

