import h5py
import numpy as np
from tblib import hamiltonian

U = 1
Tarr = np.linspace(0,0.2,30)
Narr = np.arange(9,15)[:1]

for Nv in Narr:
    cT = 0

    for Tv in Tarr:
        cT+=1
        N=Nv
        m = -U/2
        n = N**2-N+1
        T=Tv
        ham = hamiltonian.dDSLmodel(N=N, delta=np.ones(n), T=T, U=np.ones(n)*U, nu=n, ns=np.ones(n), mu=np.ones(n)*m)

        dels, ons, mus = ham.get_sc_params()

        print(f'sc for N={N} T={T} is done')

        #SFW
        dst, dsc, dsg, tent, tenc, teng = ham.get_SFW() #total SFW

        print(f'SFW calculation is done for N={N}, T={T}')

        #HDF5 data storage
        name = f'dDSL_N{N}_n{n}_U{U}_T{cT}.hdf5'
        f = h5py.File(name, 'w')

        attrdic = {'delta_initial': 1, 'on_initial':1, 'mu_initial':m, 'U': U, 'T': T, 'N':N, 'n':n, 'nk':41}
        for key, val in attrdic.items():
            f.attrs[key] = val
            
        sc_gr = f.create_group("sc_params")

        dels_data = sc_gr.create_dataset("dels", data=dels)
        ons_data = sc_gr.create_dataset("ons", data=ons)
        mus_data = sc_gr.create_dataset("mus", data=mus)        

        sfw_gr = f.create_group("SFW")

        sfw = sfw_gr.create_dataset("Ds", data=[dst, dsc, dsg])
        sfw.attrs['names']=['sq(det(total SFWtens))', 'sq(det(conventional SFWtens))', 'sq(det(geom SFWtens))']

        tot_sfwtens = sfw_gr.create_dataset("tot_sfwtens", data=tent)
        conv_sfwtens = sfw_gr.create_dataset("conv_sfwtens", data=tenc)
        geom_sfwtens = sfw_gr.create_dataset("geom_sfwtens", data=teng)

    print(f'N={N} completed \n\n---------------------------------------')
