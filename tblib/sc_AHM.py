import numpy as np

def cooper(u,v,ubar,vbar,evals,T=0):
    if np.abs(T)<1e-10:#
        return np.matmul(ubar.T,np.conjugate(vbar))
    else:
        el=np.matmul(ubar.T,np.matmul(np.diag(1/(1+np.exp(-evals/T))), np.conjugate(vbar)))
        el+=np.matmul(v.T,np.matmul(np.diag(1/(1+np.exp(evals/T))),np.conjugate(u)))
        return el
    
def hatree(u,v,ubar,vbar,evals,T=0):
    if np.abs(T)<1e-10:
        return np.matmul(vbar.T, np.conjugate(vbar))
    else:
        el = np.matmul(vbar.T,np.matmul(np.diag(1/(1+np.exp(-evals/T))), np.conjugate(vbar)))
        el += np.matmul(u.T,np.matmul(np.diag(1/(1+np.exp(evals/T))), np.conjugate(u)))
        return el 

def get_mean_fields(model, nk, HF=True):
    
    karr = np.linspace(0, 2*np.pi, nk, endpoint=False)
    a = int(model.n*2)
    N = model.lat.N            

    Pairing=np.zeros((a,a), dtype=object)
    Occupation=np.zeros((a,a), dtype=object)

    HBdG = model.get_HBdG()
    T = model.T
    c=0
    for x in karr:
        for y in karr:
            c+=1

            evals1, Evec = np.linalg.eigh(HBdG(x, y))
            Evec = Evec.T
            Evec2 = np.flip(Evec, axis=0)

            evals = np.flip(evals1)[0:a]

            u = Evec2[:a, :a]
            v = Evec2[:a, a:]
            vbar = Evec[:a, :a] #this is conjugate(v-k)
            ubar = Evec[:a, a:]  #this is conjugate(u-k)
            
            Pairing+=cooper(u,v,ubar,vbar,evals,T)
            if HF:
                Occupation += hatree(u,v,ubar,vbar,evals,T)         
    
    Dmat = [-Pairing[int(i+a/2),i]/nk**2 for i in range(int(a/2))]
    Nmat = np.diag(Occupation)/nk**2
    final_N = [Nmat[i]+Nmat[int(i+a/2)] for i in range(int(a/2))]

    deltas = [Dmat[model.lat.map_indices[(i,0)]] for i in range(N)]
    ns = [final_N[model.lat.map_indices[(i,0)]] for i in range(N)]

    return deltas,ns
    
def self_consistency_loop(model, nk=40, T=0, g=1e-4, HF=True, Nmax=100, Nmin=10, alpha=0.3):
    
    N = model.lat.N

    delarr = np.array(model.delta)
    narr = np.array(model.ns)
    muarr = np.array(model.mu)

    dels = delarr.reshape(N,1)
    ns = narr.reshape(N,1)
    mus = muarr.reshape(N,1)

    c=0

    limit1 = False
    limit2 = False
    
    while (c<Nmax and (limit1 or limit2)) or c<Nmin:
        c+=1

        Vals = get_mean_fields(model, nk, HF)

        delarro = delarr
        narro = narr

        delta = Vals[0]

        delarr = np.array(delta)
        delarr = alpha*delarro+(1-alpha)*delarr

        model.delta = delarr

        if HF:
            nv = Vals[1]
            
            narr = np.array(nv)
            narr = alpha*narro+(1-alpha)*narr

            model.ns = narr

        dels = np.concatenate((dels, delarr.reshape(N,1)), axis=1)
        ns = np.concatenate((ns, narr.reshape(N,1)), axis=1)

        limit1 = (np.std(np.abs(dels[:,-3:]), axis=1)>g).any()
        if HF:
            limit2 = (np.std(np.abs(ns[:,-3:]), axis=1)>g).any()

        if model.nu!=0:
            muarro = muarr

            en=0
            H = model.get_HBdG()(0,0)
            for i in range(int(model.n)):
                en += H[i,i]

            mun = 1/(model.n)*(model.U[0]/2*(model.nu-model.n*2)+en)

            muarr = np.array([mun for i in range(N)])           
            muarr = alpha*muarro+(1-alpha)*muarr

            model.mu = muarr
        mus = np.concatenate((mus, muarr.reshape(N,1)), axis=1)
                    
    return dels, ns, mus
    