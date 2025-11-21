import numpy as np

def fermidirac(E,T,o=0):
    
    nE=0
    if o==0:
        if np.abs(E)<1e-6 and T!=0:
            nE = 1
        elif T==0:
            if E>0:
                nE = 0
            else:
                nE = 1
        else:
            nE = 1/(1+np.exp(E/T))
    elif o==1:
        if np.abs(E)<1e-6 and T!=0:
            nE = -1/(4*T)
        elif np.abs(E)<1e-6 and T==0:
            nE = -1/4
        elif T==0:
            nE = 0
        else:
            nE = -1/((1+np.exp(E/T))**2)*np.exp(E/T)/T
        
    return nE

def SFW(model, nk, my= (1,0), ny=(1,0)):
    
    T=model.T
    
    gammaz = np.kron(np.diag([1,-1]), np.eye(int(model.site_num)))
    
    term_array = np.zeros((nk**2, 3,int((model.site_num*2)**2)))
    
    karr = np.linspace(0,2*np.pi, nk, endpoint=False)
    summe = 0
    counter =0

    for kx in karr:
        for ky in karr:
            pflist = []
            parli = []
            diali = []

            H = model.Hk(kx,ky, reduce=True)
            dHdmy = model.Hk(kx,ky, reduce=True, dnx=my[0], dny=my[1])
            dHdny = model.Hk(kx,ky, reduce=True, dnx=ny[0], dny=ny[1])
            
            M1 = np.matmul(dHdmy,gammaz)
            M2 = np.matmul(dHdny,gammaz)
            
            evals, evec = np.linalg.eigh(H)
            Evec = evec.T 

            nE = [fermidirac(E,T,o=0) for E in evals]
            dnE = [fermidirac(E,T,o=1) for E in evals]
            for k,i in enumerate(evals):
                for l,j in enumerate(evals):
                    if np.abs(i-j)<1e-6 or k==l:
                        pf = -dnE[l]
                    else:
                        pf = (nE[l]-nE[k])/(i-j)

                    if pf==0:
                        summe+=0
                        f1, f2, f3, f4 = (0,0,0,0)

                    else:
                        f1 = np.matmul(np.conjugate(Evec[k].T),np.matmul(dHdmy,Evec[l]))
                        f2 = np.matmul(np.conjugate(Evec[l].T),np.matmul(dHdny,Evec[k]))

                        f3 = np.matmul(np.conjugate(Evec[k].T),np.matmul(M1,Evec[l]))
                        f4 = np.matmul(np.conjugate(Evec[l].T),np.matmul(M2,Evec[k]))

                        s = pf*(f1*f2-f3*f4)

                        summe+=s

                    pflist.append(pf/nk)
                    diali.append(f1*f2/nk)
                    parli.append(f3*f4/nk)
                
            term_array[counter]= np.array([pflist, diali, parli])

            counter +=1
    
    return summe/nk**2, term_array

def detSFW(model, nk=80):
    xx = SFW(model, nk, my=(1,0), ny=(1,0))[0]
    xy = SFW(model, nk, my=(1,0), ny=(0,1))[0]
    yx = SFW(model, nk, my=(0,1), ny=(1,0))[0]
    yy = SFW(model, nk, my=(0,1), ny=(0,1))[0]
    ten = np.array([[xx,xy],[yx,yy]])

    return ten, np.sqrt(np.linalg.det(ten))