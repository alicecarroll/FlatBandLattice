import numpy as np

def fermidirac(E,T,o=0):
    
    nE=0
    if o==0:
        if T>=1e-10:
            nE = 1/(1+np.exp(E/T))
        elif T<1e-10:
            if E>0:
                nE = 0
            elif np.abs(E)<1e-6:
                nE = 1/2
            else:
                nE = 1
        
    elif o==1:
        if T>=1e-10:
            nE = -1/((1+np.exp(E/T))**2)*np.exp(E/T)/T 
        elif T<1e-10:
            if np.abs(E)<1e-6:
                nE = -1/(4*T+1e-10)
            else:
                nE = 0     
    return nE

def SFW(model, nk=41, my= (1,0), ny=(1,0)):
    
    T=model.T
    
    gammaz = np.kron(np.diag([1,-1]), np.eye(int(model.n)))
    
    term_array = np.zeros((nk**2, 3,int((model.n*2)**2)), dtype=complex)
    
    karr = np.linspace(0,2*np.pi, nk, endpoint=False)
    summe = 0
    counter =0

    H = model.get_reducedH()
    Hdmy = model.get_kinH(dnx=my[0], dny=my[1])
    Hdny = model.get_kinH(dnx=ny[0], dny=ny[1])

    for ky in karr:
        for kx in karr:
            pflist = []
            parli = []
            diali = []

            Hk = H(kx,ky)
            dHkdmy = Hdmy(kx,ky)
            dHkdny = Hdny(kx,ky)
            
            M1 = np.matmul(dHkdmy,gammaz)
            M2 = np.matmul(dHkdny,gammaz)
            
            evals, evec = np.linalg.eigh(Hk)
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
                        f1 = np.matmul(np.conjugate(Evec[k].T),np.matmul(dHkdmy,Evec[l]))
                        f2 = np.matmul(np.conjugate(Evec[l].T),np.matmul(dHkdny,Evec[k]))

                        f3 = np.matmul(np.conjugate(Evec[k].T),np.matmul(M1,Evec[l]))
                        f4 = np.matmul(np.conjugate(Evec[l].T),np.matmul(M2,Evec[k]))

                        s = pf*(f1*f2-f3*f4)

                        summe+=s

                    pflist.append(pf/nk**2)
                    diali.append(f1*f2)
                    parli.append(f3*f4)
                
            term_array[counter]= np.array([pflist, diali, parli])

            counter +=1
    
    return summe/nk**2, term_array

def detSFW(model, nk=41):
    xx = SFW(model, nk, my=(1,0), ny=(1,0))[0]
    xy = SFW(model, nk, my=(1,0), ny=(0,1))[0]
    yx = SFW(model, nk, my=(0,1), ny=(1,0))[0]
    yy = SFW(model, nk, my=(0,1), ny=(0,1))[0]
    ten = np.array([[xx,xy],[yx,yy]])

    return ten, np.sqrt(np.linalg.det(ten))

def SFWconv(model, nk=41, dk=1e-6, my= (1,0), ny=(1,0)):
    
    T=model.T
    a=model.n

    karr = np.linspace(0,2*np.pi, nk, endpoint=False)
    summe = 0

    term_array = np.zeros((nk**2, 3,int((a)**2)), dtype=complex)
    counter =0

    HBdG = model.get_reducedH()
    kinH = model.get_kinH()
    
    for kx in karr:
        for ky in karr:
            pref = []
            upcurr = []
            downcurr = []

            H = HBdG(kx,ky)

            H_up = kinH(kx,ky)[:a,:a]
            H_down = -kinH(kx,ky)[a:,a:]
            dHdkmy = kinH(kx+my[0]*dk,ky+my[1]*dk)[:a,:a]
            dHdkny = -kinH(kx+ny[0]*dk,ky+ny[1]*dk)[a:,a:]
            
            evals, evec = np.linalg.eigh(H)
            evals_up, evec_up = np.linalg.eigh(H_up)
            evals_down, evec_down = np.linalg.eigh(H_down)
            
            Evec = evec.T 
            Evec_up = evec_up.T 
            Evec_down = evec_down.T 

            evalsdmy = (np.linalg.eigh(dHdkmy)[0]-evals_up)/dk
            evalsdny = (np.linalg.eigh(dHdkny)[0]-evals_down)/dk

            m_mat=np.block([[Evec_up, np.zeros((a,a))], 
                            [np.zeros((a,a)), Evec_down]])
            s_array = np.zeros((2*a,2*a), dtype=complex)
            for i in range(2*a):
                s_array[i]= np.linalg.solve(m_mat.T, Evec[i])

            nE = [fermidirac(E,T,o=0) for E in evals]
            dnE = [fermidirac(E,T,o=1) for E in evals]

            for m in range(a):
                for n in range(a):
                    Cnn=0
                    for k,i in enumerate(evals):
                        for l,j in enumerate(evals):
                            
                            if np.abs(i-j)<1e-6 or k==l:
                                pf = -dnE[l]
                            else:
                                pf = (nE[l]-nE[k])/(i-j)

                            if pf==0:
                                Cnn+=0

                            else:
                                
                                s_l = s_array[l]
                                s_k = s_array[k]
                                w1 = np.conjugate(s_l[m])
                                w2 = s_k[m]
                                w3 = np.conjugate(s_k[n+a])
                                w4 = s_l[n+a]

                                Cnn+=4*pf*w1*w2*w3*w4
                    
                    upc = evalsdmy[m]
                    downc = evalsdny[n]
                    summe+=Cnn/(nk**2)*upc*downc

                            
                    pref.append(Cnn/(nk**2))
                    upcurr.append(upc)
                    downcurr.append(downc)

            term_array[counter]= np.array([pref, upcurr, downcurr])
            counter+=1

    return summe, term_array

def det_convSFW(model, nk=41):
    xx = SFWconv(model, nk, my=(1,0), ny=(1,0))[0]
    xy = SFWconv(model, nk, my=(1,0), ny=(0,1))[0]
    yx = SFWconv(model, nk, my=(0,1), ny=(1,0))[0]
    yy = SFWconv(model, nk, my=(0,1), ny=(0,1))[0]
    ten = np.array([[xx,xy],[yx,yy]])

    return ten, np.sqrt(np.linalg.det(ten))