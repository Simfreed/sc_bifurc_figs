import numpy as np

############################################
#### Functions to run the grn simulation ###
############################################

def rplus(u, m1, m2, h1, h2):
    return np.array([np.divide(m1, 1+u[1]**h1, dtype = float), np.divide(m2, 1+u[0]**h2, dtype = float)])

def rminus(u, tau):
    return np.array([1.*np.divide(u[0], tau, dtype = float), np.divide(u[1], tau, dtype = float)])

def step(u, dt, m1, m2, h1, h2, tau, nExp, scale):

    # rplus = birth
    # rminus = death
    # both processes are Poisson ==> variance = mean
    # noise term therefore includes sum of both variances
    
    rpluss = rplus(u, m1, m2, h1, h2)
    rmins  = rminus(u, tau)
    mean   = u + ( rpluss - rmins )*dt
    sig    = np.sqrt(dt * ( rpluss + rmins ) / scale, dtype = float)
    
    return np.random.normal(mean, sig, np.array([2,nExp]) )

def rplusSlave(x, a, b, k, h):

    xok_h = (x / k)**h
    return (b*(a * xok_h + 1 - a)/(1+ xok_h)).T

def rminusSlave(v, tau):
    return (v.T/tau).T

def stepSlave(u, v, didxs, a, b, k, h, tau, dt, ngenes, ncells, scale):
    
#    print((u.shape,didxs.shape,u[didxs].shape,h.shape))

    rpluss = rplusSlave(u[didxs].T, a, b, k, h) 
    rmins  = rminusSlave(v, tau) 
    mean   = v + ( rpluss - rmins ) * dt
    sig    = np.sqrt(dt * ( rpluss + rmins ) / scale, dtype = float)

    return np.random.normal(mean, sig, np.array([ngenes, ncells]))

def get_yss(m1, m2, tau, just_nodes = True): #assumes h1 = h2 = 2
    
    coeffs       = [1, -tau*m2, 2, -2*tau*m2, 1+tau*tau*m1*m1, -tau*m2]
    
    roots        = np.roots(coeffs)
    re_root_idxs = np.where(np.imag(roots)==0)[0]
    re_roots     = np.real(roots[re_root_idxs])
    
    if len(re_root_idxs) == 1 or not just_nodes:
        return re_roots
    else:
        return np.array([np.amin(re_roots),np.amax(re_roots)])

def langevin(m1, m2, h1, h2, tau, driver_idxs, alphas, betas, ks, hs, vtaus, 
        scale, ncells, dt, tmax, nc_save = 10, dt_save = 1, yrng = [0,4], 
        node_pick = 'max'):
   
    nslaves = alphas.shape[0]
    ngenes = nslaves+2
    
    if node_pick == 'rand':
        y = np.random.uniform(*yrng, size = ncells)
    else:
        yss   = get_yss(m1, m2, tau) # start from the minimum
    
        if len(yss) == 0:
            y = np.random.uniform(*yrng, size = ncells)
        elif len(yss) == 1:
            y = yss[0]*np.ones(ncells)
        else:
            if node_pick == 'min':
                y = np.amin(yss)*np.ones(ncells)
            elif node_pick == 'max':
                y = np.amax(yss)*np.ones(ncells)
            else:
                y = np.random.choice(yss, size=ncells)
	
    x = m1*tau/(1+y**h1)
    
    u =  np.array([x,y])
    v = (rplusSlave(u[driver_idxs].T, alphas, betas, ks, hs).T * vtaus).T 

    tArr     = np.arange(0,tmax, dt_save)
    nt_saves = tArr.shape[0]
    
    uArr    = np.zeros((nt_saves, u.shape[0], nc_save))
    vArr    = np.zeros((nt_saves, v.shape[0], nc_save))

    t        = 0
    save_idx = 0 #1
    eps      = dt / 1e7

    while t<tmax:
        
        if t % dt_save < eps: 
            uArr[save_idx] = u[:,0:nc_save]
            vArr[save_idx] = v[:,0:nc_save]
            save_idx += 1

        v = stepSlave(u, v, driver_idxs, alphas, betas, ks, hs, vtaus, dt, nslaves, ncells, scale)
        v = v*(v>0) # expression can't be negative (i.e., gene goes extinct)
        u = step(u, dt, m1, m2, h1, h2, tau, ncells, scale)
        u = u*(u>0)
        
        t += dt

    return(tArr, uArr, vArr, u, v)
