import numpy as np
from numpy import linalg
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial.distance import pdist

#####################################################
## a bunch of arbitrary functions i use sometimes! ##
#####################################################
norm0to1    = lambda x,ax : ((x.T-np.amin(x,axis=ax))/(np.amax(x,axis=ax)-np.amin(x,axis=ax))).T
apply_fun2d = lambda fun, x: [[fun(x[i][j]) for j in range(len(x[i]))] for i in range(len(x))]
nz_sign     = lambda x: 2*np.array(x>=0, dtype='int')-1 

def argmaxsort(m):
    ams    = m.argmax(axis = 0) # position of max for every gene
#    amsort = ams.argsort()      # ordering of those positions
#    breaks       = np.where(np.diff(ams[amsort])!=0)[0] # indexes where it switches
#    amsortSplit  =  np.split(amsort, breaks+1) # groups of indexes with common argmax
    amsortSplit = [[] for i in range(m.shape[0])]
    for j in range(m.shape[1]):
        amsortSplit[ams[j]].append(j)

    amsortSplit = [np.array(k) for k in amsortSplit]
    subOrders   = [np.argsort(m[i,amsortSplit[i]]) if amsortSplit[i].shape[0]>0 else np.array([]) for i in range(m.shape[0])] # subsort within each group
    sortOrder2  = np.hstack([amsortSplit[i][subOrders[i]] for i in range(len(subOrders)) if amsortSplit[i].shape[0]>0]) # recombine into one matrix
    
    return sortOrder2, list(map(len, amsortSplit))

def list2multiDict(l):
    d = {}
    for k,v in l:
        if k in d:
            d[k] += [v]
        else:
            d[k] = [v]
    return d

def multiDictMerge(lA, lB):
    # lA has structure: keyA:[valA1, valA2, ...]
    # lB has structure: keyB:[valB1, valB2, ...]
    # where keyB can be valA1, valA2 ...
    # output has structure: keyA: [valB1, valB2, ...]
    d = {}
    
    for kA,vsA in lA.items():
        #vsA   = list.copy(lA[kA])
        d[kA] = list.copy(lB.get(vsA[0], []))
#        d[kA] = list.copy(lB[vsA[0]]) # assumes no null keys
        for vA in vsA[1:]:
            d[kA] += list.copy(lB.get(vA,[]))
        #print('i = {0} dlens = {1}'.format(i, [len(d[k]) for k in d]))

    return d

def reverseDictList(d1):
    d2 = {}
    for k,vs in d1.items():
        for v in vs:
            d2[v] = d2.get(v, [])
            d2[v].append(k)
    return d2

flatten2d = lambda x: [i for j in x for i in j] 

def pArgsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)

cipv = lambda x,y: stats.t.interval(y, len(x)-1, loc=np.mean(x),  scale = stats.sem(x))
ci95 = lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale = stats.sem(x))

def venn3_split(sets):
    '''
    input: [A, B, C] where they're all python sets
    output: [A, B, C, AB, AC, BC, ABC] where XY is set union of X and Y
    '''
    set111 = sets[0].intersection(sets[1]).intersection(sets[2])
    set110 = sets[0].intersection(sets[1]).difference(sets[2])
    set101 = sets[0].intersection(sets[2]).difference(sets[1])
    set011 = sets[1].intersection(sets[2]).difference(sets[0])
    set100 = sets[0].difference(sets[1]).difference(sets[2])
    set010 = sets[1].difference(sets[0]).difference(sets[2])
    set001 = sets[2].difference(sets[0]).difference(sets[1])
    return [set100, set010, set001, set110, set101, set011, set111]

# source: https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
# NOTE: CURRENTLY ONLY TESTED WHEN RESAMPLING HAS SIZE 1
def bootstrappedCI(mu, samps, ciWidth):
    sampMus = np.mean(samps, axis=1) if len(samps.shape) > 1 else samps
    diffs   = sampMus - mu
    order   = np.argsort(diffs)
    pvalHf  = (1-ciWidth)/2
    idx0    = int(np.floor(pvalHf*len(diffs)))
    idx1    = int(np.ceil((1-pvalHf)*len(diffs)))
    return (mu - diffs[order][idx1], mu - diffs[order][idx0])

def set_axs_fontsize(ax,fs,inc_leg=False):
    items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    if inc_leg:
        items += ax.legend().get_texts()
    for itm in items:
        itm.set_fontsize(fs)

def clean_nanmax(s):
    if np.all(np.isnan(s)):
        return np.nan
    elif s.shape[0] == 0:
        return 0
    else:
        return np.nanmax(s)

divz = lambda x,y : np.divide(x, y, out=np.zeros_like(x), where=y!=0)
meanor0 = lambda x: np.nanmean(x) if x.shape[0]>0 else 0
stdor0 = lambda x: np.nanstd(x) if x.shape[0]>1 else 0
semor0     = lambda x: np.nanstd(x)/np.sqrt(x.shape[0]) if len(x)>1 else 0
#maxor0 = lambda x: np.nan if np.all(x) np.nanmax(x) if x.shape[0]>0 else 0

meanOrZero = lambda l: np.nanmean(l) if len(l) > 0 else 0

def logCenPca(dat,minval):
    lgdats   = np.log10(dat.T + minval)#[tpmThIdxs]
    ngenes   = lgdats.shape[0]
    mus      = lgdats.mean(axis=1)
    lgdatsfz = (lgdats.T-mus)#/sigs

    gpca    = linalg.svd(lgdatsfz, full_matrices = False)
    eigs    = gpca[1]**2/ngenes
    pcs     = lgdatsfz.dot(gpca[2].T)
    return pcs, lgdatsfz, gpca[2]

def intra_cluster_variation(kmeans_obj, samp, max_pts = 1000):
    # if too many points in the cluster, samples from them... 
    nc          = kmeans_obj.n_clusters
    clust_idxs  = [np.where(kmeans_obj.labels_== c)[0] for c in range(nc)]
    clust_var   = 0
    for i in range(nc):
        if clust_idxs[i].shape[0] < max_pts:
            clust_var += np.mean(pdist(samp[clust_idxs[i]]))
        else:
            clust_var += np.mean(pdist(samp[np.random.choice(clust_idxs[i], size = max_pts, replace = False)]))
    return clust_var
    #return sum([np.sum(pdist(samp[clust_idxs[i]])/clust_idxs[i].shape[0]) for i in range(nc)])/2.

def gap_stat(data, nsamples, ks, minibatch = False):
    '''
    nsamples = "B" in tibshirani paper
    https://statweb.stanford.edu/~gwalther/gap
    '''

    ndata, nfeatures = data.shape
    nks              = len(ks)

    # shuffled samp...
    samp_idxs = np.random.choice(np.arange(ndata),    size = (nsamples, ndata, nfeatures))
    samps     = np.array([np.random.choice(data[:,i], size = (nsamples, ndata)) for i in range(nfeatures)])
    samps     = samps.transpose(1,2,0)

    # cluster all samps

    # samp_kmeans = [[] for i in range(len(ks))]
    # dat_kmeans  = []

    wks  = np.zeros(nks)
    wkbs = np.zeros((nks, nsamples))

    kmeans_func = KMeans if not minibatch else MiniBatchKMeans

    for i in range(nks):
        print('\ncomputing k = {0}'.format(ks[i]))
        kmu    = kmeans_func(n_clusters = ks[i], random_state = 0).fit(data)
        print('\ncomputing intra cluster variation'.format(ks[i]))

        wks[i] = intra_cluster_variation(kmu, data)
        #dat_kmeans.append(kmu)

        for j in range(nsamples):
            kmu       = kmeans_func(n_clusters = ks[i], random_state = 0).fit(samps[j])
            wkbs[i,j] = intra_cluster_variation(kmu, samps[j])
            #samp_kmeans[j].append(kmu)

    gap_stats = np.mean(np.log(wkbs).T-np.log(wks),axis=0)
    lbar      = np.mean(np.log(wkbs),axis=1)
    sdk       = np.sqrt(np.mean((np.log(wkbs).T-lbar)**2,axis=0))

    opt_k_idxs = np.where(gap_stats[:-1] - (gap_stats[1:] - sdk[1:]) > 0)[0]
    opt_k      = ks[opt_k_idxs[0]] if len(opt_k_idxs) > 0 else -1

    return gap_stats, sdk, opt_k

def longest_elem(l):
    if len(l) == 0:
        return -1
    else:
        longest = l[0]
        for i in range(1,len(l)):
            if len(l[i]) > len(longest):
                longest = l[i]
        return longest

def peak_idxs(dat, ht_thresh, merge_dist):

    pks = np.where(dat > ht_thresh)[0]
    
    # if points are adjacent, merge them
    pkdiff = np.diff(pks)
    pk_grps    = [[pks[0]]]
    
    for i in range(pkdiff.shape[0]):
        if pkdiff[i] < merge_dist:
            pk_grps[-1].append(pks[i+1])
        else:
            pk_grps.append([pks[i+1]])
        
    return np.array([grp[np.argmax(dat[grp])] for grp in pk_grps])

def unNorm(g):
    return np.array(np.around(g / np.min(g[np.nonzero(g)])),dtype='int')

def unNorm2(g, fac):
    return np.array(g*fac,dtype='int')

loadDE = lambda dr,n1,n2,cols: np.genfromtxt('{0}/{1}_v_{2}.tsv'.format(dr,n1,n2), dtype='float',
                                             delimiter='\t',encoding=None,skip_header=1,usecols=cols)

# from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib      
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# from https://stackoverflow.com/questions/17794266/how-to-get-the-highest-element-in-absolute-value-in-a-numpy-matrix
def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    p = abs(maxa) > abs(mina) # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
    if axis == None:
        if p: return maxa
        else: return mina
    shape = list(a.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=a.dtype)
    out[p] = maxa[p]
    out[n] = mina[n]
    return out
