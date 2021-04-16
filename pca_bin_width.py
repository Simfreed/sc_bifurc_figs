import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir",        type=str, help  = "output subdirectory", default='eigs_bin_size') 
parser.add_argument("--neval",      type=int, help="# of evals", default=1)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=20)
parser.add_argument("--ti_frac",    type=float, help="fraction of trajectory to start at", default=0)
parser.add_argument("--tf_frac",    type=float, help="fraction of trajectory to end at",  default=1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)
parser.add_argument("--bin_szs",    type=int, help="list of numbers of cells to use (space delimited)", nargs = '+', default=[20,50,100,200,500,1000, 2000])


args = parser.parse_args()

# params
nsamps    = args.nsamp
npc       = args.neval 
#ncells    = np.array(args.ncells)

np.random.seed(args.seed)
samp_repl = True

# load gene expression and pseudotime
headdir    = '.'
datdir     = '{0}/neutrophil_data'.format(headdir)
outdir     = '{0}/{1}'.format(datdir, args.dir)
gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)


gexp_sp    = scipy.sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds

neut_psts  = np.genfromtxt(pst_fname, skip_header=True, dtype='int')
srt        = np.argsort(neut_psts[:,1])

os.makedirs(outdir, exist_ok = True)

pca   = PCA(n_components=npc)

for i in range(len(args.bin_szs)):
    bin_sz   = args.bin_szs[i]
    overlap  = int(args.ov_frac * bin_sz)

    last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
    neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
    neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
    neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
    npst              = len(neut_pst_cidxs)
    
    ti = int(max(np.floor(args.ti_frac*npst) , 0   ))
    tf = int(min(np.ceil( args.tf_frac*npst) , npst)) 

    trange  = np.arange(ti,tf)
    evals   = np.zeros((trange.shape[0], npc))


    print('bin size = {0}'.format(bin_sz))

    for j in range(trange.shape[0]):
        t = trange[j]
        if t%100==0:
            print('\tbin {0}/{1}'.format(t,npst))
        pca.fit(gexp_lil[neut_pst_cidxs[t]].toarray())
        evals[j] = pca.explained_variance_

    np.save('{0}/trange_bsz{1}.npy'.format(outdir, bin_sz), trange)
    np.save('{0}/evals_bsz{1}.npy'.format( outdir, bin_sz), evals)

