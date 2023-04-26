import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os
import myfun as mf

parser = argparse.ArgumentParser()

parser.add_argument("--neval",      type=int, help="# of evals", default=1)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=20)
parser.add_argument("--ti_frac",    type=float, help="fraction of trajectory to start at", default=0)
parser.add_argument("--tf_frac",    type=float, help="fraction of trajectory to end at",  default=1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)
parser.add_argument("--bin_szs",    type=int, help="list of numbers of cells to use (space delimited)", nargs = '+', default=[20,50,100,200,500,1000, 2000])

parser.add_argument("--gexp_fname", type=str, help = "ncell x ngene gene expression matrix in scipy.sparse format", default='neutrophil_data/gene_expr.npz')
parser.add_argument("--pst_fname",  type=str, help = "input/output data location",                                  default='neutrophil_data/pseudotime.txt')
parser.add_argument("--outdir",     type=str, help = "directory for output",                                        default='neutrophil_data/eigs_bin_size') 

args = parser.parse_args()

# params
nsamps    = args.nsamp
npc       = args.neval 

np.random.seed(args.seed)
samp_repl = True

# load gene expression and pseudotime
gexp_sp    = scipy.sparse.load_npz(args.gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds

neut_psts  = np.genfromtxt(args.pst_fname, skip_header=True, dtype='int')
srt        = np.argsort(neut_psts[:,1])

os.makedirs(args.outdir, exist_ok = True)

pca   = PCA(n_components=npc)

for i in range(len(args.bin_szs)):
    bin_sz   = args.bin_szs[i]
    overlap  = int(args.ov_frac * bin_sz)

    pst_bins          = mf.get_bins(srt, bin_sz, overlap)
    neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in pst_bins]
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

    np.save('{0}/trange_bsz{1}.npy'.format(args.outdir, bin_sz), trange)
    np.save('{0}/evals_bsz{1}.npy'.format( args.outdir, bin_sz), evals)

