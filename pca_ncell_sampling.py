import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os
import myfun as mf

parser = argparse.ArgumentParser()

parser.add_argument("--neval",      type=int, help="# of evals", default=1)
parser.add_argument("--bin_sz",     type=int, help="# of cells per bin", default=1000)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=20)
parser.add_argument("--ti",         type=int, help="first tau", default=0)
parser.add_argument("--tf",         type=int, help="last tau", default=-1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)
parser.add_argument("--ncells",     type=int, help="list of numbers of cells to use (space delimited)", nargs = '+', default=[5,10,20,50,100,200,500,1000])

parser.add_argument("--gexp_fname", type=str, help = "ncell x ngene gene expression matrix in scipy.sparse format", default='neutrophil_data/gene_expr.npz')
parser.add_argument("--pst_fname",  type=str, help = "input/output data location",                                  default='neutrophil_data/pseudotime.txt')
parser.add_argument("--outdir",     type=str, help = "directory for output",                                        default='neutrophil_data/eigs_ncell_sample') 

args = parser.parse_args()

# params
nsamps    = args.nsamp
bin_sz    = args.bin_sz
overlap   = int(args.ov_frac * bin_sz)
npc       = args.neval 
ncells    = np.array(args.ncells)

np.random.seed(args.seed)
samp_repl = True

# load gene expression
os.makedirs(args.outdir, exist_ok = True)

gexp_sp    = scipy.sparse.load_npz(args.gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds

# load pseudotime cell indexes
neut_psts         = np.genfromtxt(args.pst_fname, skip_header=True, dtype='int')

srt               = np.argsort(neut_psts[:,1])
pst_bins          = mf.get_bins(srt, bin_sz, overlap)
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in pst_bins]
npst              = len(neut_pst_cidxs)

# run sampling
ti = args.ti
tf = args.tf if args.tf > 0 else npst

trange      = np.arange(ti,tf)
pst_nc_eig1 = np.zeros((trange.shape[0], nsamps, npc))
np.save('{0}/trange.npy'.format(args.outdir), trange)

pca   = PCA(n_components=npc)

for i in range(ncells.shape[0]):
    nc = ncells[i]
    print('ncells = {0}'.format(nc))

    for j in range(trange.shape[0]):
        t = trange[j]

        for s in range(nsamps):
            cidxs = np.random.choice(neut_pst_cidxs[t], nc, replace = samp_repl)
            pca.fit(gexp_lil[cidxs].toarray())
            pst_nc_eig1[j,s] = pca.explained_variance_

    np.save('{0}/ncell{1}.npy'.format(args.outdir, nc), pst_nc_eig1)
