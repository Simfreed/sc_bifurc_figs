import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir",        type=str, help  = "output subdirectory", default='eigs_ncell_sample') 
parser.add_argument("--neval",      type=int, help="# of evals", default=1)
parser.add_argument("--bin_sz",     type=int, help="# of cells per bin", default=1000)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=20)
parser.add_argument("--ti",         type=int, help="first tau", default=0)
parser.add_argument("--tf",         type=int, help="last tau", default=-1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)
parser.add_argument("--ncells",     type=int, help="list of numbers of cells to use (space delimited)", nargs = '+', default=[5,10,20,50,100,200,500,1000])

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
headdir    = '.'
datdir     = '{0}/neutrophil_data'.format(headdir)
outdir     = '{0}/{1}'.format(datdir, args.dir)
os.makedirs(outdir, exist_ok = True)

gexp_fname = '{0}/gene_expr.npz'.format(datdir)
gexp_sp    = scipy.sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds

# load pseudotime cell indexes
pst_fname         = '{0}/pseudotime.txt'.format(datdir)
neut_psts         = np.genfromtxt(pst_fname, skip_header=True, dtype='int')

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npst              = len(neut_pst_cidxs)

# run sampling
ti = args.ti
tf = args.tf if args.tf > 0 else npst

trange      = np.arange(ti,tf)
pst_nc_eig1 = np.zeros((trange.shape[0], nsamps, npc))
np.save('{0}/trange.npy'.format(outdir), trange)

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

    np.save('{0}/ncell{1}.npy'.format(outdir, nc), pst_nc_eig1)
