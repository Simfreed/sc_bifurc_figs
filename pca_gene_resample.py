import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os
import myfun as mf

parser = argparse.ArgumentParser()

parser.add_argument("--neval",      type=int, help="# of evals", default=-1)
parser.add_argument("--bin_sz",     type=int, help="# of cells per bin", default=1000)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=0)
parser.add_argument("--ti",         type=int, help="first tau", default=0)
parser.add_argument("--tf",         type=int, help="last tau", default=-1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)

parser.add_argument("--gexp_fname", type=str, help = "ncell x ngene gene expression matrix in scipy.sparse format", default='neutrophil_data/gene_expr.npz')
parser.add_argument("--pst_fname",  type=str, help = "input/output data location",                                  default='neutrophil_data/pseudotime.txt')
parser.add_argument("--outdir",     type=str, help = "directory for output",                                        default='neutrophil_data/eig') 


args = parser.parse_args()

# params
nsamp     = args.nsamp
samp_repl = True
bin_sz    = args.bin_sz
overlap   = int(args.ov_frac * bin_sz)
npc       = args.neval if args.neval > 0 else 1 
np.random.seed(args.seed)

# r / w
print('creating output directory at {0}'.format(args.outdir))
os.makedirs(args.outdir, exist_ok = True)

print('loading gene expr matrix')
gexp_sp    = scipy.sparse.load_npz(args.gexp_fname) # ~ 20s (filesize is 1.1GB)
gexp_lil   = gexp_sp.tolil() # ~4min
print('done loading gene expr matrix')

# load pseudotime cell indexes
print('binning by pseudotime')
neut_psts         = np.genfromtxt(args.pst_fname, skip_header=True)
srt               = np.argsort(neut_psts[:,1])
pst_bins          = mf.get_bins(srt, bin_sz, overlap)
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in pst_bins]
npst              = len(neut_pst_cidxs)

# run sampling
print('sampling')

cov_evals      = np.zeros([npst, npc])
cov_evecs      = np.zeros((npst, npc, gexp_lil.shape[1]))
cov_evals_shuf = np.zeros([nsamp, npc])
pca            = PCA(n_components=npc)

ti = args.ti
tf = args.tf if args.tf > 0 else npst

bin_time = []
for t in range(ti, tf):

    if t%5 == 0:
        print('bin {0}/{1}'.format(t,npst))

    bin_time.append(np.mean(neut_psts[neut_pst_cidxs[t],1]))

    gexpt        = gexp_lil[neut_pst_cidxs[t]].toarray()
    ncell, ngene = gexpt.shape

    pca.fit(gexpt)
    cov_evals[t]   = pca.explained_variance_
    cov_evecs[t]   = pca.components_

    for i in range(nsamp):
        gexp_shuf = np.array([gexpt[:,g][np.random.choice(ncell, ncell, replace=True)] 
                              for g in range(ngene)]).T
        pca.fit(gexp_shuf)
        cov_evals_shuf[i] = pca.explained_variance_
    
    np.save('{0}/shuf_eval_t{1}.npy'.format(args.outdir, t), cov_evals_shuf)
    
np.save('{0}/dat_evec.npy'.format(args.outdir), cov_evecs)
np.save('{0}/dat_eval.npy'.format(args.outdir), cov_evals)
np.save('{0}/bin_psts.npy'.format(args.outdir), np.array(bin_time))
