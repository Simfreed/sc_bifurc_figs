import numpy as np
import scipy as scipy
from scipy import io as scio
from sklearn.decomposition import PCA
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--dir",        type=str, help  = "output subdirectory", default='eig') 
parser.add_argument("--neval",     type=int, help="# of evals", default=-1)
parser.add_argument("--bin_sz",     type=int, help="# of cells per bin", default=1000)
parser.add_argument("--ov_frac",     type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=0)
parser.add_argument("--quest",       dest='quest', action='store_true')
parser.add_argument("--ti",         type=int, help="first tau", default=0)
parser.add_argument("--tf",         type=int, help="last tau", default=-1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)

args = parser.parse_args()

# params
nsamp     = args.nsamp
samp_repl = True
bin_sz    = args.bin_sz
overlap   = int(args.ov_frac * bin_sz)
npc       = args.neval if args.neval > 0 else bin_sz

np.random.seed(args.seed)

# r / w
if args.quest:
    headdir = '/projects/p31095/simonf/out/weinreb_2020/'
else:
    headdir = '/Users/simonfreedman/cqub/bifurc/weinreb_2020'

datdir     = 'neutrophil_data'
gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)

print('loading gene expr matrix')
gexp_sp    = scipy.sparse.load_npz(gexp_fname) # ~ 20s (filesize is 1.1GB)
gexp_lil   = gexp_sp.tolil() # ~4min
print('done loading gene expr matrix')

outdir    = '{0}/{1}'.format(datdir, args.dir)
os.makedirs(outdir, exist_ok = True)

# load pseudotime cell indexes
print('binning by pseudotime')
neut_psts         = np.genfromtxt(pst_fname, skip_header=True, dtype='int')

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npst              = len(neut_pst_cidxs)

# run sampling
print('sampling')

cov_evals      = np.zeros([npst, npc])
cov_evecs      = np.zeros((npst, npc, gexp_lil.shape[1]))
cov_evals_shuf = np.zeros([nsamp, npc])
pca            = PCA(n_components=npc)

ti = args.ti
tf = args.tf if args.tf > 0 else npst

for t in range(ti, tf):

    if t%5 == 0:
        print('bin {0}/{1}'.format(t,npst))
        
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
    
    np.save('{0}/shuf_eval_t{1}.npy'.format(outdir, t), cov_evals_shuf)
    
np.save('{0}/dat_evec.npy'.format(outdir), cov_evecs)
np.save('{0}/dat_eval.npy'.format(outdir), cov_evals)
