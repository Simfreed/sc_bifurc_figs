import numpy as np
import scipy as scipy
from sklearn.decomposition import PCA
import argparse
import os
import myfun as mf
import pickle as pkl

parser = argparse.ArgumentParser()

parser.add_argument("--neval",      type=int, help="# of evals", default=-1)
parser.add_argument("--bin_sz",     type=int, help="# of cells per bin", default=1000)
parser.add_argument("--ov_frac",    type=float, help="fraction of bin to overlap", default=0.5)
parser.add_argument("--nsamp",      type=int, help="# of samps for null", default=0)
parser.add_argument("--nboot",      type=int, help="# of samps for bootstrapping", default=0)
parser.add_argument("--ti",         type=int, help="first tau", default=0)
parser.add_argument("--tf",         type=int, help="last tau", default=-1)
parser.add_argument("--seed",       type=int, help="random number seed", default=None)

parser.add_argument("--gexp_fname", type=str, help = "ncell x ngene gene expression matrix in scipy.sparse format", default='neutrophil_data/gene_expr.npz')
parser.add_argument("--pst_fname",  type=str, help = "input/output data location",                                  default='neutrophil_data/pseudotime.txt')
parser.add_argument("--outdir",     type=str, help = "directory for output",                                        default='neutrophil_data/eig') 
parser.add_argument("--tau_bins",   dest='tau_bins', action='store_true', help="if true, use bins of constant latent time width, not constant size")

args = parser.parse_args()

# params
nsamp     = args.nsamp
nboot     = args.nboot
bin_sz    = args.bin_sz
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
tau         = np.genfromtxt(args.pst_fname, skip_header=True)
if args.tau_bins:
    pst_bins = mf.get_time_bin_idxs(tau[:,1], bin_sz, args.ov_frac) # returns indexes of tau
else:
    pst_bins = mf.get_bin_idxs(tau[:,1], bin_sz, args.ov_frac) #returns elements of "srt" which are indexes of tau

bin_cidxs   = [np.array(tau[grp,0], dtype = 'int')  for grp in pst_bins]
bin_time    = np.array([mf.meanor0(tau[grp,1])         for grp in pst_bins])
npst        = len(bin_cidxs)

print('saving bins')
with open('{0}/bin_cidxs.pkl'.format(args.outdir), 'wb') as output_file:
    pkl.dump(bin_cidxs, output_file)

np.save('{0}/bin_psts.npy'.format(args.outdir), bin_time)

# run sampling
print('sampling')

cov_evals           = np.zeros([npst, npc])
cov_eval_rat        = np.zeros([npst, npc])
cov_evecs           = np.zeros((npst, npc, gexp_lil.shape[1]))

cov_evals_shuf      = np.zeros([nsamp, npc])
cov_evals_boot      = np.zeros([nboot, npc])

cov_eval_rat_shuf   = np.zeros([nsamp, npc])
cov_eval_rat_boot   = np.zeros([nboot, npc])

pca                 = PCA(n_components=npc)

ti = args.ti if args.ti < npst else 0
tf = args.tf if args.tf > ti else npst
ts = []

for t in range(ti, tf):

    if len(bin_cidxs[t]) < npc + 1:
        print('at t = {0}, ncells = {1}, so skipping pca (npc = {2})'.format(t, len(pst_bins[t]), npc))
        continue

    ts.append(t)
    if t%5 == 0:
        print('bin {0}/{1}'.format(t,npst))

    gexpt        = gexp_lil[bin_cidxs[t]].toarray()
    ncell, ngene = gexpt.shape

    pca.fit(gexpt)
    cov_evals[t]    = pca.explained_variance_
    cov_evecs[t]    = pca.components_
    cov_eval_rat[t] = pca.explained_variance_ratio_

    for i in range(nsamp):
        gexp_shuf = np.array([gexpt[:,g][np.random.choice(ncell, ncell, replace=True)] 
                              for g in range(ngene)]).T
        pca.fit(gexp_shuf)
        cov_evals_shuf[i]    = pca.explained_variance_
        cov_eval_rat_shuf[i] = pca.explained_variance_ratio_
    
    np.save('{0}/shuf_eval_t{1}.npy'.format(args.outdir, t), cov_evals_shuf)
    np.save('{0}/shuf_eval_rat_t{1}.npy'.format(args.outdir, t), cov_eval_rat_shuf)

    for i in range(nboot):
        cidxs     = np.random.choice(ncell, size=ncell, replace=True)

        while np.unique(cidxs).shape[0]==1: # redraw if the same cell is chosen bc can't be pca'd
            cidxs     = np.random.choice(ncell, size=ncell, replace=True)
        
        gexp_shuf = gexpt[cidxs,:]
        pca.fit(gexp_shuf)
        cov_evals_boot[i] = pca.explained_variance_
        cov_eval_rat_boot[i] = pca.explained_variance_ratio_
    
    np.save('{0}/boot_eval_t{1}.npy'.format(args.outdir, t), cov_evals_boot)    
    np.save('{0}/boot_eval_rat_t{1}.npy'.format(args.outdir, t), cov_eval_rat_boot)
    
np.save('{0}/bin_ids.npy'.format(args.outdir),  np.array(ts))
np.save('{0}/dat_evec.npy'.format(args.outdir), cov_evecs)
np.save('{0}/dat_eval.npy'.format(args.outdir), cov_evals)
np.save('{0}/dat_eval_rat.npy'.format(args.outdir), cov_eval_rat)
