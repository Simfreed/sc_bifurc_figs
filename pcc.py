import numpy as np
from scipy import sparse
import argparse
import os
from os import path

parser = argparse.ArgumentParser()

parser.add_argument("--wdir",       type=str, help="output subdirectory", default='neut_pst_corr') 
parser.add_argument("--quest",              dest='quest', action='store_true')
parser.add_argument("--skip_if_exists",     dest='skip_if_exists', action='store_true', help='skip pcc calculation if file already exists')
parser.add_argument("--bin_sz",     type = int, help="number of cells per pseudotime bin", default=100) 
parser.add_argument("--overlap",    type = float, help="fraction of cells that overlap between bins", default=0.5) 

parser.add_argument("--min_cell_gexp",     type = int, 
        help="min (non-inclusive) gene expression of an individual cell to use in pcc calc", default=0) 
parser.add_argument("--min_ncells",       type = int, 
        help="min (non-inclusive) number of cells to compute pcc", default=1) 

parser.add_argument("--t0", type = int, help="first time point to measure correlation", default=400) 
parser.add_argument("--tf", type = int, help="last time point to measure correlation",  default=600) 
#parser.add_argument("--seed",       type=int, help="random number seed", default=None)

args = parser.parse_args()

#np.random.seed(args.seed)

os.makedirs(args.wdir, exist_ok = True)

if args.quest:
    headdir = '/projects/p31095/simonf/out/weinreb_2020/'
else:
    headdir = '.'
    #headdir = '/Users/simonfreedman/cqub/bifurc/weinreb_2020'

#datdir     = '{0}/data'.format(headdir)
#gexp_fname = '{0}/in_vitro_normd_counts.npz'.format(datdir)
#pst_fname  = '{0}/stateFate_inVitro_neutrophil_pseudotime.txt'.format(datdir)

datdir     = '{0}/neutrophil_data'.format(headdir)
gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)

#########################################################
###### FUNCTION TO RUN CORRELATION WITH THRESHOLDS ######
#########################################################
def thresh_corr(gexp, min_cell_gexp = 0, min_ncells = 1):
    
    #gexp is an ngenes x ncells matrix
    
    # correlations are computed between genes gI and gJ between a subset of cells C
    # IFF
    #    C has greater than min_ncells elements AND
    #    there exists a cell cK in C for which gI > min_cell_gexp AND
    #    there exists a cell cL in C for which gJ > min_cell_gexp
    #
    # Thus, there is no need to examine genes who's 
    #       maximum expression < min_cell_gexp OR
    #       have fewer than min_ncells with greater than min_cell_gexp
    
    gidxs = []
    cidxs = []
    for i in range(gexp.shape[0]):
        ci = np.where(gexp[i] > min_cell_gexp)[0]
        if ci.shape[0] > min_ncells:
            gidxs.append(i)
            #cidxs.append(ci)
            cidxs.append(set(ci))

    gidxs = np.array(gidxs)
    ngns  = gidxs.shape[0]

    max_ncorr  = int(ngns*(ngns-1)/2.)
    corrs      = np.zeros(max_ncorr)                      # stores correlations (float array)
    corr_gidxs = np.zeros((max_ncorr,3), dtype=np.uint16) # stores g1, g2, noverlap, in a separate 'int' array for storage saving

    k = 0

    for i in range(ngns):

        cidxs_i = cidxs[i]

        for j in range(i+1, ngns):

            #cidxs_ij   = np.intersect1d(cidxs_i, cidxs[j])
            cidxs_ij   = np.array(list(cidxs_i.intersection(cidxs[j])))
            ncells     = cidxs_ij.shape[0]

            if ncells > min_ncells:
                corrs[k]      = np.corrcoef(gexp[gidxs[[i,j]]][:, cidxs_ij])[0,1]
                corr_gidxs[k] = gidxs[i], gidxs[j], ncells
                k += 1

    return corrs[:k], corr_gidxs[:k]

#########################################################
################## BACK TO THE CODE #####################
#########################################################

print('loading pseudotime data')
neut_psts         = np.genfromtxt(pst_fname, skip_header=True, dtype='int')
bin_sz            = args.bin_sz
overlap           = int(bin_sz*args.overlap)

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
ngrp              = len(neut_pst_cidxs)

print('loading gene expr matrix')
gexp_sp    = sparse.load_npz(gexp_fname) # ~ 20s (filesize is 1.1GB)
gexp_lil   = gexp_sp.tolil() # ~4min
print('done loading gene expr matrix')

ts = np.arange(args.t0,args.tf)
for t in ts:
    
    cidxs = neut_pst_cidxs[t]
    gexp_t = gexp_lil[cidxs].toarray()

    if args.min_cell_gexp < 0: # compute covariance, because cheap to go from there to pccs
        print('covariancing group {0}'.format(t))
        gidxs = np.where(gexp_lil[cidxs].sum(axis=0)>0)[1]
        gexp  = np.hstack([gexp_lil[cidxs, i].toarray() for i in gidxs])
        gcov  = np.cov(gexp.T)

        np.save('{0}/cov_{1}.npy'.format(args.wdir, t), gcov)
        np.save('{0}/nz_idxs_{1}.npy'.format(args.wdir, t), gidxs)
    else:
        pcc_f =  '{0}/pccs_{1}.npy'.format(args.wdir, t)
        gidx_f = '{0}/gidxs_nc_{1}.npy'.format(args.wdir, t)
        
        if args.skip_if_exists and path.exists(pcc_f) and path.exists(gidx_f):
            print('not correlating group {0} because already complete'.format(t))
        else:
            print('correlating group {0}'.format(t))
            pccs, gidxs_nc = thresh_corr(gexp_t.T, args.min_cell_gexp, args.min_ncells)
            np.save(pcc_f, pccs)
            np.save(gidx_f, gidxs_nc)
        
    #np.save('{0}/corr_{1}.npy'.format(args.wdir, t), gcorr)
