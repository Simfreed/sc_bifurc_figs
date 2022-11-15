import numpy as np
from scipy import sparse
import pickle as pkl

# Needs: 
# pseudotime trajectory
# eigenvalues as function of time
# gexp for neutrophil genes
# gexp for myelocite genes


headdir    = '.' #'/Users/simonfreedman/cqub/bifurc/weinreb_2020/'
figdir     = '{0}/figs'.format(headdir)
datdir     = '{0}/neutrophil_data'.format(headdir)
eigdir     = '{0}/eig'.format(datdir)
#eigdir     = '{0}/eig_ncell_sample'.format(datdir)

gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)
gnm_fname  = '{0}/gene_names.txt'.format(datdir)
meta_fname = '{0}/metadata.txt'.format(datdir)

# In[6]:

print('loading gene expression matrix')
gexp_sp    = sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds


# In[8]:

print('loading cluster labels and SPRING positions')
dtp      = np.dtype([('Library Cell', np.unicode_, 16),('barcode', np.unicode_, 20),
              ('Time point', int),('Starting population', np.unicode_, 20),
               ('Cell type annotation', np.unicode_, 60),
               ('Well', int), ('SPRING-x', np.float64), ('SPRING-y', np.float64)])

metadata = np.genfromtxt(meta_fname, delimiter='\t',skip_header=1, dtype=dtp)

nms      = dtp.names
gnms     = np.genfromtxt(gnm_fname,dtype='str')

print('loading neutrophil pseudotime ranking')
neut_psts = np.genfromtxt(pst_fname, skip_header=True, dtype='int')


# In[12]:

print('binning gene expression')
bin_sz            = 1000
overlap           = int(bin_sz/2)

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npsts             = len(neut_pst_cidxs)

print('eigen-decomposition')

pst_eig1     = np.load('{0}/dat_eval.npy'.format(eigdir))[:,0]
pst_pc1      = np.load('{0}/dat_evec.npy'.format(eigdir))[:,0]

###############################################################
# gene expression of highly varying, highly expressed genes...#
###############################################################
print('matrix of gene expression for highly expressed / varying genes')

nnz_thresh  = 0
cv_thresh   = 0.5
gexp_thresh = 1

mu_gexp = np.array([np.mean(gexp_lil[cidxs].toarray(),axis=0) for cidxs in neut_pst_cidxs]) # takes like a minute

np.save('{0}/high_var_gexp_trajs.npy'.format(datdir), mu_gexp)


###############################################################
# cos theta
###############################################################

print('eigenvalue projection')
# In[21]:


t_bifurc    = np.argmax(pst_eig1)
mag_bifurc  = np.amax(pst_eig1)


# In[22]:


bif_pc1 = pst_pc1[t_bifurc]
cosths = []

for t in range(npsts):
    gexp_t = gexp_lil[neut_pst_cidxs[t]].toarray()
    gexp_t_norm = (gexp_t.T / np.linalg.norm(gexp_t,axis=1))
    cosths.append(bif_pc1.dot(gexp_t_norm))

outf = open('{0}/costh_projs.pkl'.format(datdir),'wb')
pkl.dump(cosths, outf)
outf.close()

###############################################################
# marker gene expression
###############################################################

# In[26]:

print('marker gene expression')
gene_group_labs  = ['neutrophil','MPP','GPP','PMy','My']

neut_gnms        = np.array(['S100a9', 'Itgb2l', 'Elane', 'Fcnb', 'Mpo', 'Prtn3', 
                              'S100a6', 'S100a8', 'Lcn2', 'Lrg1'])
mpp_gnms         = np.array(['Ly6a','Meis1','Flt3','Cd34'])
gmp_gnms         = np.array(['Csf1r','Cebpa'])
pmy_gnms         = np.array(['Gfi1','Elane'])
my_gnms          = np.array(['S100a8','Ngp','Ltf'])

grp_gnms  = [neut_gnms, mpp_gnms, gmp_gnms, pmy_gnms, my_gnms]

grp_gidxs = [np.array([np.where(gnms==gnm)[0][0] for gnm in k]) for k in grp_gnms]


# In[27]:


grp_gexp = [[np.hstack([gexp_lil[cidxs, k].toarray() for k in grp]) 
             for cidxs in neut_pst_cidxs]
            for grp in grp_gidxs]
outf = open('{0}/marker_gene_trajs.pkl'.format(datdir),'wb')
pkl.dump(grp_gexp, outf)
outf.close()
