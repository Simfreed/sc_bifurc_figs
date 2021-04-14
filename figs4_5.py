import numpy as np
import scipy as scipy
from scipy import io as scio
import pickle
import sys
import copy
import myfun as mf
from sklearn.decomposition import PCA

import matplotlib
from matplotlib import colors, ticker, gridspec, rc, transforms
from matplotlib.ticker import PercentFormatter, LogFormatter, FuncFormatter, LogLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#sys.path.append('/Users/simonfreedman/cqub/xenopus/python/')


rc('text',usetex=False)
#matplotlib.rc('text.latex', preamble=r'\usepackage{color}')


# In[3]:


# Needs: 
# pseudotime trajectory
# eigenvalues as function of time
# gexp for neutrophil genes
# gexp for myelocite genes


headdir    = '/Users/simonfreedman/cqub/bifurc/weinreb_2020/'
figdir     = 'figs'

datdir     = '{0}/data'.format(headdir)
plotdir    = '{0}/plots'.format(headdir)

fpref      = 'GSM4185642_stateFate_inVitro_'
gexp_fname = '{0}/in_vitro_normd_counts.npy'.format(datdir)
pst_fname  = '{0}/stateFate_inVitro_neutrophil_pseudotime.txt'.format(datdir)
gnm_fname  = '{0}/{1}gene_names.txt'.format(datdir,fpref)


# In[6]:


gexp_fname = '{0}/in_vitro_normd_counts.npz'.format(datdir)
gexp_sp    = scipy.sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds


# In[8]:


dtp      = np.dtype([('Library Cell', np.unicode_, 16),('barcode', np.unicode_, 20),
              ('Time point', np.int),('Starting population', np.unicode_, 20),
               ('Cell type annotation', np.unicode_, 60),
               ('Well', np.int), ('SPRING-x', np.float64), ('SPRING-y', np.float64)])

metadata = np.genfromtxt('{0}/{1}metadata.txt'.format(datdir,fpref),
                         delimiter='\t',skip_header=1, dtype=dtp)

nms      = dtp.names
gnms     = np.genfromtxt(gnm_fname,dtype='str')


# In[9]:


keys           = metadata['Cell_type_annotation']
ctypes         = np.unique(keys)
ctype_idx_dict = dict(zip(ctypes,range(ctypes.shape[0])))
ctype_idxs     = np.array([ctype_idx_dict[k] for k in keys])
ctype_grps     = [np.where(ctype_idxs==i)[0] for i in range(len(ctypes))]
ctype_mean_pos = np.array([[np.mean(metadata['SPRINGx'][grp]),np.mean(metadata['SPRINGy'][grp])] 
                           for grp in ctype_grps])


# In[11]:


neut_psts = np.genfromtxt(pst_fname, skip_header=True, dtype='int')


# In[12]:


bin_sz            = 1000
overlap           = int(bin_sz/2)

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])


# In[14]:


neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npsts             = len(neut_pst_cidxs)


# In[15]:


# this can be run quickly so i'll leave it here
pst_eig1 = np.zeros(npsts)
pst_pc1  = np.zeros((npsts, gexp_lil.shape[1]))
for i in range(npsts):
    if i%100==0:
        print(i)
    pca   = PCA(n_components=1)
    pca.fit(gexp_lil[neut_pst_cidxs[i]].toarray())

    # plain ol pca
    pst_eig1[i] = pca.explained_variance_[0]
    pst_pc1[i]  = pca.components_[0]


# In[16]:

# this takes a while, so run with script: pca_gene_resample.py
# grp_evals_null  = np.array([np.load('{0}/full_evals_null{1}.npy'.format(datdir,i)) 
#                       for i in range(neut_grp_psts.shape[0])])
nsamp                 = 20
pst_grp_null_eval     = np.load('{0}/w1_gene_resample_bsz{1}_ns{2}.npy'.format(datdir,bin_sz, nsamp))
pst_grp_null_eval_mu  = np.mean(pst_grp_null_eval,axis=1)
pst_grp_null_eval_err = np.std(pst_grp_null_eval,axis=1)


# In[17]:


pst_eig1_n              = pst_eig1 - np.amin(pst_eig1)
pst_grp_null_eval_n     = pst_grp_null_eval-np.amin(pst_grp_null_eval)
pst_grp_null_eval_mu_n  = np.mean(pst_grp_null_eval_n,axis=1)
pst_grp_null_eval_err_n = np.std(pst_grp_null_eval_n,axis=1)


# In[18]:

###############################################################
# gene expression of highly varying, highly expressed genes...#
###############################################################
nnz_thresh  = 0
cv_thresh   = 0.5
gexp_thresh = 1

mu_gexp = np.array([np.mean(gexp_lil[cidxs].toarray(),axis=0) for cidxs in neut_pst_cidxs]) # takes like a minute


# In[19]:


nnzs         = np.sum(mu_gexp>0,axis=0)
mu_mu_gexp   = np.mean(mu_gexp,axis=0)
max_mu_gexp  = np.max(mu_gexp,axis=0)

std_mu_gexp = np.std(mu_gexp,axis=0)
cvs         = np.divide(std_mu_gexp, mu_mu_gexp, out = np.zeros(mu_gexp.shape[1]), where=mu_mu_gexp>0)

gidxs       = np.where((nnzs > nnz_thresh) & (cvs>cv_thresh) & (max_mu_gexp>gexp_thresh))[0]
am_sort     = mf.argmaxsort(mu_gexp[:,gidxs])
gexp_arr    = mu_gexp[:,gidxs[am_sort[0]]].T


# In[20]:


###############################################################
# cos theta
###############################################################


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


# In[23]:


cos_th_min = -1
cos_th_max = 0.6
d_cos_th = 0.1
cos_th_rng = np.arange(cos_th_min, cos_th_max+d_cos_th, d_cos_th)
cos_th_hists = np.array([np.histogram(cosths[t], bins = cos_th_rng, density=True)[0] for t in range(npsts)])

cos_th_bin_ctrs = 0.5*(cos_th_rng[1:]+cos_th_rng[:-1])
cos_ths_flat = cos_th_hists.reshape(-1)
cos_th_nz_min = np.amin(cos_ths_flat[np.nonzero(cos_ths_flat)])
cos_th_eps = cos_th_nz_min/10


# In[31]: null difference + tau_d


null_diff   = pst_eig1_n - pst_grp_null_eval_mu
t_bifurc_pf = np.where(null_diff>0)[0][0]
t_bifurc_pf = np.where(np.diff(pst_eig1_n)>125)[0][0]



############################
#### fig 4 #################
############################

plt.style.reload_library()
plt.style.use('one_col_fig')
taulims = np.array([-3,124])

leg_ht  = 2
leg_spc = 7
spc1_ht  = 10
spc2_ht = 3
spc3_ht  = 5
spc1 = 1
schem1_ht   = 30
tseries_ht = 20
tau_series_ht = 25

col1_wd = 30
col2_wd = 20
spc_wd = 20
spc2_wd=1
spc3_wd=15

wds = np.array([
    spc3_wd,
    col1_wd,
    spc_wd,
    col2_wd-spc2_wd-leg_ht,
    spc2_wd,
    leg_ht
])

cs = np.cumsum(wds) # starting cols 
nc = np.sum(wds)

schem2_ht = int(schem_dy/schem_dx*nc)

# row heights
hts = np.array([
    
    leg_ht,
    leg_spc,
    
    schem1_ht,   
    spc1_ht,
    
#    spc1,
#    schem2_ht,
    
    spc2_ht,
    tau_series_ht,
    spc3_ht,
    tau_series_ht
    
])

rs = np.cumsum(hts) # starting rows
nr = np.sum(hts)

wid = 8.7/2.54
ht  = wid*nr/nc


fig = plt.figure(figsize=(wid, ht), dpi=200) 

gs = gridspec.GridSpec(nr, nc)

axAL = plt.subplot( gs[0    :rs[0], cs[0]:cs[1]]) # SPRING heat map legend
axA  = plt.subplot( gs[rs[1]:rs[3],0:cs[2]]) # SPRING heat map

axBL = plt.subplot( gs[0    :rs[0], cs[2]:]) # gene expression heat map legend
axB  = plt.subplot( gs[rs[1]:rs[2], cs[2]:]) # gene expression heat map

axC  = plt.subplot( gs[rs[4]:rs[5], cs[0]:cs[3]]) # covariance eigenvalue
axD  = plt.subplot( gs[rs[6]:rs[7], cs[0]:cs[3]]) # costh
axDL = plt.subplot( gs[rs[6]:rs[7], cs[4]:     ]) # costh legend


caps = ['A','B','C','D']
ri   = [0,   0, rs[4],rs[6]]
ci   = [0,cs[1],0,0]
ys   = [0,0,1,1]
xs   = [-2.5,10,-2.5,-2.5]
for i in range(len(caps)):
    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i], 
                x=xs[i],
                y=ys[i],fontsize=14,verticalalignment='top',horizontalalignment='left')
    cap_ax.axis('off')
    
#####################################
## A: SPRING plot                ####
#####################################
skip=1

# plot non-neut points
traj_idxs = np.array(neut_psts[:,0],dtype='int')
idxs      = np.setdiff1d(np.arange(metadata.shape[0]), traj_idxs)
springXlims = [np.amin(metadata['SPRINGx'][idxs[::skip]]),np.amax(metadata['SPRINGx'][idxs[::skip]])]
springYlims = [np.amin(metadata['SPRINGy'][idxs[::skip]]),np.amax(metadata['SPRINGy'][idxs[::skip]])]
axA.scatter(metadata['SPRINGx'][idxs[::skip]],metadata['SPRINGy'][idxs[::skip]], c='gray',alpha=0.01)
axA.set_xlim(springXlims[0],springXlims[1]+1500)
axA.set_ylim(springYlims[0]-1000,springYlims[1])

# annotate cell types
ctype_offset = {'Monocyte':np.array([-500,200]),
               'Undifferentiated':np.array([-600,300]),
               'Lymphoid':np.array([-1300,100]),
                'pDC':np.array([-50,150]),
                'Erythroid':np.array([-400,0]),
               'Baso':np.array([-200,-400]),
               'Meg':np.array([-150,100])}

for i in range(len(ctypes)):
    axA.annotate(xy=ctype_mean_pos[i]+ctype_offset.get(ctypes[i],np.array([0,0])),
                 text=ctypes[i],alpha=0.8,fontsize=6)
    
# plot points in neutrophil trajectory
skip = 1
cols = plt.cm.viridis(np.linspace(0,1,idxs.shape[0]))
axA.scatter(metadata['SPRINGx'][traj_idxs[::skip]],metadata['SPRINGy'][traj_idxs[::skip]], 
            c=cols[np.array(neut_psts[:,1],dtype='int')],alpha=0.5)

axA.set_xticks([])
axA.set_yticks([])

#frameless
axA.axis('off')

#colorbar
cmap   = plt.cm.get_cmap('viridis', traj_idxs.shape[0])
sm     = plt.cm.ScalarMappable(cmap=cmap)
nticks = 5
plt.colorbar(sm, cax=axAL, orientation='horizontal',ticks=np.linspace(0,1,nticks))
#axAL.xaxis.tick_top()
axAL.set_xticklabels(np.array(np.around(np.linspace(0,npsts,nticks)),dtype='int'))
axAL.set_title(r'pseudotime ($\tau$)')

#####################################
## B: gene expression trajectory ####
#####################################

cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap.set_bad('white')

im = axB.imshow(gexp_arr, cmap=cmap,aspect='auto',norm=matplotlib.colors.LogNorm(vmin=1e-3,vmax=2e2))

axB.set_xlabel(r'$\tau$',labelpad=-3)
axB.set_ylabel('gene index')
axB.set_xticks(np.arange(0,121,40))


cbar = fig.colorbar(im, cax=axBL, orientation='horizontal', aspect=1)
cbar.set_label(r'$\langle$expr$\rangle$ (tpm)',rotation=0, labelpad=4)
axBL.xaxis.set_label_position('top')
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=3) 
axBL.xaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=100)
axBL.xaxis.set_minor_locator(locmin)


#####################################
## C: covariance eigenvalue        ##
#####################################
xtix_all = np.array([0,20,40,60,80,100,120])
cols = ['k','gray']

axC.plot(pst_eig1_n,'o-', label = 'data',markersize=1, color=cols[0])
axC.errorbar(np.arange(npsts),pst_grp_null_eval_mu, yerr=pst_grp_null_eval_err, 
             color=cols[1], capsize=2,alpha=0.5, label='null')

axC.set_ylabel(r'cov. eval. 1 $(\omega_1)$')
# leg = axC.legend(loc = (0.7,0.6),labelspacing=0,frameon=False,handlelength=0.5,handletextpad=0)
# for i,text in zip(range(len(cols)),leg.get_texts()):
#     plt.setp(text, color = cols[i])

axC.set_xticks(xtix_all)
axC.set_xticklabels([])
axC.set_xlim(*taulims)
axC.set_yticks([0,5000,10000,15000,20000])

axC.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

### inset ###
axCin = axC.inset_axes([0.18, 0.15, 0.5, 0.6],transform=axC.transAxes)
ti=70
tf=121

col = 'blue'
axCin.errorbar(np.arange(ti,tf),pst_eig1_n[ti:tf]-pst_grp_null_eval_mu[ti:tf],
               yerr=pst_grp_null_eval_err[ti:tf],
               color=col, capsize=1)
axCin.set_yscale('symlog')
axCin.set_yticks([-100,0,100,10000])

#difftxt = r'$\omega_1$-$\omega_1^{\rm null}$'
#difftxt = r'$\textcolor{black}{\omega_1({\rm data})}-\textcolor{gray}{\omega_1({\rm null})}$'
difftxt = r'${\omega_1({\rm data})}-{\omega_1({\rm null})}$'
difftxts = [r'${\omega_1({\rm data})}$-', r'${\omega_1({\rm null})}$']

# axCin.text(x=0,y=0.98,s=difftxt,color=col,
#            transform=axCin.transAxes, verticalalignment='bottom',fontsize=8)
halines = ['right','left']
for i in range(len(difftxts)):
    axCin.text(x=0.55,y=1.05,s=difftxts[i],color=cols[i],horizontalalignment = halines[i],
               transform=axCin.transAxes, verticalalignment='bottom',fontsize=8)

axCin.set_xticks(xtix_all[np.where(xtix_all>ti)])
axCin.set_xticklabels([])

plt.setp(axCin.spines.values(), color=col)
plt.setp([axCin.get_xticklines(), axCin.get_yticklines()], color=col)
axCin.tick_params(color=col, labelcolor=col)


#####################################
## D: cos(theta)                   ##
#####################################

arr = cos_th_hists.T
arr[arr==0]=np.nan

tick_skip = 4

im=axD.imshow(arr, aspect='auto', origin = 'upper', cmap = cmap, zorder=0)
axD.set_yticks(np.arange(-0.5,cos_th_bin_ctrs.shape[0]+0.5,1)[::tick_skip])
axD.set_yticklabels(['{0:.1f}'.format(-i) for i in cos_th_rng[::tick_skip]])


#axD.set_ylabel(r'$\hat{g}^m_i\cdot \vec{s}^1_c$')
axD.set_ylabel(r'cov. evec. proj.')
axD.set_xlabel(r'control parameter ($\tau$)')
axD.text(s=r'$\hat{g}(\tau)\cdot \vec{s}^1(\tau_m)$',x=0.05,y=0.8,fontsize=10,transform=axD.transAxes)

#axD.set_xticklabels([])
axD.set_xlim(*taulims)
axD.set_xticks(xtix_all)

cbar = fig.colorbar(im, cax=axDL, orientation='vertical')
cbar.set_label('frequency',rotation=270, labelpad=8)
axDL.yaxis.set_label_position('right')
cbar.set_ticks(np.arange(0,10,2))


# bifurcation lines

bif_axs = [axC, axCin, axD]
zord = [0,0,1,0,0]
cols = ['orange','green']
bifts = np.array([t_bifurc,t_bifurc_pf])
lss = ['--','-.']
bifnms = [r'$\tau_{m}$',r'$\tau_{d}$']
bifxs = (bifts - taulims[0])/(taulims[1]-taulims[0]) #[0.8,0.6] #bifts / tf
for j in range(len(bifts)):
    axC.text(s=bifnms[j],x=bifxs[j],y=1,transform=axC.transAxes,color=cols[j],
             horizontalalignment = 'center', verticalalignment='bottom')
    for i in range(len(bif_axs)):
        bif_axs[i].axvline(bifts[j],color=cols[j], linestyle = lss[j], alpha=0.5,zorder=zord[i],lw=2)


plt.savefig('{0}/fig4_neut_cov.pdf'.format(figdir), bbox_inches='tight')


# In[ ]:


# intentionally left blank


# In[24]:

###############################################################
# In[33]: schematic
###############################################################

haem_dev_schem = plt.imread('pngs/neut_tree.png')
schem_dy, schem_dx, _ = haem_dev_schem.shape



###############################################################
# type densities
###############################################################
types_per_group_neut = [ctype_idxs[np.array(grp,dtype='int')] for grp in neut_pst_cidxs] 
type_bins            = np.arange(-0.5,len(ctypes)+0.5,1)
type_denss_neut      = np.array([
    np.histogram(grp, bins=type_bins, density=True)[0] if len(grp)>0 else blank_hist 
    for grp in types_per_group_neut])


# In[25]:


###############################################################
# marker gene expression
###############################################################

# In[26]:


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


# In[28]:


grp_mu_gexp   = [np.array([np.mean(grp_gexp[i][t],axis=0) for t in range(len(grp_gexp[i]))]).T
                         for i in range(len(grp_gexp))]
grp_std_gexp  = [np.array([np.std(grp_gexp[i][t],axis=0) for t in range(len(grp_gexp[i]))]).T
                         for i in range(len(grp_gexp))]
grp_sem_gexp  = [np.array([grp_std_gexp[i][:,t]/np.sqrt(grp_gexp[i][t].shape[1]) 
                           for t in range(len(grp_gexp[i]))]).T
                         for i in range(len(grp_gexp))]


###############################################################
# In[127]: fig 5 ##############################################
###############################################################


plt.style.reload_library()
plt.style.use('one_col_fig')
taulims = [-3,124]

marg_ht = 2
#schem_ht = 6
tau_series_ht = 8
spc1_ht = 2
spc2_ht = 1

marg_wd = 1
tau_series_wd = 30

wds = np.array([
    marg_wd,
    tau_series_wd
])

wds = np.array(wds/np.min(wds),dtype='int')

cs = np.cumsum(wds) # starting cols 
nc = np.sum(wds)

schem_ht = int(schem_dy/schem_dx*nc)

# row heights
hts = np.array([
    
    marg_ht,
    schem_ht,   
    spc1_ht,
    tau_series_ht,
    spc2_ht,
    tau_series_ht
    
])

hts = np.array(hts/np.min(hts),dtype='int')

rs = np.cumsum(hts) # starting rows
nr = np.sum(hts)

wid = 8.7/2.54
ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=200) 

gs = gridspec.GridSpec(nr, nc)

axA  = plt.subplot( gs[rs[0]:rs[1],:]) # neut_dev_schem
axB  = plt.subplot( gs[rs[2]:rs[3], cs[0]:]) # clusters
axC  = plt.subplot( gs[rs[4]:rs[5], cs[0]:]) # genes


caps = ['A','B','C']
ri   = [0,rs[2],rs[4]]
ci   = [0,0,0]
ys   = [-0.5,1,1]
xs   = [-4,-4,-4]
for i in range(len(caps)):
    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i], 
                x=xs[i],
                y=ys[i],fontsize=14,verticalalignment='top',horizontalalignment='right')
    cap_ax.axis('off')
    
#####################################
## A: neutrophil dev schematic     ##
#####################################
taulims = [-3,124]

axA.set_xticks([])
axA.set_yticks([])
axA.set_ylim(0,schem_dy)
axA.set_xlim(0,schem_dx)
axA.imshow(haem_dev_schem,extent=[-0.2*schem_dx,1.03*schem_dx,
                                  -0.15*schem_dy,1.08*schem_dy],clip_on=False)
#axA.imshow(haem_dev_schem)
#axA.margins(0)

axA.axis('off')

#####################################
## B: cell type fraction           ##
#####################################

cols = ['red','blue','green']
col_idx = 0
lab_dict = {'Undifferentiated':'Pluripotent','Neutrophil':'Fate-committed'}
for i in np.arange(len(ctypes)):
    if max(type_denss_neut[:,i]>=0.1):
        axB.plot(type_denss_neut[:,i], label=lab_dict[ctypes[i]], lw=1.5, color = cols[col_idx])
        col_idx += 1

axB.set_ylabel('frac. cells')
leg = axB.legend(labelspacing=0,frameon=False, loc=(0.01,0.3), handletextpad=0.1)
axB.set_xticklabels([])
axB.set_xlim(*taulims)
axB.set_yticks([0,0.5,1])

#####################################
## C: myeloid gene expression      ##
#####################################

# other version which includes promyelocite genes
grps = [4,3]
cols = ['goldenrod','dodgerblue']
marks = [['<','^','v'],['o','s']]
msz = [2,2,2]
fs = ['none','none','none']

for i in range(len(grps)):
    mu_gexp = grp_mu_gexp[grps[i]]
    ggnms = grp_gnms[grps[i]]
    for g in range(mu_gexp.shape[0]):
        axC.plot(np.arange(npsts),mu_gexp[g], marker = marks[i][g],markersize=msz[i],
             color=cols[i], label=ggnms[g], fillstyle=fs[i], markeredgewidth=0.5,linestyle='none') 
    
axC.set_yscale('symlog')
#axC.set_yscale('log')
axC.tick_params(axis='y', which='major', pad=0)
leg = axC.legend(labelspacing=0,frameon=False, loc=(0,0.25),ncol=2,columnspacing=0.4, handlelength=1,
                handletextpad=0.4)
axC.set_ylabel(r'$\langle$expr$\rangle$ (tpm)',labelpad=-2.5)

for i,text in zip(range(5),leg.get_texts()):
    plt.setp(text, color = cols[0] if i<len(marks[0]) else cols[1])

# both versions, x-axis
axC.set_xlim(*taulims)
axC.set_xlabel(r'pseudotime ($\tau$)')
axC.text(s='myelocyte',x=0,y=0.85,color=cols[0],fontsize=8,fontweight='bold',transform=axC.transAxes)
axC.text(s='promyelocyte',x=0.3,y=0.85,color=cols[1],fontsize=8,fontweight='bold',transform=axC.transAxes)


bif_axs = [axB,axC]
zord = [0,0,1,0,0]
cols = ['orange','green']
bifts = np.array([t_bifurc,t_bifurc_pf])
lss = ['--','-.']
bifnms = [r'$\tau_{m}$',r'$\tau_{d}$']
bifxs = (bifts - taulims[0])/(taulims[1]-taulims[0]) #[0.8,0.6] #bifts / tf
for j in range(len(bifts)):
    axB.text(s=bifnms[j],x=bifxs[j],y=1,transform=axB.transAxes,color=cols[j],
             horizontalalignment = 'center', verticalalignment='bottom')
    for i in range(len(bif_axs)):
        bif_axs[i].axvline(bifts[j],color=cols[j], linestyle = lss[j], alpha=0.5,zorder=zord[i],lw=2)

plt.savefig('{0}/fig5_neut_gexp.pdf'.format(figdir), bbox_inches='tight')


# In[ ]:


#intentionally left blank


# In[202]:

###########################################
# w1 for different bin sizes
###########################################
bin_szs            = np.array([20,50,2000])
overlaps           = np.array(bin_szs/2,dtype='int')

last_full_bins    = np.array(np.floor(srt.shape[0]/overlaps)*overlaps, dtype='int') - bin_szs + overlaps
neut_pst_grpss    = [[srt[i:(i+bin_szs[j])] for i in range(0,last_full_bins[j],overlaps[j])] 
                     for j in range(bin_szs.shape[0])]
for j in range(bin_szs.shape[0]):
    neut_pst_grpss[j][-1] = np.union1d(neut_pst_grpss[j][-1], srt[last_full_bins[j]:])


# In[203]:


neut_pst_cidxss    = [[np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grpss[i]] 
                      for i in range(bin_szs.shape[0])]
npstss             = np.array([len(x) for x in neut_pst_cidxss])


# In[204]: eigenvalues at different bin sizes

# this can be run quickly so i'll leave it here
pst_eig1s = []
for i in range(bin_szs.shape[0]):
    pst_eig1s.append(np.zeros(npstss[i]))
    print('number of pseudotime bins: {0}'.format(npstss[i]))
    for t in range(npstss[i]):    
        if t%100==0:
            print('\tbin {0}'.format(t))
        pca   = PCA(n_components=1)
        pca.fit(gexp_lil[neut_pst_cidxss[i][t]].toarray())
        # plain ol pca
        pst_eig1s[i][t] = pca.explained_variance_[0]


# In[209]: could save it and load it later
#for i in range(bin_szs.shape[0]):
#    np.save('{0}/pst_eval1_bsz{1}_overlap{2}.npy'.format(datdir,bin_szs[i],overlaps[i]), pst_eig1s[i])
#
#
## In[242]:
#
#
#bin_szs     = np.array([20,50,100,200,500,1000,2000])
#overlaps    = np.array(bin_szs/2,dtype='int')
#pst_eig1ss  = [np.load('{0}/pst_eval1_bsz{1}_overlap{2}.npy'.format(datdir,bin_szs[i],overlaps[i])) 
#               for i in range(len(bin_szs))]
pst_eig1ss = pst_eig1s


# In[286]:


nc = np.array([5,10,20,50,100,200,500,1000])
pst_nc_eig1b = np.array([np.load('{0}/pst_nc_sample_ns20_bsz1000/ncell{1}.npy'.format(datdir,nc[i]))
                for i in range(len(nc))])
trange = np.load('{0}/pst_nc_sample_ns20_bsz1000/trange.npy'.format(datdir))


# In[ ]:


#nc2, tr2, pst_nc_eig1b = np.load('{0}/pst_nc_sample_eval1_ns20_bsz1000.npy'.format(datdir), allow_pickle=True)


# In[287]:


bifurc_t               = trange[np.argmax(pst_nc_eig1b,axis=1)]
bifurc_mag             = pst_nc_eig1b[:,t_bifurc-tr2[0]] #np.max(pst_nc_eig1b,axis=1)

nnc, nt, nsamp = pst_nc_eig1b.shape

mu_bifurc_t  = np.mean(bifurc_t,axis=1)
std_bifurc_t = np.std(bifurc_t,axis=1)
err_bifurc_t = std_bifurc_t/np.sqrt(nsamp)

mu_bifurc_mag  = np.mean(bifurc_mag,axis=1)
std_bifurc_mag = np.std(bifurc_mag,axis=1)
err_bifurc_mag = std_bifurc_mag/np.sqrt(nsamp)


######################################################
# gene expression distributions
######################################################

min_gexp = np.zeros(npsts)
max_gexp = np.zeros(npsts)
for t in range(npsts):
    if t%5==0:
        print(t)
    gexpt = gexp_lil[neut_pst_cidxs[t]].toarray()
    min_gexp[t] = np.amin(gexpt[gexpt>0])
    max_gexp[t] = np.amax(gexpt)


# In[54]:


nbin       = 500
eps        = 1
min_gexp_a = np.amin(min_gexp)
max_gexp_a = np.amax(max_gexp)
linbins    = np.hstack([[-min_gexp_a],np.linspace(min_gexp_a, max_gexp_a+eps, nbin)])
logbins    = np.hstack([[-min_gexp_a],np.logspace(np.log10(min_gexp_a), np.log10(max_gexp_a+eps), nbin)])
ncell      = bin_sz
ngene      = gexp_lil.shape[1]


# In[ ]:


ts          = np.array([0,90,95,100,105,120])
ts = np.arange(npsts)
lin_hists_g = np.zeros((len(ts),ngene,nbin))
log_hists_g = np.zeros((len(ts),ngene,nbin))
for t in range(ts.shape[0]):
    if t%10==0:
        print(t)
    gexpt = gexp_lil[neut_pst_cidxs[ts[t]]].toarray().T
    for i in range(ngene):
        lin_hists_g[t,i] = np.histogram(gexpt[i],bins=linbins)[0]
        log_hists_g[t,i] = np.histogram(gexpt[i],bins=logbins)[0]
    


######################################################
# In[302]: fig S5 ####################################
######################################################


plt.style.reload_library()
plt.style.use('one_col_fig')

tseries_ht = 8
spc_ht  = 5
distr_ht = 8

marg_wd = 5
distr_wd= 10
spc_wd = 3

# row heights
hts = np.array([
    
    tseries_ht,
    spc_ht,
    distr_ht,
    spc_ht,
    tseries_ht
])

wds = np.array([
    marg_wd,
    distr_wd,
    spc_wd,
    distr_wd
])

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols 

nr = np.sum(hts)
nc = np.sum(wds)

wid = 8.7/2.54
ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=100) 

gs   = gridspec.GridSpec(nr, nc)

axA  = plt.subplot( gs[0    :rs[0], cs[0]:cs[3]]) # different sized bins
#axB  = plt.subplot( gs[rs[1]:rs[2], cs[0]:cs[1]]) # time shuffled distributions
axB  = plt.subplot( gs[rs[1]:rs[2], cs[0]:cs[3]]) # bifurcation detection time
axC  = plt.subplot( gs[rs[3]:rs[4], cs[0]:cs[3]]) # distributional change over time...

# cap_axs = [axAC,axBC,axCC,axCC,axEC]
caps = ['A','B','C']
ci = [0,0,0]
ri = [0,rs[1],rs[3]]
yht = [0,0,1]
for i in range(len(caps)):
    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i],x=0,y=yht[i],fontsize=14)
    cap_ax.axis('off')
    

#####################################
## A: w1 for different bin sizes ####
#####################################
cols = plt.cm.viridis(np.linspace(0,1,len(bin_szs)))
for i in range(len(bin_szs)):
    axA.plot(np.linspace(0,neut_psts.shape[0],pst_eig1ss[i].shape[0]), pst_eig1ss[i],'-',
             color=cols[i], label=bin_szs[i])
    
leg = axA.legend(labelspacing=0,ncol=2,title='bin size',frameon=False, columnspacing=0.6,loc=(0.05,0.05),
           handlelength=0.4)
# for i,text in zip(range(len(cols)),leg.get_texts()):
#     plt.setp(text, color = cols[i])
for hand in leg.legendHandles:
    hand.set_lw(4)
    
axA.set_xlabel('pseudotime rank')
axA.set_ylabel(r'$\omega_1$')

axA.set_yticks(np.arange(0,60000,10000))
axA.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

axA.set_xticks(np.arange(0,62000,10000))
axA.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


# #####################################
# #### B: error with null #############
# #####################################

# axC.errorbar(np.arange(npsts),pst_eig1_n - pst_grp_null_eval_mu, yerr=pst_grp_null_eval_err, 
#              color='k', capsize=2,alpha=1, label='shuffled expression')
# axC.set_yscale('symlog')
# axC.set_xlim(70,123)
# axC.set_xlabel(r'$\tau$')

# ###############################################################
# #### B: ncells required for bifucation detection #############
# ##############################################################
axB2  = axB.twinx()
cols = ['b','r']
y2sc = 1000
fs = 20
axB.errorbar(nc2, mu_bifurc_t,    yerr=err_bifurc_t,     color = cols[0], capsize=2)
axB2.errorbar(nc2, mu_bifurc_mag, yerr=err_bifurc_mag, color = cols[1], capsize=2)

axB2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


axB.tick_params(axis='y', labelcolor=cols[0])
axB2.tick_params(axis='y', labelcolor=cols[1])

axB.set_xlabel('number of cells')
axB.set_ylabel(r'$\tau_{sn}$', color = cols[0])
axB2.set_ylabel(r'$\omega_1(\tau_{sn})$', rotation=270, color = cols[1], labelpad=10)

axB.axhline(t_bifurc,   color = cols[0], linestyle='--', lw=1)
axB2.axhline(mag_bifurc, color = cols[1], linestyle='--', lw=1)

# reverse right axis so data doesn't collide
axB.set_xscale('log')
axB2.set_ylim(axB2.get_ylim()[::-1])

axB2.set_yticks([20000,30000,40000])
axB.set_yticks(np.arange(108,114,2))

#####################################
#### C: distributions #############
#####################################
log_bin_ctrs = 0.5*(logbins[1:]+logbins[:-1])
ts = np.array([0,90,95,100,105,120])
cols = plt.cm.viridis(np.linspace(0,1, len(ts)))
for i in range(ts.shape[0]):
    axC.plot(log_bin_ctrs,log_hists_ga[ts[i]]/(neut_pst_cidxs[ts[i]].shape[0]),'o',
             color=cols[i],label=ts[i],alpha=1,markersize=1)
    
axC.set_yscale('symlog',linthresh=1e-2)
axC.set_xscale('symlog')
axC.set_xlabel('gene expression per cell')
axC.set_ylabel(r'$\langle$# genes$\rangle_{\rm cell}$')
leg= axC.legend(loc=(0.17,0.05),labelspacing=0, ncol=2,handletextpad=0.05,frameon=False,columnspacing=0,
                title=r'$\tau$')
for hand in leg.legendHandles:
    hand._legmarker.set_markersize(4)

leg._legend_box.sep = 0.1
# axC.text(x=0.17,y=0.25,s= r'$\tau$',transform=axC.transAxes)
axC.set_ylim(-0.003,50)

plt.savefig('{0}/figS5_neut_cov_supp.pdf'.format(figdir), bbox_inches='tight')