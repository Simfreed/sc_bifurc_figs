import numpy as np
import scipy as scipy
from scipy import io as scio
import pickle
import sys
import copy
import myfun as mf
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')

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

headdir    = '.' #'/Users/simonfreedman/cqub/bifurc/weinreb_2020/'
figdir     = '{0}/figs'.format(headdir)
datdir     = '{0}/neutrophil_data'.format(headdir)
eigdir     = '{0}/eig'.format(datdir)
ncelldir   = '{0}/eigs_ncell_sample'.format(datdir)

gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)
gnm_fname  = '{0}/gene_names.txt'.format(datdir)

# In[6]:

print('loading gene expression matrix')
gexp_fname = '{0}/gene_expr.npz'.format(datdir)
gexp_sp    = scipy.sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds


# In[11]:

print('loading neutrophil pseudotime ranking')
neut_psts = np.genfromtxt(pst_fname, skip_header=True, dtype='int')


# In[12]:

print('cells in each pseudotime bin of width 1000')
bin_sz            = 1000
overlap           = int(bin_sz/2)

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npsts             = len(neut_pst_cidxs)

###########################################
# In[286]: different bin sizes ############
###########################################


print('different pseudotime bin sizes')

bin_szs     = np.array([20,50,100,200,500,1000,2000])
overlaps    = np.array(bin_szs/2,dtype='int')
pst_eig1ss  = [np.load('{0}/eigs_bin_size/evals_bsz{1}.npy'.format(datdir,bin_szs[i])) 
               for i in range(len(bin_szs))]

t_bifurc    = np.argmax(pst_eig1ss[5])
mag_bifurc  = np.amax(pst_eig1ss[5])

###########################################
# In[286]: sampling different numbers of cells per bin
###########################################

print('sampling different numbers of cells per bin')
ncell = np.array([5,10,20,50,100,200,500,1000])
pst_nc_eig1b = np.array([np.load('{0}/ncell{1}.npy'.format(ncelldir,nc))[...,0] for nc in ncell])
trange = np.load('{0}/trange.npy'.format(ncelldir))

# In[287]:


bifurc_t               = trange[np.argmax(pst_nc_eig1b,axis=1)]
bifurc_mag             = pst_nc_eig1b[:,t_bifurc-trange[0]] #np.max(pst_nc_eig1b,axis=1)

nnc, nt, nsamp = pst_nc_eig1b.shape

mu_bifurc_t  = np.mean(bifurc_t,axis=1)
std_bifurc_t = np.std(bifurc_t,axis=1)
err_bifurc_t = std_bifurc_t/np.sqrt(nsamp)

mu_bifurc_mag  = np.mean(bifurc_mag,axis=1)
std_bifurc_mag = np.std(bifurc_mag,axis=1)
err_bifurc_mag = std_bifurc_mag/np.sqrt(nsamp)


######################################################
# gene expression distributions at critical times ###
######################################################
print('distribution of gene expression at different points in pseudotime')
print('computing min max')
#ts       = np.arange(npsts)
ts       = np.array([0,90,95,100,105,120])
nts = ts.shape[0]
min_gexp = np.zeros(nts)
max_gexp = np.zeros(nts)
for t in range(nts):
    #if t%100==0:
    print('\t t={0}'.format(ts[t]))
    gexpt = gexp_lil[neut_pst_cidxs[ts[t]]].toarray()
    min_gexp[t] = np.amin(gexpt[gexpt>0])
    max_gexp[t] = np.amax(gexpt)


# In[54]:


nbin       = 500
eps        = 1
min_gexp_a = np.amin(min_gexp)
max_gexp_a = np.amax(max_gexp)
linbins    = np.hstack([[-min_gexp_a],np.linspace(min_gexp_a, max_gexp_a+eps, nbin)])
logbins    = np.hstack([[-min_gexp_a],np.logspace(np.log10(min_gexp_a), np.log10(max_gexp_a+eps), nbin)])
ngene      = gexp_lil.shape[1]


# In[ ]:


#lin_hists_g = np.zeros((len(ts),ngene,nbin))
log_hists_g = np.zeros((len(ts),ngene,nbin))
print('histogramming')
for t in range(nts):
    #if t%100==0:
    print('\t t={0}'.format(ts[t]))
    gexpt = gexp_lil[neut_pst_cidxs[ts[t]]].toarray().T
    for i in range(ngene):
 #       lin_hists_g[t,i] = np.histogram(gexpt[i],bins=linbins)[0]
        log_hists_g[t,i] = np.histogram(gexpt[i],bins=logbins)[0]
    
log_hists_ga = np.sum(log_hists_g,axis=1)

######################################################
# In[302]: fig S5 ####################################
######################################################

print('generating figure s5')


plt.style.reload_library()
plt.style.use('one_col_fig.mplstyle')

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

# ###############################################################
# #### B: ncells required for bifucation detection #############
# ##############################################################
axB2  = axB.twinx()
cols = ['b','r']
y2sc = 1000
fs = 20
axB.errorbar(ncell, mu_bifurc_t,    yerr=err_bifurc_t,     color = cols[0], capsize=2)
axB2.errorbar(ncell, mu_bifurc_mag, yerr=err_bifurc_mag, color = cols[1], capsize=2)

axB2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))


axB.tick_params(axis='y', labelcolor=cols[0])
axB2.tick_params(axis='y', labelcolor=cols[1])

axB.set_xlabel('number of cells')
axB.set_ylabel(r'$\tau_m$', color = cols[0])
axB2.set_ylabel(r'$\omega_1(\tau_m)$', rotation=270, color = cols[1], labelpad=10)

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
#ts = np.array([0,90,95,100,105,120])
cols = plt.cm.viridis(np.linspace(0,1,nts))
for i in range(nts):
    axC.plot(log_bin_ctrs,log_hists_ga[i]/(neut_pst_cidxs[ts[i]].shape[0]),'o',
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

print('saved figure s5')
