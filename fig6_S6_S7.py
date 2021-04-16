#!/usr/bin/env python

import numpy as np
import scipy as scipy
from scipy import io as scio
import pickle
import sys
import copy
import myfun as mf
import networkx as nx

import matplotlib
from matplotlib import gridspec, rc
from matplotlib.ticker import LogLocator
import matplotlib.pyplot as plt

rc('text',usetex=False)

# Needs: 
# pseudotime trajectory
# eigenvalues as function of time
# gexp for neutrophil genes
# gexp for myelocite genes


headdir    = '.' #'/Users/simonfreedman/cqub/bifurc/weinreb_2020/'
figdir     = '{0}/figs'.format(headdir)
datdir     = '{0}/neutrophil_data'.format(headdir)
eigdir     = '{0}/eig'.format(datdir)

gexp_fname = '{0}/gene_expr.npz'.format(datdir)
pst_fname  = '{0}/pseudotime.txt'.format(datdir)
gnm_fname  = '{0}/gene_names.txt'.format(datdir)
meta_fname = '{0}/metadata.txt'.format(datdir)

# In[6]:

print('loading gene expression matrix')
gexp_sp    = scipy.sparse.load_npz(gexp_fname) # WT: 18.3 seconds
gexp_lil   = gexp_sp.tolil() # WT: 3 min 55 seconds

# In[4]:
gnms     = np.genfromtxt(gnm_fname,dtype='str')


# In[5]:
neut_psts = np.genfromtxt(pst_fname, skip_header=True, dtype='int')
bin_sz            = 1000
overlap           = int(bin_sz/2)

srt               = np.argsort(neut_psts[:,1])
last_full_bin     = int(np.floor(srt.shape[0]/overlap)*overlap) - bin_sz + overlap
neut_pst_grps     = [srt[i:(i+bin_sz)] for i in range(0,last_full_bin,overlap)]
neut_pst_grps[-1] = np.union1d(neut_pst_grps[-1], srt[last_full_bin:])
neut_pst_cidxs    = [np.array(neut_psts[grp,0], dtype = 'int') for grp in neut_pst_grps]
npsts             = len(neut_pst_cidxs)

# In[6]:

pst_eig1 = np.load('{0}/eig/dat_eval.npy'.format(datdir))[:,0]
tf       = pst_eig1.shape[0]
bift     = np.argmax(pst_eig1)
ti       = 2*bift - tf


# In[8]:
corr_dir       = '{0}/corr/min_nc9_gexp_0'.format(datdir)
binmin         = -1
binmax         = 1
dbin           = 0.05
corr_bin_edges = np.arange(binmin-dbin/2,binmax+dbin,dbin)
corr_bin_ctrs  = 0.5*(corr_bin_edges[1:]+corr_bin_edges[:-1])
corr_nbin      = corr_bin_ctrs.shape[0]

nc_thresh    = 400
thresh_hists = np.zeros((tf-ti, corr_nbin))

for t in range(ti,tf):
    corrs                = np.load('{0}/pccs_{1}.npy'.format(    corr_dir, t))
    gidxs_nc             = np.load('{0}/gidxs_nc_{1}.npy'.format(corr_dir, t))
    corr_idxs            = np.where(gidxs_nc[:,2]>=nc_thresh)[0]
    thresh_hists[t-ti]   = np.histogram(corrs[corr_idxs], bins = corr_bin_edges, density=True)[0]

# gene graph

t = bift
corr_hi_thresh = 0.65
corr_lo_thresh = -0.3
corrs_bi    = np.load('{0}/pccs_{1}.npy'.format(corr_dir, t))
gidxs_nc_bi = np.load('{0}/gidxs_nc_{1}.npy'.format(corr_dir, t))


# In[14]:
hi_pos_corr_bi_cidxs = np.where((corrs_bi>corr_hi_thresh) & (gidxs_nc_bi[:,2]>=nc_thresh))[0]
hi_neg_corr_bi_cidxs = np.where((corrs_bi<corr_lo_thresh) & (gidxs_nc_bi[:,2]>=nc_thresh))[0]
hi_corr_bi_cidxs     = np.hstack([hi_pos_corr_bi_cidxs,hi_neg_corr_bi_cidxs])

hi_pos_corr_bi_gidxs = gidxs_nc_bi[hi_pos_corr_bi_cidxs,0:2]
hi_neg_corr_bi_gidxs = gidxs_nc_bi[hi_neg_corr_bi_cidxs,0:2]
hi_corr_bi_gidxs     = np.vstack([hi_pos_corr_bi_gidxs, hi_neg_corr_bi_gidxs])

hi_corr_bi_gidxs_unq = np.unique(hi_corr_bi_gidxs.reshape(-1))



# In[19]:
gidx_dict = {gnms[i].lower():i for i in range(len(gnms))}

# In[55]:

# genes listed in weinreb2020

neut_gnms        = np.array(['S100a9', 'Itgb2l', 'Elane', 'Fcnb', 'Mpo', 'Prtn3', 
                              'S100a6', 'S100a8', 'Lcn2', 'Lrg1'])
mpp_gnms         = np.array(['Ly6a','Meis1','Flt3','Cd34'])
gpp_gnms         = np.array(['Csf1r','Cebpa'])
pmy_gnms         = np.array(['Gfi1','Elane'])
my_gnms          = np.array(['S100a8','Ngp','Ltf'])

weinreb_gnms  = set([k.lower() for k in mf.flatten2d([neut_gnms, mpp_gnms, gpp_gnms, pmy_gnms, my_gnms])])



# In[48]:
housekeeping = ['cstb','ctsd','fth1']
myeloid      = ['ngp', 's100a8'] 
neutrophil   = ['s100a6','s100a9','lcn2','ccl6'] 
membrane     = ['rab7','anxa4','cd9']
metabolism   = ['psap','laptm5','sat1','gstm1','sqstm1','ctsb','ftl1','sirpa']
sig_dev      = ['sdcbp','bri3','plin2','cybb','srgn']
mitochon     = ['mt-cytb','mt-atp6','mt-co2']
misc         = ['h3f3a','mpeg1','h2-d1']

gtype_nms = ['membrane','metabolism', 'neutrophil','housekeep','mitochondria', 
          'misc', 'signalling', 'myelocyte']
gtypes = [membrane, metabolism, neutrophil, housekeeping, mitochon, misc, sig_dev, myeloid]

gtype_dict = {}
gtype_gidx_dict = {}
for i in range(len(gtypes)):
    for gn in gtypes[i]:
        gtype_dict[gn] = i

# In[49]:

#gnms_net = nx.Graph()
#gnms0   = [x.lower() for x in gnms[gidxs_nc_bi[hi_corr_bi_cidxs,0]]]
#gnms1   = [x.lower() for x in gnms[gidxs_nc_bi[hi_corr_bi_cidxs,1]]]
#edg_wts       = corrs_bi[hi_corr_bi_cidxs]
#edg_wts_normd = (edg_wts - np.min(edg_wts)) / (np.amax(edg_wts) - np.amin(edg_wts))
#
#for i in range(hi_corr_bi_cidxs.shape[0]):
#    gnms_net.add_edge(gnms0[i],gnms1[i],weight=edg_wts_normd[i], corr = edg_wts[i])


# In[50]:


gnms_net = nx.Graph()
gnms0   = [x.lower() for x in gnms[gidxs_nc_bi[hi_corr_bi_cidxs,0]]]
gnms1   = [x.lower() for x in gnms[gidxs_nc_bi[hi_corr_bi_cidxs,1]]]

edg_wts       = corrs_bi[hi_corr_bi_cidxs]
pos_corr_idxs = np.where(edg_wts  > 0)[0]
neg_corr_idxs = np.where(edg_wts <= 0)[0]

edg_wts_pos   = edg_wts[pos_corr_idxs]
edg_wts_neg   = edg_wts[neg_corr_idxs]

edg_wts_pos_normd = (edg_wts_pos - np.min(edg_wts_pos)) / (np.amax(edg_wts_pos) - np.amin(edg_wts_pos))
edg_wts_neg_normd = (edg_wts_neg - np.min(edg_wts_neg)) / (np.amax(edg_wts_neg) - np.amin(edg_wts_neg))

edg_wts_normd = np.zeros(edg_wts.shape[0])
for i in range(len(pos_corr_idxs)):
    edg_wts_normd[pos_corr_idxs[i]] = edg_wts_pos_normd[i]
for i in range(len(neg_corr_idxs)):
    edg_wts_normd[neg_corr_idxs[i]] = edg_wts_neg_normd[i]

for i in range(hi_corr_bi_cidxs.shape[0]):
    gnms_net.add_edge(gnms0[i],gnms1[i],weight=edg_wts_normd[i], corr = edg_wts[i])


# In[91]:

cols    = plt.cm.tab10.colors
node_cols =[]
edge_cols =[]
edge_widths =[]

node_cols_d = {}
edge_cols_d = {}
edge_widths_d = {}
for node in gnms_net.nodes():
    node_cols.append(cols[gtype_dict[node]])
    node_cols_d[node] = cols[gtype_dict[node]]
    

wt_fac = 1
wt_min = 0.1
for edge in gnms_net.edges():
    edge_corr = gnms_net.get_edge_data(edge[0],edge[1])['corr']
    edge_wt   = gnms_net.get_edge_data(edge[0],edge[1])['weight']
    
    col = 'red' if edge_corr < 0 else 'blue'
    wd  = edge_wt*wt_fac+wt_min
    
    edge_cols.append(col)
    edge_widths.append(wd)
    
    edge_cols_d[edge] = col
    edge_widths_d[edge] = wd
    

node_grp_posx = np.array([0.28]*4+[0.72]*4)
node_grp_posy = np.array([0.05,0.35,0.65,0.95]*2)
node_grp_rads = np.array([0.07+0.015*(len(grp)-3) for grp in gtypes])
node_grp_angs = [np.linspace(0,2*np.pi,len(grp)+1)-np.pi/4 for grp in gtypes]
node_grp_rads[4]=0.09 #big words
node_pos = {}
for i in range(len(gtypes)):
    xc = node_grp_posx[i]
    yc = node_grp_posy[i]
    r  = node_grp_rads[i]
    th = node_grp_angs[i]
    for j in range(len(gtypes[i])):
        node_pos[gtypes[i][j]] = np.array([xc + r*np.cos(th[j]), yc + r*np.sin(th[j])]) 
        
gxmin,gymin = np.amin(np.array(list(node_pos.values())),axis=0)
gxmax,gymax = np.amax(np.array(list(node_pos.values())),axis=0)
gdx, gdy = gxmax-gxmin, gymax-gymin

gtxt_col_d = {}
cols = plt.cm.tab10.colors
for i in range(len(gtypes)):
    for k in range(len(gtypes[i])):
        gnm  = gtypes[i][k]
        #gtxt_col_d[gnm] = cols[k] # different colors for each gene in the same group
        gtxt_col_d[gnm] = cols[i] # different colors for each group
        


# In[95]:

print('making fig 6')

plt.style.reload_library()
plt.style.use('one_col_fig')

marg_ht = 1
leg_ht  = 1
leg_spc = 2
hist_ht = 15
grph_ex_ht = 2

marg_wd = 1
col1_wd = 10
spc_wd  = 1
col2_wd = 19

# row heights
hts = np.array([
    
    marg_ht,
    leg_ht,
    leg_spc,
    hist_ht,
    grph_ex_ht

])

wds = np.array([
    marg_wd,
    col1_wd,
    spc_wd,
    col2_wd,
])

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols 

nr = np.sum(hts)
nc = np.sum(wds)

wid = 11.4 / 2.54 #17.8/2.54
ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=200) 

gs = gridspec.GridSpec(nr, nc)

axAL = plt.subplot( gs[rs[0]:rs[1],  cs[0]:cs[1]]) # corr hist heat map legend
axA  = plt.subplot( gs[rs[2]:rs[3], cs[0]:cs[1]]) # corr hist

axB  = plt.subplot( gs[rs[0]:, cs[2]:cs[3]])      # gene graph

caps = ['A','B']
ri   = [0,0]
ci   = [0,cs[2]]
ys   = [-0.5,-0.5]
xs   = [-3,0]
for i in range(len(caps)):
    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i], 
                x=xs[i],
                y=ys[i],fontsize=14,verticalalignment='top',horizontalalignment='left')
    cap_ax.axis('off')

#####################################
## A: correlation distribution   ####
#####################################

cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap.set_bad('white')

hist_min = 1e-4
hist_max = 6
ti = ti

corr_hist_idxs = np.where((corr_bin_ctrs<0.85)&(corr_bin_ctrs>-0.6))[0]

im = axA.imshow(thresh_hists[:,th_idx,corr_hist_idxs].T, cmap=cmap,aspect='auto',
                norm=matplotlib.colors.LogNorm(vmin=hist_min,vmax=hist_max))

axA.set_xlabel('pseudotime')
axA.set_ylabel('correlation coefficient')
tskip = 7
axA.set_xticks(np.arange(0,tf-ti,tskip))
axA.set_xticklabels(np.arange(ti,tf,tskip))
bskip = 4
# axA.set_yticks(np.arange(0,corr_bin_ctrs.shape[0],bskip))
# axA.set_yticklabels(['{0:.2f}'.format(i) for i in corr_bin_ctrs[::bskip]])
axA.set_yticks(np.arange(0,corr_hist_idxs.shape[0],bskip))
axA.set_yticklabels(['{0:.2f}'.format(i) for i in corr_bin_ctrs[corr_hist_idxs][::bskip]])
axA.axvline(bift-ti, linestyle='--',color='k')

cbar = fig.colorbar(im, cax=axAL, orientation='horizontal', aspect=1)
cbar.set_label('frequency',rotation=0,labelpad=2)
axAL.xaxis.set_label_position('top')

locmaj = matplotlib.ticker.LogLocator(base=10,numticks=3) 
axAL.xaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=100)
axAL.xaxis.set_minor_locator(locmin)

#####################################
## B: gene graph                 ####
#####################################

bkg_col = 'bisque'
transf = axB.transData.inverted()

props    = dict(boxstyle='round', facecolor='white', alpha=1)
grp_bnds = np.zeros((len(gtypes),2,2))
grp_bnds[:,:,0] =  np.inf
grp_bnds[:,:,1] = -np.inf

xoffs = 0.035*np.array([-1,-1,-1,-1,
                       1,1,1,1])
#plot nodes, keep track of min and max position of group
for node in gnms_net.nodes():
    props['facecolor'] = gtxt_col_d[node]
    props['edgecolor'] = 'k' if node in weinreb_gnms else 'none'
    
    x,y = node_pos[node]
    m=axB.text(x,y, node.capitalize(), fontsize=6, bbox=props, color='white', #gtxt_col_d[node],
             horizontalalignment='center',verticalalignment='center', zorder=2)
    
rots = [0,90,90,0,0,270,270,0]
gtype_nms = ['membrane','metabolism', 'neutrophil','housekeep','mitochondria', 'misc', 'signalling', 'myelocyte']
gtxtx = np.array([ 0.25,        0.025,         0.09,       0.25,          0.75,   0.95,         0.97,      0.75])
gtxty = np.array([0.01,            0.35,       0.66,       1,             0,   0.35,         0.65,      0.98])
haligns = [    'center',      'left',       'left',   'center',      'center', 'right',     'right',  'center']
valigns = [    'bottom',    'center',     'center',      'top',      'bottom','center',    'center',     'top']

for i in range(len(gtypes)):
    
    axB.text(s = gtype_nms[i], x=gtxtx[i], y = gtxty[i], fontsize=7,#bbox=props,
        horizontalalignment=haligns[i],verticalalignment=valigns[i],rotation=rots[i],zorder=1, color=cols[i],
            transform=axB.transAxes)

xlims = [gxmin-gdx/6,gxmax+gdx/6]
ylims = [gymin-gdy/12,gymax+gdy/12]


axB.set_xlim(*xlims)
axB.set_ylim(*ylims)

# edges
for edge in gnms_net.edges:
    g0,g1 = edge
    x0,y0 = node_pos[g0]
    x1,y1 = node_pos[g1]
    axB.plot([x0,x1],[y0,y1], c=edge_cols_d[edge], lw = edge_widths_d[edge], zorder=1)
    
axB.set_xticks([])
axB.set_yticks([])

plt.savefig('{0}/neut_corr.pdf'.format(figdir), bbox_inches='tight')

print('saved fig 6')
# In[69]:

ti = 90
tf = npsts

gtype_gidxs = [[gidx_dict[gtype[k]] for k in range(len(gtype))] for gtype in gtypes]
gexp_hi_corr = []
for gidxs in gtype_gidxs:
    gexp_hi_corr.append([np.hstack([gexp_lil[neut_pst_cidxs[t],gidx].toarray() 
                   for gidx in gidxs])
                  for t in range(ti,tf)]
    )


# In[144]:


# first index is cc
# second index is pseudotime
# third is cell
# fourth is gene
thresh = 0
gexp_hi_corr_mu = [np.array([[mf.meanor0(gexp_t[gexp_t>thresh]) 
                              for gexp_t in gexp_hi_corr[i][t].T]
                             for t in range(tf-ti)])
                   for i in range(len(gexp_hi_corr))]
gexp_hi_corr_std = [np.array([[mf.stdor0(gexp_t[gexp_t>thresh]) 
                              for gexp_t in gexp_hi_corr[i][t].T]
                             for t in range(tf-ti)])
                   for i in range(len(gexp_hi_corr))]
gexp_hi_corr_sem = [np.array([[mf.semor0(gexp_t[gexp_t>thresh])
                              for gexp_t in gexp_hi_corr[i][t].T]
                             for t in range(tf-ti)])
                   for i in range(len(gexp_hi_corr))]
# gexp_hi_corr_std = [np.array([[mf.stdor0(gexp_t[gexp_t>thresh]) for gexp_t in gexp_hi_corr[i][j]]
#                              for j in range(len(gexp_hi_corr[i]))])
#                    for i in range(len(gexp_hi_corr))]


# In[28]:


evec1 = np.load('{0}/eig/dat_evec.npy'.format(datdir))[:,0]
hi_g_idx = np.unravel_index(np.argmax(np.abs(evec1)), evec1.shape)[1]
evec1_nn = (np.sign(evec1[:,hi_g_idx])*evec1.T).T


# In[29]:


# gexp_hi_corr = [np.array([gexp_lil[neut_pst_cidxs[t]][:,cc_grps[i]].toarray() 
#                            for t in range(npsts)]) for i in range(len(cc_grps))]

# gexp_hi_corr_mu  = [np.zeros((len(cc_grps[i]),npsts)) for i in range(len(cc_grps))]
# gexp_hi_corr_std = [np.zeros((len(cc_grps[i]),npsts)) for i in range(len(cc_grps))]

# for i in range(len(cc_grps)):
#     for t in range(npsts):
#         gexp_t = gexp_hi_corr[i][t]
#         for g in range(gexp_t.shape[1]):
#             nz_idxs = np.where(gexp_t[:,g]>0)[0]
#             gexp_hi_corr_mu[i][g,t]  = mf.meanor0(gexp_t[nz_idxs,g])
#             gexp_hi_corr_std[i][g,t] = mf.stdor0(gexp_t[nz_idxs,g])


#axB.set_ymargin(0)
#axB.set_constrained_layout_pads(h_pad=-3)


# In[67]:


'hello'.capitalize()


# In[59]:


print(axB.set_ymargin.__doc__)


# In[359]:


# intentionally left blank


# In[79]:


plt.style.reload_library()
plt.style.use('one_col_fig')
# nr = 90
# nc = 45

leg_ht  = 3
leg_spc = 6
spc_ht  = 2
spc2_ht = 10

schem_ht   = 30
tau_series_ht = 15

col1_wd = 35
col2_wd = 60
spc0_wd = 5
spc1_wd = 2
spc2_wd=12
spc3_wd=2
tau_series_wd = 29



# row heights
hts = np.array([
    
    leg_ht,
    leg_spc,
    
    tau_series_ht - leg_spc,
    spc_ht,
    tau_series_ht,
    spc_ht,
    tau_series_ht,
    spc_ht,
    tau_series_ht,
    spc_ht,
    tau_series_ht,
    spc2_ht
])

wds = np.array([
    spc0_wd,
    col1_wd,
    spc1_wd,
    col2_wd,
    spc2_wd,
    tau_series_wd,
    spc3_wd,
    tau_series_wd
])

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols 

nr = np.sum(hts)
nc = np.sum(wds)

wid = 17.8/2.54
ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=200) 

gs = gridspec.GridSpec(nr, nc)

axAL = plt.subplot( gs[0    :rs[0],  cs[0]:cs[1]]) # corr hist heat map legend
axA  = plt.subplot( gs[rs[1]:rs[10], cs[0]:cs[1]]) # corr hist

axB  = plt.subplot( gs[rs[0]:, cs[2]:cs[3]])      # gene graph

trs0 = [rs[0], rs[3], rs[5], rs[7], rs[9]]
trsF = [rs[2], rs[4], rs[6], rs[8], rs[10]]

##### gtypes = [housekeeping, myeloid, neutrophil, membrane, metabolism, sig_dev, mitochon, misc]
##### gtypes = [membrane, metabolism, neutrophil, housekeeping, mitochon, misc, sig_dev, myeloid]

gtype_plt_grps = [2,7,6,5,1]
ngrp = len(gtype_plt_grps)
axC  = [plt.subplot( gs[trs0[i]:trsF[i], cs[4]:cs[5]]) for i in range(ngrp)] # gene expression
axD  = [plt.subplot( gs[trs0[i]:trsF[i], cs[6]:cs[7]]) for i in range(ngrp)] # v1

capd_axs = [axAL, axB,axC[0],axD[0]]
caps = ['A','B','C','D']
cap_xs = [-4,1,-6,-4]


for i in range(len(capd_axs)):
    r = capd_axs[i].get_subplotspec().rowspan[0]
    c = capd_axs[i].get_subplotspec().colspan[0]
    if i > 0:
        r-=leg_ht
    else:
        c=c-cs[0]
    #print(i,r,c)
    
    cap_ax=plt.subplot(gs[r:r+1,c:c+1])
    cap_ax.text(s=caps[i],
                x=cap_xs[i],
                y=-1,fontsize=14)
    cap_ax.axis('off')
    

#####################################
## A: correlation distribution   ####
#####################################

cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap.set_bad('white')

hist_min = 1e-4
hist_max = 6
ti = ti

corr_hist_idxs = np.where((corr_bin_ctrs<0.85)&(corr_bin_ctrs>-0.6))[0]

im = axA.imshow(thresh_hists[:,th_idx,corr_hist_idxs].T, cmap=cmap,aspect='auto',
                norm=matplotlib.colors.LogNorm(vmin=hist_min,vmax=hist_max))

axA.set_xlabel('pseudotime')
axA.set_ylabel('correlation coefficient')
tskip = 7
axA.set_xticks(np.arange(0,tf-ti,tskip))
axA.set_xticklabels(np.arange(ti,tf,tskip))
bskip = 4
# axA.set_yticks(np.arange(0,corr_bin_ctrs.shape[0],bskip))
# axA.set_yticklabels(['{0:.2f}'.format(i) for i in corr_bin_ctrs[::bskip]])
axA.set_yticks(np.arange(0,corr_hist_idxs.shape[0],bskip))
axA.set_yticklabels(['{0:.2f}'.format(i) for i in corr_bin_ctrs[corr_hist_idxs][::bskip]])
axA.axvline(bift-ti, linestyle='--',color='k')

cbar = fig.colorbar(im, cax=axAL, orientation='horizontal', aspect=1)
cbar.set_label('frequency',rotation=0,labelpad=2)
axAL.xaxis.set_label_position('top')

locmaj = matplotlib.ticker.LogLocator(base=10,numticks=3) 
axAL.xaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=100)
axAL.xaxis.set_minor_locator(locmin)

#####################################
## B: gene graph                 ####
#####################################

bkg_col = 'bisque'
transf = axB.transData.inverted()

props    = dict(boxstyle='round', facecolor='white', alpha=1)
grp_bnds = np.zeros((len(gtypes),2,2))
grp_bnds[:,:,0] =  np.inf
grp_bnds[:,:,1] = -np.inf

xoffs = 0.035*np.array([-1,-1,-1,-1,
                       1,1,1,1])
#plot nodes, keep track of min and max position of group
for node in gnms_net.nodes():
    props['facecolor'] = gtxt_col_d[node]
    props['edgecolor'] = 'k' if node in weinreb_gnms else 'none'
    
    x,y = node_pos[node]
    m=axB.text(x,y, node, fontsize=6, bbox=props, color='white', #gtxt_col_d[node],
             horizontalalignment='center',verticalalignment='center', zorder=2)
  
    bb            = m.get_window_extent(renderer = fig.canvas.get_renderer())
    bb_datacoords = bb.transformed(transf)
    grp_idx        = gtype_dict[node]
    
    grp_bnds[grp_idx,0,0] = np.amin([grp_bnds[grp_idx,0,0],bb_datacoords.x0])
    grp_bnds[grp_idx,0,1] = np.amax([grp_bnds[grp_idx,0,1],bb_datacoords.x1])
    grp_bnds[grp_idx,1,0] = np.amin([grp_bnds[grp_idx,1,0],bb_datacoords.y0])
    grp_bnds[grp_idx,1,1] = np.amax([grp_bnds[grp_idx,1,1],bb_datacoords.y1])



props    = dict(boxstyle='round', facecolor=bkg_col, alpha=1, edgecolor='none')

# [left, right] padding
xpads = np.array([[-0.01,-0.02],[0.025,-0.02],[0.02,-0.01],[0,0],
                  [-0.01,-0.01],[-0.01,0.01],[0,0.02],[-0.01,-0.01]])

# [below, above] padding
ypads = np.array([[0.05,0.02],[0.02,0.02],[0.04,0.04  ],[0.0,0.06],
                  [0.06,0.02],[0.02,0.02],[0.02,0.02],[0.02,0.053]])
#ypads = np.zeros(8)

rect_wid = grp_bnds[:,0,1]-grp_bnds[:,0,0]+np.sum(xpads,axis=1)
rect_ht  = grp_bnds[:,1,1]-grp_bnds[:,1,0]+np.sum(ypads,axis=1)

anchx   = grp_bnds[:,0,0]-xpads[:,0]
anchy   = grp_bnds[:,1,0]-ypads[:,0]

rect_midx = 0.5*(grp_bnds[:,0,1]+grp_bnds[:,0,0])
rect_midy = 0.5*(grp_bnds[:,1,1]+grp_bnds[:,1,0])

rots = [0,90,90,0,
       0,270,270,0]

gtxtx = np.array([rect_midx[0], grp_bnds[1,0,0]-xpads[1,0], grp_bnds[2,0,0]-xpads[2,0], rect_midx[3],
                  rect_midx[4], grp_bnds[5,0,1]+xpads[5,1], grp_bnds[6,0,1]+xpads[6,1], rect_midx[7]])

gtxty = np.array([grp_bnds[0,1,0]-ypads[0,0], rect_midy[1], rect_midy[2], grp_bnds[3,1,1]+ypads[3,1],
                  grp_bnds[4,1,0]-ypads[4,0], rect_midy[5], rect_midy[6], grp_bnds[7,1,1]+ypads[7,1]])

haligns = ['center','left','left','center',
          'center','right','right','center']

valigns = ['bottom','center','center','top',
          'bottom','center','center','top']

for i in range(len(gtypes)):

    wfac = 1
    axB.add_artist(Rectangle((anchx[i], anchy[i]), 
                  width=rect_wid[i]*wfac, height = rect_ht[i], 
                             facecolor=bkg_col, #edgecolor = type_cols[i], 
                             zorder=0))
    
    if i < 4:
        xside = 0
        halign = 'left'
    else:
        xside = 1
        halign = 'right'
    
    axB.text(s = gtype_nms[i], x=gtxtx[i], y = gtxty[i], fontsize=7,#bbox=props,
        horizontalalignment=haligns[i],verticalalignment=valigns[i],rotation=rots[i],zorder=1)

xlims = [gxmin-gdx/6,gxmax+gdx/6]
ylims = [gymin-gdy/12,gymax+gdy/12]


axB.set_xlim(*xlims)
axB.set_ylim(*ylims)

# edges
for edge in gnms_net.edges:
    g0,g1 = edge
    x0,y0 = node_pos[g0]
    x1,y1 = node_pos[g1]
    axB.plot([x0,x1],[y0,y1], c=edge_cols_d[edge], lw = edge_widths_d[edge], zorder=1)
    
axB.set_xticks([])
axB.set_yticks([])

#####################################
## C: gene expression            ####
#####################################

ti = 90
tf = npsts
marks = ['o','s','^','v','x','D','+']
ysc = ['log','log','log','linear','linear']
props = dict(facecolor=bkg_col,boxstyle='square,pad=0.2')
for j in range(ngrp):
    i = gtype_plt_grps[j]

    for k in range(len(gtypes[i])):
        gnm  = gtypes[i][k]
        gidx = gidx_dict[gnm]
        dat  = mf.norm0to1(evec1_nn[ti:tf, gidx]**2,0)
        #dat  = evec1_nn[ti:tf, gidx]**2
        
        axC[j].plot(np.arange(ti,tf), dat, label=gnm, color = gtxt_col_d[gnm])#, marker=marks[k])
        
        dat = mf.norm0to1(gexp_hi_corr_mu[i][:,k],0)
        axD[j].plot(np.arange(ti,tf), dat, label=gnm, color = gtxt_col_d[gnm])#, marker=marks[k])

#         axD[j].errorbar(np.arange(ti,tf), gexp_hi_corr_mu[i][:,k], yerr = gexp_hi_corr_sem[i][:,k], 
#                         color = cols[k], capsize=0)
    
    #axD[j].set_yscale(ysc[j])
    axC[j].axvline(bift,color='k',linestyle='--')
    axD[j].axvline(bift,color='k',linestyle='--')
    
    axC[j].set_yticks([0,0.5,1])
    axC[j].set_yticklabels(['0','.5','1'])
    
    axD[j].set_yticks([0,0.5,1])
    axD[j].set_yticklabels([])
    
    axC[j].tick_params(axis='y',pad=0.3)
    
    if j < ngrp-1:
        axC[j].set_xticklabels([])
        axD[j].set_xticklabels([])
    else:
        axC[j].set_xlabel('pseudotime')
        axD[j].set_xlabel('pseudotime')
    
    #props = dict(boxstyle='roundtooth', facecolor=type_cols[i], alpha=0.5)
    axC[j].text(x=-0.33,y=0.5,s=gtype_nms[i].replace('\n',' '), color = 'k', #color = type_cols[i],
                transform=axC[j].transAxes, fontsize=7, bbox=props, rotation=90,
               verticalalignment='center', horizontalalignment='left')
    
axC[0].set_title(r'$||\vec{s}^1_i||^2$ (norm.)')
axD[0].set_title(r'$\langle$expr$\rangle$ (norm.)')
    
#     nlegcol = 1 if len(gtypes[i])<4 else 2 
#     leg = axD[j].legend(labelspacing=0, ncol=nlegcol, loc=(0,0.1), fontsize=6, 
#                         frameon=False,handlelength=0, handletextpad=0)
#     for i,text in zip(range(3),leg.get_texts()):
#         plt.setp(text, color = cols[i])
#     for item in leg.legendHandles:
#         item.set_visible(False)
    
#####################################
## D: Evec loading               ####
#####################################


#plt.savefig('{0}/neut_corr.pdf'.format(figdir), bbox_inches='tight')


# In[ ]:


# distribution analysis


# In[178]:


#corr_bi_gidxs.shape[0]==np.sum(thresh_hist_cts[bift-ti,th_idx])


# In[190]:


def group_dicts(m):
    # m is an Nx2 array where the first column is sorted
    # returns a dict of {m(i,0):[m(i,1),m(i+1,1),m(i+2,1)...]} for all unique elements in first column
    # 
    splits     = np.hstack([[0],np.where(np.diff(m[:,0])>0)[0]+1,m.shape[0]])
    #splitdiffs = np.diff(splits)
    mdict      = {}
    idict     = {}
    for i in range(len(splits)-1):
        mdict[m[splits[i],0]] = m[splits[i]:splits[i+1],1]
        idict[m[splits[i],0]] = splits[i] # start idx
    return mdict, idict


# In[229]:


#ti,tf,bifts


# In[228]:


#t        = bift #ti


# In[230]:


nc_thresh             = 400
hi_nc_idxs_bi         = np.where(gidxs_nc_bi[:,2]>=nc_thresh)[0]
corr_bi_gidxs         = gidxs_nc_bi[hi_nc_idxs_bi,0:2]
corr_bi               = corrs_bi[hi_nc_idxs_bi]
mdict_bi, idict_bi    = group_dicts(corr_bi_gidxs)


# In[231]:


# correlations that are used at bifurcation time distribution across the whole time window
corrs_t               = np.nan+np.zeros((tf-ti,corr_bi_gidxs.shape[0]))

for t in range(ti, tf):
    gidxs_nc            = np.load('{0}/gidxs_nc_{1}.npy'.format(corr_dir, t))
    corrs               = np.load('{0}/pccs_{1}.npy'.format(corr_dir, t))
    mdict_t,  idict_t   = group_dicts(gidxs_nc)
    koverlap            = sorted(list(set(mdict_t.keys()).intersection(set(mdict_bi.keys()))))
    for k in koverlap:
        _,mbi_idxs,mt_idxs = np.intersect1d(mdict_bi[k],mdict_t[k],return_indices=True)
        locs               = mbi_idxs + idict_bi[k]
        val_idxs           = mt_idxs  + idict_t[k]
        corrs_t[t-ti,locs] = corrs[val_idxs]


# In[234]:


corrs_t.shape,np.where(np.isnan(corrs_t)),np.array_equal(corrs_t[bift-ti],corr_bi)


# In[237]:


#[np.array_equal(corrs_t[t-ti],corr_bi) for t in range(ti,tf)]


# In[246]:


hi_pos_corr_bi_cidxs = np.where(corr_bi>0.65)[0]
hi_neg_corr_bi_cidxs = np.where(corr_bi<-0.3)[0]


# In[296]:


i         = 0
bidx      = bift-ti
nboot     = 20
samp_sz   = tf-ti-1
corrs_nbi = np.hstack([corrs_t[:bidx-1,i],corrs_t[bidx:,i]])
corrs_bi  = corrs_t[bidx]

samp_idxs = np.random.choice(corrs,np)


# In[547]:


def boot_pval(x, y, nboot):
    
    n = len(x)
    m = len(y)
    
    xbar = np.mean(x)
    ybar = np.mean(y)
    zbar = np.mean(np.hstack([x,y]))
    
    t = (xbar - ybar)/np.sqrt(np.var(x)/n + np.var(y)/m)
    
    xp = x - xbar + zbar
    yp = y - ybar + zbar

    xsamp  = np.random.choice(xp, (n, nboot), replace=True)
    ysamp  = np.random.choice(yp, (m, nboot), replace=True)
    tsamps = (np.mean(xsamp, axis=0) - np.mean(ysamp, axis=0)
                ) / np.sqrt(np.var(xsamp, axis=0)/n+np.var(ysamp, axis=0)/m)
    
    return np.where(tsamps>=t)[0].shape[0] / nboot

# assume y is a number, not an array
# this doesn't work well!
def boot_pval1(x, y, nboot):
    
    # right sided tail...
    sgn    = np.sign(y)
    n      = len(x)
    sqrtn  = np.sqrt(n)
    
    xbar   = np.mean(x)
    zbar   = np.mean(np.hstack([x,[y]]))
    
    t      = sgn*(y - xbar) * sqrtn / np.std(x)
    
    xp     = x - xbar + zbar
    xsamps = np.random.choice(xp, (n, nboot),replace=True)
    tsamps = sgn*(zbar - np.mean(xsamps,axis=0)) * sqrtn / np.std(xsamps, axis=0)
    
    return np.where(tsamps>=t)[0].shape[0] / nboot

def boot_qs(x, qs, nboot):
    xsamps = np.random.choice(x, (len(x), nboot),replace=True)
    
def boot_norm_cdf(x, bidx, nboot):
    
    # right sided tail...
    #xsgn   = np.sign(x[bidx])*x
    xsgn   = np.sign(x[np.argmax(np.abs(x))])*x
    n      = len(x)
    xsamps = np.random.choice(xsgn, (n, nboot),replace=True)
    mus    = np.mean(xsamps,axis=0)
    return 1 - norm.cdf(xsgn[bidx], loc=np.mean(mus), scale=np.std(mus))


# In[558]:


norm_cdfs = np.array([boot_norm_cdf(corrs_t[:,i], bidx, 10000) for i in np.arange(corrs_t.shape[1])])


# In[559]:


lowpv_gidxs = np.where(norm_cdfs==0)[0]
lowpv_gidxs.shape[0]


# In[560]:


corr_idx = lowpv_gidxs[6]#hi_pos_corr_bi_cidxs[1]#26990 #hi_neg_corr_bi_cidxs[1]
fig,axs=plt.subplots(1,2,figsize=(4,1),dpi=200)
axs[0].plot(np.arange(ti,tf),corrs_t[:,corr_idx],'ko-')
axs[0].axvline(bift,color='k',ls='--',alpha=0.5)

axs[1].hist(corrs_t[:,corr_idx])
axs[1].axvline(corrs_t[bidx,corr_idx],color='k',ls='--',alpha=0.5)


# In[ ]:





# In[18]:


binmin   = -1
binmax   = 1
dbin     = 0.05
corr_bin_edges = np.arange(binmin-dbin/2,binmax+dbin,dbin)
corr_bin_ctrs  = 0.5*(corr_bin_edges[1:]+corr_bin_edges[:-1])
corr_nbin      = corr_bin_ctrs.shape[0]


# In[561]:


corrs_t.shape


# In[562]:


corr_t_hists    = np.zeros((tf-ti, corr_nbin))
for t in range(tf-ti):
    corr_t_hists[t]   =np.histogram(corrs_t[t], bins = corr_bin_edges, density=True)[0]


# In[565]:


corr_t_hists.shape


# In[569]:


fig,axA=plt.subplots(figsize=(2,4),dpi=150)

cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
cmap.set_bad('white')

hist_min = 1e-4
hist_max = 6
ti = ti

corr_hist_idxs = np.where((corr_bin_ctrs<0.85)&(corr_bin_ctrs>-0.6))[0]

im = axA.imshow(corr_t_hists[:,corr_hist_idxs].T, cmap=cmap,aspect='auto',
                norm=matplotlib.colors.LogNorm(vmin=hist_min,vmax=hist_max))


# In[563]:


axs.imshow(corr_t_hists)


# In[ ]:


corrs_t_nbi = np.vstack([corrs_t[:bidx-1],corrs_t[bidx:]])


# In[533]:


norm_cdfs[hi_neg_corr_bi_cidxs],norm_cdfs[1]


# In[396]:


bidx


# In[388]:


#should take about 30 seconds with nboot = 1000
pvals = np.array([boot_pval1(corrs_t_nbi[:,i], corrs_t[bidx,i], 10000) for i in np.arange(corrs_t.shape[1])])


# In[386]:


corrs_t_nb


# In[400]:


np.where(pvals==0)[0]


# In[399]:


[pvals[k] for k in hi_pos_corr_bi_cidxs]


# In[367]:


pvals[hi_pos_corr_bi_cidxs[1]]


# In[446]:


corrs_t[bidx,corr_idx]


# In[458]:


# right sided tail...
i = 1
x = corrs_t_nbi[:,i]#-np.min(corrs_t[:,i])
y = 0.0381 #corrs_t[bidx,i]-np.min(corrs_t[:,i])
nboot = 1000000

sgn    = np.sign(y)
n      = len(x)
sqrtn  = np.sqrt(n)

xbar   = np.mean(x)
zbar   = np.mean(np.hstack([x,[y]]))

t      = sgn*(y - xbar) * sqrtn / np.std(x)

xp     = x - xbar + zbar
xsamps = np.random.choice(xp, (n, nboot),replace=True)
tsamps = sgn*(zbar - np.mean(xsamps,axis=0)) * sqrtn / np.std(xsamps, axis=0)

np.where(tsamps>=t)[0].shape[0] / nboot


# In[506]:


exbad  = corrs_t[:,1]
exgood = corrs_t[:,hi_neg_corr_bi_cidxs[0]]
nboot  = 1000
xs     = np.array([exbad, exgood])
xsamps = np.array([np.random.choice(x,(len(x),nboot)) for x in xs])


# In[502]:


mus  = np.mean(xsamps,axis=1)
sigs = np.mean(xsamps,axis=1)


# In[ ]:


#what is the probability that x >= xmu and x <= xmu given xmu


# In[513]:


from scipy.stats import norm


# In[520]:


np.array([norm.cdf(xs[i,bidx],loc=np.mean(mus[i]),scale=np.std(mus[i])) for i in range(len(xs))])


# In[514]:


print(norm.cdf.__doc__)


# In[509]:


xsampsf = np.reshape(xsamps,(2,-1))
np.where(xsampsf[0]>xs[0,bidx])[0].shape[0]/xsampsf[0].shape[0]


# In[512]:


np.where(xsampsf[1]<xs[1,bidx])[0].shape[0]/xsampsf[1].shape[0]


# In[500]:


exbad[bidx]


# In[499]:


fig,axs=plt.subplots()
axs.hist(mus)


# In[ ]:


qs = [1e-4,1e-3,1e-2,5e-2,0.95,]


# In[487]:


from scipy.stats import wilcoxon, mannwhitneyu,kruskal


# In[496]:


i=hi_pos_corr_bi_cidxs[0]
stat, p = kruskal(corrs_t_nbi[:,i], corrs_t[[bidx],i])
p


# In[476]:


stat


# In[464]:


xsampqs = np.quantile(xsamps,[0.01,0.99],axis=0)


# In[468]:


np.quantile(xsampqs[1],[0.025,0.95])


# In[ ]:





# In[451]:


np.std(x)


# In[452]:


np.std(xsamps,axis=0)


# In[457]:


y,zbar


# In[459]:


np.mean(zbar - np.mean(xsamps,axis=0))


# In[441]:


np.where(tsamps>=t)[0]


# In[433]:


xsamps.shape,np.mean(xsamps,axis=0)


# In[435]:


xsamps[:,0]


# In[429]:


x,np.mean(x),np.mean(xp),zbar,y,xbar,t


# In[428]:


np.vstack([xp,x]).T


# In[416]:


t,tsamps.shape


# In[417]:


zbar


# In[411]:


fig,axs=plt.subplots()
axs.hist(tsamps)
axs.axvline(t)


# In[306]:



corrs_t_bi  = corrs_t[[bidx]]


# In[307]:


corrs_t_nbi.shape,corrs_t_bi.shape


# In[317]:


get_ipython().run_line_magic('timeit', 'boot_pval(corrs_t_nbi[:,0], corrs_t_bi[:,0], 1000)')


# In[323]:


get_ipython().run_line_magic('timeit', 'boot_pval1(corrs_t_nbi[:,0], corrs_t_bi[0,0], 1000)')


# In[329]:


boot_pval1(corrs_t_nbi[:,0], corrs_t_bi[0,0], 1000)


# In[300]:


np.var,np.std


# In[295]:


corrs.shape


# In[293]:


bidx


# In[ ]:


# np.array_equal(corrs_t[t-ti],corr_bi)


# In[209]:


mm=np.zeros(10)
mm[[2,3,5]]=[1,2,-1]
mm


# In[202]:


mbi_idxs,mt_idxs


# In[203]:


k,len(mdict_t[k])


# In[206]:


len(koverlap)


# In[159]:


len(koverlap)


# In[148]:


mdict_bi.


# In[132]:


np.sum(np.diff(splits))


# In[136]:


len(np.diff(splits))


# In[134]:


len(mdict)


# In[133]:


m.shape


# In[127]:


m[splits[-2]-1:splits[-2]+1]


# In[118]:


mdict.get(1)


# In[119]:


i=0
splits[0],m[splits[0],0],m[splits[i]:splits[i+1],1]


# In[111]:


gidxs_nc[0:5,0:2]


# In[106]:


np.max(gidxs_nc[:,0])


# In[ ]:





# In[103]:


np.array_equal(gidxs_nc[:,0],np.sort(gidxs_nc[:,0]))


# In[102]:


gidxs_nc[0:5,0:2]


# In[93]:


ti = ti
tf = tf
thresh_hists = np.zeros((tf-ti, nc_thresh.shape[0], corr_nbin))

for t in range(ti,tf):
    corrs    = np.load('{0}/pccs_{1}.npy'.format(corr_dir, t))
    gidxs_nc = np.load('{0}/gidxs_nc_{1}.npy'.format(corr_dir, t))
    for i in range(nc_thresh.shape[0]):
        corr_idxs = np.where(gidxs_nc[:,2]>=nc_thresh[i])[0]
        thresh_hists[t-ti,i]   =np.histogram(corrs[corr_idxs], bins = corr_bin_edges, density=True)[0]
#        thresh_hist_cts[t-ti,i]=np.histogram(corrs[corr_idxs], bins = corr_bin_edges)[0]

    
#thresh_hists = np.array(thresh_hists)


# In[25]:


# gene graph


# In[26]:


t = bift
corrs_bi    = np.load('{0}/pccs_{1}.npy'.format(corr_dir, t))
gidxs_nc_bi = np.load('{0}/gidxs_nc_{1}.npy'.format(corr_dir, t))


# In[166]:


corrs_bi.shape,gidxs_nc_bi.shape[0]


# In[245]:


hi_pos_corr_bi_cidxs


# In[27]:


th_idx = np.where(nc_thresh==400)[0][0]
hi_pos_corr_bi_cidxs = np.where((corrs_bi>0.65) & (gidxs_nc_bi[:,2]>=nc_thresh[th_idx]))[0]
hi_neg_corr_bi_cidxs = np.where((corrs_bi<-0.3) & (gidxs_nc_bi[:,2]>=nc_thresh[th_idx]))[0]
hi_corr_bi_cidxs     = np.hstack([hi_pos_corr_bi_cidxs,hi_neg_corr_bi_cidxs])


# In[28]:


hi_pos_corr_bi_gidxs = gidxs_nc_bi[hi_pos_corr_bi_cidxs,0:2]
hi_neg_corr_bi_gidxs = gidxs_nc_bi[hi_neg_corr_bi_cidxs,0:2]
hi_corr_bi_gidxs = np.vstack([hi_pos_corr_bi_gidxs, hi_neg_corr_bi_gidxs])


# In[ ]:





# In[191]:


fig,axs=plt.subplots(figsize=(2,4))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
for node in gnms_net.nodes():
    props['facecolor'] = node_cols_d[node]
    x,y = node_pos[node]
    axs.text(x,y, node, fontsize=6, bbox=props,horizontalalignment='center',verticalalignment='center') #, verticalalignment='top'

for edge in gnms_net.edges:
    g0,g1 = edge
    x0,y0 = node_pos[g0]
    x1,y1 = node_pos[g1]
    axs.plot([x0,x1],[y0,y1], c=edge_cols_d[edge], lw = edge_widths_d[edge])
    
axs.axis('off')
    


# In[226]:


wid,ht


# In[261]:


gtype_dict


# In[335]:



fig,axB = plt.subplots(figsize=(col2_wd/nc*wid*1.17,ht),dpi=200)

transf = axB.transData.inverted()

props    = dict(boxstyle='round', facecolor='white', alpha=1)
grp_bnds = np.zeros((len(gtypes),2,2))
grp_bnds[:,:,0] =  np.inf
grp_bnds[:,:,1] = -np.inf

xoffs = 0.035*np.array([-1,-1,-1,-1,
                       1,1,1,1])
#plot nodes, keep track of min and max position of group
for node in gnms_net.nodes():
    props['facecolor'] = gtxt_col_d[node]

    x,y = node_pos[node]
    m=axB.text(x,y, node, fontsize=6, bbox=props, color='white', #gtxt_col_d[node],
             horizontalalignment='center',verticalalignment='center', zorder=2)
  
    bb            = m.get_window_extent(renderer = fig.canvas.get_renderer())
    bb_datacoords = bb.transformed(transf)
    grp_idx        = gtype_dict[node]
    
    grp_bnds[grp_idx,0,0] = np.amin([grp_bnds[grp_idx,0,0],bb_datacoords.x0])
    grp_bnds[grp_idx,0,1] = np.amax([grp_bnds[grp_idx,0,1],bb_datacoords.x1])
    grp_bnds[grp_idx,1,0] = np.amin([grp_bnds[grp_idx,1,0],bb_datacoords.y0])
    grp_bnds[grp_idx,1,1] = np.amax([grp_bnds[grp_idx,1,1],bb_datacoords.y1])



props    = dict(boxstyle='round', facecolor='wheat', alpha=1, edgecolor='none')

# [left, right] padding
xpads = np.array([[0.01,0.01],[0.03,0],[0.04,0],[0.01,0.01],
                  [0,0],[0,0.03],[0,0.03],[0,0]])

# [below, above] padding
ypads = np.array([[0.07,0.02],[0.04,0.04],[0.06,0.06  ],[0.025,0.06],
                  [0.06,0.02],[0.03,0.03],[0.03,0.03],[0.03,0.06]])
#ypads = np.zeros(8)

rect_wid = grp_bnds[:,0,1]-grp_bnds[:,0,0]+np.sum(xpads,axis=1)
rect_ht  = grp_bnds[:,1,1]-grp_bnds[:,1,0]+np.sum(ypads,axis=1)

anchx   = grp_bnds[:,0,0]-xpads[:,0]
anchy   = grp_bnds[:,1,0]-ypads[:,0]

rect_midx = 0.5*(grp_bnds[:,0,1]+grp_bnds[:,0,0])
rect_midy = 0.5*(grp_bnds[:,1,1]+grp_bnds[:,1,0])

rots = [0,90,90,0,
       0,270,270,0]

gtxtx = np.array([rect_midx[0], grp_bnds[1,0,0]-xpads[1,0], grp_bnds[2,0,0]-xpads[2,0], rect_midx[3],
                  rect_midx[4], grp_bnds[5,0,1]+xpads[5,1], grp_bnds[6,0,1]+xpads[6,1], rect_midx[7]])

gtxty = np.array([grp_bnds[0,1,0]-ypads[0,0], rect_midy[1], rect_midy[2], grp_bnds[3,1,1]+ypads[3,1],
                  grp_bnds[4,1,0]-ypads[4,0], rect_midy[5], rect_midy[6], grp_bnds[7,1,1]+ypads[7,1]])

haligns = ['center','left','left','center',
          'center','right','right','center']

valigns = ['bottom','center','center','top',
          'bottom','center','center','top']

for i in range(len(gtypes)):

    wfac = 1
    axB.add_artist(Rectangle((anchx[i], anchy[i]), 
                  width=rect_wid[i]*wfac, height = rect_ht[i], 
                             facecolor='wheat', #edgecolor = type_cols[i], 
                             zorder=0))
    
    if i < 4:
        xside = 0
        halign = 'left'
    else:
        xside = 1
        halign = 'right'
    
    axB.text(s = gtype_nms[i], x=gtxtx[i], y = gtxty[i], fontsize=7,#bbox=props,
        horizontalalignment=haligns[i],verticalalignment=valigns[i],rotation=rots[i],zorder=1)

xlims = [gxmin-gdx/6,gxmax+gdx/6]
ylims = [gymin-gdy/12,gymax+gdy/12]


axB.set_xlim(*xlims)
axB.set_ylim(*ylims)

# edges
for edge in gnms_net.edges:
    g0,g1 = edge
    x0,y0 = node_pos[g0]
    x1,y1 = node_pos[g1]
    axB.plot([x0,x1],[y0,y1], c=edge_cols_d[edge], lw = edge_widths_d[edge], zorder=1)
    
axB.set_xticks([])
axB.set_yticks([])


# In[274]:


# previous gene graph code
# rect_wids = gdx/3*np.array([1,1,1,1,
#                             1,1,1,1])
# rect_hts  = gdy/5*np.array([1,1,1,1,
#                             1,1,1,1])
rect_wids = 2*node_grp_rads+0.1
rect_hts  = 2*node_grp_rads+0.02

for i in range(len(gtypes)):
    #props['facecolor'] = type_cols[i]
    props['edgecolor'] = type_cols[i]
    axB.text(s = gtype_nms[i], x=node_grp_posx[i]+xoffs[i], 
             y = node_grp_posy[i]+yoffs[i], fontsize=7,bbox=props,
            horizontalalignment='center',verticalalignment='center',rotation=rots[i])
    axB.add_artist(Rectangle((node_grp_posx[i]-rect_wids[i]/2, node_grp_posy[i]-rect_wids[i]/2), 
                  width=rect_wids[i], height = rect_hts[i], edgecolor = type_cols[i], fill=False,zorder=0))


# nodes
# props = dict(boxstyle='round', facecolor='wheat', alpha=1)
# for node in gnms_net.nodes():
#     props['facecolor'] = node_cols_d[node]
#     x,y = node_pos[node]
#     axB.text(x,y, node, fontsize=6, bbox=props,horizontalalignment='center',verticalalignment='center', zorder=1)

props = dict(boxstyle='round', facecolor='white', alpha=1)
for node in gnms_net.nodes():
    props['edgecolor'] = node_cols_d[node]
    x,y = node_pos[node]
    axB.text(x,y, node, fontsize=6, bbox=props, color=gtxt_col_d[node],
             horizontalalignment='center',verticalalignment='center', zorder=2)
    
# edges
for edge in gnms_net.edges:
    g0,g1 = edge
    x0,y0 = node_pos[g0]
    x1,y1 = node_pos[g1]
    axB.plot([x0,x1],[y0,y1], c=edge_cols_d[edge], lw = edge_widths_d[edge], zorder=1)
    
axB.set_xticks([])
axB.set_yticks([])

#axs.axis('off')
# limits
axB.set_xlim(gxmin-gdx/5,gxmax+gdx/5)
axB.set_ylim(gymin-gdy/10,gymax+gdy/10)

# annotations
##### gtypes = [membrane, metabolism, neutrophil, housekeeping, mitochon, misc, sig_dev, myeloid]

type_cols = plt.cm.tab10.colors
xoffs = 0.18*np.array([-1,-1,-1,-1,
                    0.75,0.75,0.75,0.75])
yoffs = np.array([0,0,0,0,
                  0,0,0,0])
rots = [90,90,90,90,
       270,270,270,270]
props = dict(boxstyle='roundtooth', facecolor='white', alpha=1)



#axB.axis('off')


# In[253]:


m.get_position(), m.get_size()


# In[ ]:


m.get_window_extent()


# In[254]:


transf = axB.transData.inverted()
bb = m.get_window_extent(renderer = fig.canvas.renderer)
bb_datacoords = bb.transformed(transf)


# In[259]:


bb_datacoords.x0,bb_datacoords.x1,bb_datacoords.y0,bb_datacoords.y1


# In[169]:


# node_cols =[]
# edge_cols =[]
# edge_widths =[]
# for node in gnms_net.nodes():
#     node_cols.append(cols[gtype_dict[node]])


# In[87]:


fig,axs=plt.subplots()
cols = plt.cm.viridis(np.linspace(0,1,nc_thresh.shape[0]))
for i in range(nc_thresh.shape[0]):
    axs.plot(corr_bin_ctrs, thresh_hists[i],'o-', color=cols[i], label=nc_thresh[i])
    
axs.legend()
#axs.set_yscale('symlog')


# In[94]:


thresh_mu = np.sum(thresh_hists*corr_bin_ctrs*dbin,axis=2)
#thresh_std = np.sum


# In[95]:


thresh_hists.shape


# In[99]:


fig,axs=plt.subplots()
cols = plt.cm.viridis(np.linspace(0,1,thresh_mu.shape[0]))

# for i in range(tf-ti):
#     axs.plot(nc_thresh, thresh_mu[i],'o-', color=cols[i])

axs.errorbar(nc_thresh, np.mean(thresh_mu,axis=0), yerr=np.std(thresh_mu,axis=0), color='k', linestyle='--')


# In[387]:


np.amax(thresh_hists[:,th_idx]),np.amin(thresh_hists[:,th_idx][np.nonzero(thresh_hists[:,th_idx])])


# In[14]:


fig,axs=plt.subplots(figsize=(4,7))

cmap = matplotlib.cm.viridis
cmap.set_bad('white')

th_idx = np.where(nc_thresh==400)[0][0]
hist_min = 1e-4
hist_max = 6
ti = ti

#axs.imshow(np.log10(thresh_hists[:,th_idx].T), aspect='auto')

im = axs.imshow(thresh_hists[:,th_idx].T, cmap=cmap,aspect='auto',
                norm=matplotlib.colors.LogNorm(vmin=hist_min,vmax=hist_max))

axs.set_xlabel('pseudotime')
axs.set_ylabel('correlation coefficient')
tskip = 5
axs.set_xticks(np.arange(0,tf-ti,tskip))
axs.set_xticklabels(np.arange(ti,tf,tskip))
bskip = 2
axs.set_yticks(np.arange(0,corr_bin_ctrs.shape[0],bskip))
axs.set_yticklabels(['{0:.2f}'.format(i) for i in corr_bin_ctrs[::bskip]])
axs.axvline(bift-ti, alpha=0.5)

mf.set_axs_fontsize(axs,20)

divider = make_axes_locatable(axs)
cax1    = divider.append_axes("top", size="3%", pad=0.05)

cbar = fig.colorbar(im, cax=cax1, orientation='horizontal', aspect=1)
cbar.set_label('frequency',rotation=0)
cax1.xaxis.set_label_position('top')
cax1.xaxis.set_ticks_position('top')
ticks = np.power(10.,np.arange(np.floor(np.log10(hist_min)),np.ceil(np.log10(hist_max))+1))
cbar.set_ticks([])

minorticks = np.array([np.arange(ticks[i],ticks[i+1],ticks[i]) for i in range(ticks.shape[0]-1)]).flatten()
mticksf    = minorticks[np.logical_and(minorticks>=hist_min,minorticks<=hist_max)]
ticksf     = ticks[np.logical_and(ticks>=hist_min,ticks<=hist_max)]

#cbar.ax.xaxis.set_ticks(minorticks, minor=True)
cax1.xaxis.set_ticks(mticksf, minor=True)
cax1.xaxis.set_ticks(ticksf)

mf.set_axs_fontsize(cax1,20)


#plt.savefig('{0}/corr_hist_v_pseudotime.jpg'.format(plotdir), bbox_inches='tight')


# In[394]:


ticks,hist_max


# In[396]:


np.ceil(np.log10(6))


# In[61]:


np.where(gidxs_nc[:,2]>950)[0].shape[0]


# In[18]:


#hi_pos_corr_bi_gidxs


# In[19]:


#G.get_edge_data(7786,5506)['weight']


# In[223]:


fig,axs=plt.subplots()
for i in range(hi_corr_bi_gidxs.shape[0]):
    axs.plot(hi_corr_cov_t[:,i],'-')
    
axs.set_xlabel('pseudotime')
axs.set_ylabel('covariance of genes\nhighly correlated at $t_c$')
tskip = 4
axs.set_xticks(np.arange(0,tf-ti,tskip))
axs.set_xticklabels(np.arange(ti,tf,tskip))

axs.axvline(bifts[3]-ti, alpha=0.5)

axs.set_yscale('symlog')
mf.set_axs_fontsize(axs,20)


# In[198]:


gexp_lil[neut_pst_cidxs[t]][:,[i,j]].toarray().T.shape


# In[197]:


t = bifts[3]
i = 0
j = 1
np.cov(gexp_lil[neut_pst_cidxs[t]][:,[i,j]].toarray().T)


# In[152]:


np.unique(mf.flatten2d(gnms[gidxs_nc_bi[hi_pos_corr_bidxs,0:2]]))

hi_cornp.unique(mf.flatten2d(gnms[gidxs_nc_bi[hi_neg_corr_bidxs,0:2]]))
# In[154]:


ltf_idx = np.where(gnms=='Ltf')[0][0]


# In[161]:


ltf_corr_idxs = np.hstack([np.where(gidxs_nc_bi[:,0]==ltf_idx)[0],np.where(gidxs_nc_bi[:,1]==ltf_idx)[0]])


# In[242]:


evec1.shape


# In[226]:





# In[874]:


#evec1.shape,evec1_nn.shape


# In[354]:


nrow,ncol = 5,7
fig, axs= plt.subplots(nrow,ncol, figsize=(20,7))

k = 0
cols = ['blue','red']
ti = 90
for i in range(len(cc_grps)):
    for j in range(cc_grps[i].shape[0]):
        gidx = cc_grps[i][j]
        r,c = int(k/ncol),k%ncol
        lbl = gnms[gidx]+r' ($R_{\rm max}$='+'{0:.2f})'.format(highest_weight[gidx])
        axs[r,c].plot(evec1_nn[ti:,gidx],'-',color = cols[i])
        
        axs[r,c].set_title(lbl)
        
        #axs[r,c].legend()
        axs[r,c].axvline(bifts[3]-ti, alpha=0.5, color='k', linestyle = '--')
        
        if r!=nrow-1:
            axs[r,c].set_xticklabels([])
        else:
            axs[r,c].set_xticks()
            axs[r,c].set_xlabel('pseudotime')
        #axs[r,c].set_xlim(90,122)
        
        k+=1
axs[2,0].set_ylabel('covariance eigenvector 1 loadings')

#axs.set_xlabel('pseudotime')
#axs.set_ylabel('eigenvector 1 loading')

# tskip = 4
# axs.set_xticks(np.arange(0,tf-ti,tskip))
# axs.set_xticklabels(np.arange(ti,tf,tskip))

#     axs[r,c].axvline(bifts[3], alpha=0.5, color='k', linestyle = '--')

#     axs[r,c].set_xlim(90,122)

#axs.set_ylim(0,0.05)
#axs.set_yscale('symlog',linthreshy=1e-4)
#mf.set_axs_fontsize(axs,20)

plt.subplots_adjust(wspace=0.4,hspace=0.5)


# In[464]:


ti = 105
fig,axs=plt.subplots()
axs.imshow(evec1_nn[ti:,cc_grps[0]].T, norm=matplotlib.colors.SymLogNorm(linthresh=1e-4,vmin=-0.7,vmax=0.7))
axs.axvline(bifts[3]-ti)


# In[493]:


ti = 90
tf = npsts
evec1_nn_birng = evec1_nn[ti:tf,cc_grps[0]]
evec1_nn_birng_nrmd = (( evec1_nn_birng - np.amin(evec1_nn_birng,axis=0) )  / 
( np.amax(evec1_nn_birng,axis=0) - np.amin(evec1_nn_birng,axis=0) ))
evec1_nn_birng.shape


# In[ ]:





# In[522]:


max_load_idxs = np.argmax(evec1_nn_birng,axis=0)
min_load_idxs = np.argmin(evec1_nn_birng,axis=0)


# In[524]:


max_load_idxs


# In[520]:


max_abs_loads     = np.amax(np.abs(evec1_nn_birng),axis=0)
max_abs_loads_arg = np.argmax(np.abs(evec1_nn_birng),axis=0)
srt_ord           = np.argsort(max_abs_loads)


# In[521]:


max_abs_loads_arg


# In[511]:


from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# In[620]:


pad_with_spaces = lambda s,n: s+' '*(n-len(s))


# In[626]:


crv_labs = ['{0} [{1:0.3f},{2:0.3f}]'.format(pad_with_spaces(gnms[cc_grps[0][srt_ord[i]]], 0),
                                             evec1_nn_birng[min_load_idxs[srt_ord[i]],srt_ord[i]],
                                             evec1_nn_birng[max_load_idxs[srt_ord[i]],srt_ord[i]]) 
            for i in range(cc_grps[0].shape[0])]


# In[627]:


crv_labs


# In[542]:


crv_labs = ['{0}({1:0.3f})'.format(gnms[cc_grps[0][srt_ord[i]]], 
                                    max_abs_loads[srt_ord[i]]) 
            for i in range(cc_grps[0].shape[0])]


# In[628]:


fig,axs = plt.subplots(1,1)
cols = plt.cm.viridis(np.linspace(0,1,cc_grps[0].shape[0]))

ax = axs
for i in range(cc_grps[0].shape[0]):
    ax.plot(np.arange(ti, tf), evec1_nn_birng_nrmd[:,srt_ord[i]],'-o',markersize=3, color = cols[i]) #label=crv_labs[i])

ax.axvline(bifts[3],color='k',linestyle='--')
ax.set_xlabel('pseudotime')
ax.set_ylabel('pc1 loading (norm.)')
mf.set_axs_fontsize(ax,20)


col_patches = [Patch(facecolor=cols[i],label=crv_labs[i]) for i in range(len(cols))]

axs.legend(handles=col_patches, labelspacing=0,loc=(1,0),ncol=2,fontsize=14, handlelength=1)
# the_table = plt.table(cellText=dat_fmt,
#                       rowLabels=rows,
#                       rowColours=cols,
#                       colLabels=columns,
#                       loc='right',
#                      colWidths=[0.1,0.1])

# #axs.add_table(the_table)

# plt.subplots_adjust(left=0,right=0.3,bottom=0,wspace=10)


# In[638]:


gn_labels = {}
neut_markers = ['S100a9','Itgb21','Elane','Fcnb','Mpo','Prtn3','S100a6','S100a8','Lcn2','Lrg1']
myel_markers = ['S100a8','Ngp','Ltf']
pmy_markers = ['Gfi1','Elane']
gpp_markers = ['Csf1r','Cebpa']
mpp_markers = ['Cd34','Flt3','Meis1','Ly6a']

all_markers = [neut_markers, myel_markers, pmy_markers, gpp_markers, mpp_markers]
mark_nm = ['neutrophil', 'myeloid', 'protomyeloid', 'GPP', 'MPP']
for i in range(len(all_markers)):
    for gn in all_markers[i]:
        gn_labels[gn] = gn_labels.get(gn,[]) + ['{0} marker'.format(mark_nm[i])]pa 


# In[716]:


4745.04+752.55-1023.82-750


# In[722]:


np.where(gnms=='Rab-7A')


# -ctsb/d -- cathepsin b / d -- intracellular degradation / protein turnover... cancer related (https://en.wikipedia.org/wiki/Cathepsin_D)
# -sirpa -- myeloid marker (https://en.wikipedia.org/wiki/Signal-regulatory_protein_alpha) membrane bound, brings stuff into cell, meidates mast cell and dendritic cell activation
# -mpeg1 -- cell cycle (https://www.genecards.org/cgi-bin/carddisp.pl?gene=MPEG1)
# -h2-d1 -- antigen presenting / immune system
# -psap -- hydrolosis membrane protein / neurotrophic activity (https://www.genecards.org/cgi-bin/carddisp.pl?gene=PSAP)
# -laptm5 -- membrane protein / lysosome / embryogenesis (https://www.genecards.org/cgi-bin/carddisp.pl?gene=LAPTM5)
# -sat1  -- transferase (??) (https://www.genecards.org/cgi-bin/carddisp.pl?gene=sat1)
# -cc16 -- lung biomarker
# -sdcbp -- frizzled binding , interleukin-5 binding (https://www.genecards.org/cgi-bin/carddisp.pl?gene=SDCBP)
# -gstm1 -- transferase (gene cards)
# -cd9 -- membrane fusion, migration, cancer, exosome marker, lots of stuff (https://en.wikipedia.org/wiki/CD9)
# -cstb -- cathepsin inhibitor (https://www.genecards.org/cgi-bin/carddisp.pl?gene=CSTB)
# -plin2 -- adipocyte differentiation, formation of lipid droplets
# -sqstm1 -- enzyme binding, ubiquitin binding
# -h3f3a -- histone protein-- maintains gene integrity during development (pluripotency?) (https://en.wikipedia.org/wiki/H3F3A)
# -cybb -- impacts neutrophil recruitment ? (https://en.wikipedia.org/wiki/NOX2#cite_note-three-11) (https://www.openaccessrepository.it/record/22584#.X9js6stKg-Q)
# -bri3 -- maybe target of wnt / beta-catenin signalling, participates in tumor necrosis (https://www.genecards.org/cgi-bin/carddisp.pl?gene=BRI3#function)                                                                                     
# -srgn-- expressed in expressed in hematopoietic cells and endothelial cells, plays a role in localizing neutrophil elastase in azurophil granules of neutrophils. protease storage. (https://www.genecards.org/cgi-bin/carddisp.pl?gene=srgn#function) (https://en.wikipedia.org/wiki/Serglycin)
# -fth1 -- iron storage/delivery (https://www.genecards.org/cgi-bin/carddisp.pl?gene=FTH1)
# -ftl1 -- iron binding
# -rab7 -- involved in endocytosis-- prob important for neutrophils (https://en.wikipedia.org/wiki/RAB7A) granulocyte marker but not myeloid marker (or vise versa!) (https://www.genecards.org/cgi-bin/carddisp.pl?gene=RAB7A#function)
# -anxa4 -- expressed in epithelial cells. probably membrane activity related (e.g. endocytosis)
# 
# genes listed in paper:
# 
# -ngp    -- myeloid and neutrophil marker
# -lcn2   -- neutrophil marker but also iron binding (probably means fth1 and ftl1 belong with it??)
# -s100a6 -- neutrophil marker but also calcium signalling (https://www.genecards.org/cgi-bin/carddisp.pl?gene=S100A6#function)
# -s100a8 -- neutrophil + myeloid marker and calcium signalling (https://www.genecards.org/cgi-bin/carddisp.pl?gene=S100A8#function)
# -s100a9 -- neutrophil marker and calcium binding + whole shitload of stuff (https://www.genecards.org/cgi-bin/carddisp.pl?gene=S100A9#function )

# In[ ]:


housekeeping:
    cstb
    ctsd
    fth1
    
myeloid
    sirpa
    ngp (also neutrophil, weinreb)
    s100a8 (also neutrophil, weinreb)

neutrophil
    s100a6 (weinreb)
    s100a9 (weinreb)
    lcn2 (weinreb)
    

iron binding: (maybe characterize as neutrophil due to lcn2 overlap?)
    fth1 (also housekeeping)
    ftl1 (could probably move to metabolism since function is to remove iron)
    lcn2 (also neutrophil)
    
membrane processes / endocytosis:
    rab7
    anxa4
    cd9
    
hydrolosis / metabolism / enzyme:
    psap
    laptm5
    sat1
    gstm1
    sqstm1
    ctsb 
    ctsd (also housekeeping)
    
signalling (combine with development?)
    sdcbp
    cstb (also housekeeping)
    sqstm1
    bri3
    
development
    plin2 -- adipocyte 
    cybb -- neutrophil recruitment
    srgn -- hematopoietic  
    
uncategorized
    cc16 -- lung biomarker
    h3f3a -- histone
    mpeg1 -- cell cycle
    h2-d1 -- immune system


# In[869]:


cols    = plt.cm.tab10.colors
fig,axs = plt.subplots(figsize=(12,12))
nx.draw_networkx(gnms_net, node_size=2000, font_size=12, 
                 node_color = node_cols, edge_color = edge_cols, width = edge_widths, 
                 pos = node_pos)# nx.kamada_kawai_layout(gnms_net))
xoffs = np.array([-0.13,-0.14,-0.14,-0.14,0.11,0.09,0.09,0.09])
yoffs = np.array([-0.12,0.07,0,0,0.05,0.05,0,0])
for i in range(len(gtypes)):
    axs.text(s = gtype_nms[i], x=node_grp_posx[i]+xoffs[i], 
             y = node_grp_posy[i]+yoffs[i], color = cols[i],fontsize=15)
#axs.axis('off')
plt.savefig('{0}/hi_corr_grn.jpg'.format(plotdir),bbox_inches = 'tight')


# In[97]:


norm01 = lambda x:(x-np.amin(x))/(np.amax(x)-np.amin(x))


# In[141]:


nrow = 4
ncol = 2

type_cols = plt.cm.tab10.colors


ti = 90
tf = npsts
asp = 1.5
fig,axs = plt.subplots(nrow,ncol,figsize=(8.7/2.54,8.7/2.54*asp),dpi=200)
marks = ['o','s','^','v','x','D','+','*']
cols = plt.cm.tab20.colors
for i in range(nrow):
    
    for j in range(ncol):
        type_idx = j*nrow+3-i
        ax = axs[i,j]
        #cols = plt.cm.viridis(np.linspace(0,1,len(gtypes[type_idx])))

        for k in range(len(gtypes[type_idx])):
            gnm  = gtypes[type_idx][k]
            gidx = gidx_dict[gnm]
            dat  = norm01(evec1_nn[ti:tf, gidx]**2)
            #dat  = evec1_nn[ti:tf, gidx]**2
            #dat = mf.norm0to1(gexp_hi_corr_mu[i][:,k],0)
            ax.plot(np.arange(ti,tf),dat, label=gnm, color = cols[k], marker=marks[k])
            
        ax.axvline(bift,color='k',linestyle='--')
        
        if j == 0:
            if i==2:
                ax.set_ylabel(r'fraction of pc1 $(\vec{v}_i^2)$ (norm.)')
                ax.yaxis.set_label_coords(-0.2,1.5)

        else:
            ax.set_yticklabels([])
        
        if i == nrow-1:
            ax.set_xlabel('pseudotime')
        else:
            ax.set_xticklabels([])
    
        ax.set_title(gtype_nms[type_idx].replace('\n',' '),color = type_cols[type_idx])
        
        #mf.set_axs_fontsize(ax,20)
        nlcol = 1
        lloc = None
        if len(gtypes[type_idx])>5:
            nlcol=2
            lloc = (0.27,0.5)
        elif type_idx == 6:
            lloc = (0.2,0.3)
            
        leg = ax.legend(labelspacing=0, ncol= nlcol,fontsize=6,frameon=False,handlelength=0.5,handletextpad=0,
                  columnspacing=2.5, loc=lloc)
        for k,text in zip(range(len(gtypes[type_idx])),leg.get_texts()):
            plt.setp(text, color = cols[k])
        
plt.subplots_adjust(hspace=0.2,wspace=0.1)
plt.savefig('{0}/v1sq.pdf'.format(figdir),bbox_inches = 'tight')


# In[160]:


nrow = 4
ncol = 2

type_cols = plt.cm.tab10.colors

ti = 90
tf = npsts
asp = 1.5
fig,axs = plt.subplots(nrow,ncol,figsize=(8.7/2.54,8.7/2.54*asp),dpi=200)
marks = ['o','s','^','v','x','D','+','*']
cols = plt.cm.tab20.colors
for i in range(nrow):
    
    for j in range(ncol):
        type_idx = j*nrow+3-i
        ax = axs[i,j]
        #cols = plt.cm.viridis(np.linspace(0,1,len(gtypes[type_idx])))

        for k in range(len(gtypes[type_idx])):
            gnm  = gtypes[type_idx][k]
            gidx = gidx_dict[gnm]
            #dat  = norm01(evec1_nn[ti:tf, gidx]**2)
            #dat  = evec1_nn[ti:tf, gidx]**2
            dat = mf.norm0to1(gexp_hi_corr_mu[type_idx][:,k],0)
            ax.plot(np.arange(ti,tf),dat, label=gnm, color = cols[k], marker=marks[k])
            
        ax.axvline(bift,color='k',linestyle='--')
        
        if j == 0:
            if i==2:
                ax.set_ylabel(r'gene expression (norm.)')
                ax.yaxis.set_label_coords(-0.2,1.5)

        else:
            ax.set_yticklabels([])
        
        if i == nrow-1:
            ax.set_xlabel('pseudotime')
        else:
            ax.set_xticklabels([])
    
        ax.set_title(gtype_nms[type_idx].replace('\n',' '),color = type_cols[type_idx])
        
        #mf.set_axs_fontsize(ax,20)
        nlcol = 1
        lloc = None
#         if len(gtypes[type_idx])>5:
#             nlcol=2
#             lloc = (0.27,0.5)
#         elif type_idx == 6:
#             lloc = (0.2,0.3)
        if type_idx==6:
            lloc=(0.74,0.14)
        elif type_idx== 1:
            lloc = (0.27,0.28)
            
            
        leg = ax.legend(labelspacing=0, ncol= nlcol,fontsize=6,frameon=False,handlelength=0.5,handletextpad=0,
                  columnspacing=2.5, loc=lloc,borderpad=0)
        for k,text in zip(range(len(gtypes[type_idx])),leg.get_texts()):
            plt.setp(text, color = cols[k])
        
plt.subplots_adjust(hspace=0.2,wspace=0.1)
plt.savefig('{0}/corr_graph_gexp.pdf'.format(figdir),bbox_inches = 'tight')


# In[ ]:


#intentionally left blank


# In[104]:


nrow = 2
ncol = 4

type_cols = plt.cm.tab10.colors


ti = 90
tf = npsts
fig,axs = plt.subplots(2,4,figsize=(20,6))
marks = ['o','s','^','v','x','D','+','*']
for i in range(nrow):
    
    
    for j in range(ncol):
        type_idx = i*ncol+j
        ax = axs[i,j]
        cols = plt.cm.viridis(np.linspace(0,1,len(gtypes[type_idx])))

        for k in range(len(gtypes[type_idx])):
            gnm  = gtypes[type_idx][k]
            gidx = gidx_dict[gnm]
            dat  = norm01(evec1_nn[ti:tf, gidx]**2)
            #dat  = evec1_nn[ti:tf, gidx]**2
            ax.plot(np.arange(ti,tf),dat, label=gnm, color = cols[k], marker=marks[k])
            
        ax.axvline(bift,color='k',linestyle='--')
        
        if j == 0:
            if i==0:
                ax.set_ylabel('pc1 loading (norm.)                            ')
                ax.set_ylabel(r'fraction of pc1 $(\vec{v}_i^2)$ (norm.)                           ')

        else:
            ax.set_yticklabels([])
        
        if i == 1:
            ax.set_xlabel('pseudotime')
        else:
            ax.set_xticklabels([])
    
        ax.set_title(gtype_nms[type_idx].replace('\n',' '),color = type_cols[type_idx])
        
        mf.set_axs_fontsize(ax,20)
        nlegcol = 1 #if len(gtypes[type_idx])<5 else 2
        ax.legend(fontsize=14,labelspacing=0, ncol= nlegcol)
        
        #ax.set_yscale('log')

    

plt.subplots_adjust(hspace=0.25,wspace=0.05)
#plt.savefig('{0}/hi_corr_pc1sqnorm_by_group.jpg'.format(plotdir),bbox_inches = 'tight')
#axs.legend(handles=col_patches, labelspacing=0,loc=(1,0),ncol=2,fontsize=14, handlelength=1)


# ##### cc_grps

# In[898]:


cc_gidx_dict = {cc_grps[i][j]:[i,j] for i in range(len(cc_grps)) for j in range(len(cc_grps[i]))}


# In[899]:


cc_gidx_dict[5506][0]


# In[955]:


nrow = 2
ncol = 4

type_cols = plt.cm.tab10.colors


ti = 90
tf = npsts
fig,axs = plt.subplots(2,4,figsize=(20,6))
marks = ['o','s','^','v','x','D','+']
for i in range(nrow):
    
    
    for j in range(ncol):
        type_idx = i*ncol+j
        ax = axs[i,j]
        cols = plt.cm.viridis(np.linspace(0,1,len(gtypes[type_idx])))

        for k in range(len(gtypes[type_idx])):
            gnm  = gtypes[type_idx][k]
            gidx = gidx_dict[gnm]
            ccx, ccy  = cc_gidx_dict[gidx]
            coeff_var = coeff_vars[ccx][ccy]
            dat  = norm01(coeff_var[ti:tf])
            #dat = coeff_var[ti:tf]
            ax.plot(np.arange(ti,tf),dat, label=gnm, color = cols[k], marker=marks[k])
            
        ax.axvline(bifts[3],color='k',linestyle='--')
        
        if j == 0:
            if i==0:
                #ax.set_ylabel('coefficient of variation (norm.)                            ')
                ax.set_ylabel('coefficient of variation                                ')
        else:
           ax.set_yticklabels([])
        
        if i == 1:
            ax.set_xlabel('pseudotime')
        else:
            ax.set_xticklabels([])
    
        ax.set_title(gtype_nms[type_idx].replace('\n',' '),color = type_cols[type_idx])
        
        mf.set_axs_fontsize(ax,20)
        ax.legend(fontsize=12,labelspacing=0)

    

plt.subplots_adjust(hspace=0.25,wspace=0.05)
plt.savefig('{0}/hi_corr_cov_by_group_norm.jpg'.format(plotdir),bbox_inches = 'tight')


# In[887]:


type_idx,j,k


# In[888]:


gtypes[type_idx]


# In[767]:


nx.__version__


# In[642]:


[(gnms[gidx],gn_labels.get(gnms[gidx],[])) for gidx in cc_grps[0]]


# In[403]:


gexp_hi_corr_mu[0].shape


# In[384]:





# In[375]:


i = 0
j = 0
t = 0
m = gexp_hi_corr[i][j][t]
np.mean(m[m>0])


# In[ ]:


nrow,ncol = 5,7
fig, axs= plt.subplots(nrow,ncol, figsize=(20,7))

k = 0
cols = ['blue','red']
ti = 90
for i in range(len(cc_grps)):
    for j in range(cc_grps[i].shape[0]):
        gidx = cc_grps[i][j]
        r,c = int(k/ncol),k%ncol
        lbl = gnms[gidx]+r' ($R_{\rm max}$='+'{0:.2f})'.format(highest_weight[gidx])
        axs[r,c].plot(evec1_nn[ti:,gidx],'-',color = cols[i])
        
        axs[r,c].set_title(lbl)
        
        #axs[r,c].legend()
        axs[r,c].axvline(bifts[3]-ti, alpha=0.5, color='k', linestyle = '--')
        
        if r!=nrow-1:
            axs[r,c].set_xticklabels([])
        else:
            axs[r,c].set_xticks()
            axs[r,c].set_xlabel('pseudotime')
        #axs[r,c].set_xlim(90,122)
        
        k+=1
axs[2,0].set_ylabel('covariance eigenvector 1 loadings')

plt.subplots_adjust(wspace=0.4,hspace=0.5)


# In[ ]:


fig,axs=plt.subplots()
i = 0
j = 7
ti = 0
for t in range(npsts):
    gexp = gexp_hi_corr[i][j][t]
    axs.scatter(t*np.ones(gexp.shape[0]), gexp, color='k')
axs.axvline(bifts[3]-ti, alpha=0.5, color='k', linestyle = '--')
#axs.set_yscale('symlog')
axs.set_ylim(1e-1)
axs.set_ylabel('gene expression')
axs.set_ylim(90)


# In[631]:


fig,axs=plt.subplots()
i = 0
j = 7
ti = 90
axs.errorbar(np.arange(ti,npsts), gexp_hi_corr_mu[i][j,ti:], yerr=gexp_hi_corr_std[i][j,ti:])
axs.axvline(bifts[3], alpha=0.5, color='k', linestyle = '--')


# In[ ]:


fig,axs=plt.subplots()
i = 0
j = 7
ti = 90
axs.plot(np.arange(ti,npsts),gexp_hi_corr_std[i][j,ti:]/gexp_hi_corr_mu[i][j,ti:],'ko-')
#axs.plot(gexp_hi_corr_std[i][j,ti:]/gexp_hi_corr_mu[i][j,ti:],'ko-')

axs.axvline(bifts[3], alpha=0.5, color='k', linestyle = '--')
axs.set_ylabel('coeff of variation')


# In[917]:


len(gexp_hi_corr[0])


# In[921]:


gexp_hi_corr[i][0].shape


# In[ ]:


gexp_hi_corr_mu[9]


# In[924]:


gidx_dict['s100a9']


# In[927]:


fig,axs=plt.subplots(3,1,figsize=(5,10))

i = 0
j = np.where(cc_grps[i]==gidx_dict['fth1'])[0][0]
ti = 90

for t in range(ti,npsts):
    gexp = gexp_hi_corr[i][t][:,j]
    axs[0].scatter(t*np.ones(gexp.shape[0]), gexp, color='k')
    
axs[0].set_ylabel('gene expression')

axs[1].errorbar(np.arange(ti,npsts), gexp_hi_corr_mu[i][j,ti:], yerr=gexp_hi_corr_std[i][j,ti:])

axs[1].set_ylabel(r'$\mu \pm \sigma$')

axs[2].plot(np.arange(ti,npsts),gexp_hi_corr_std[i][j,ti:]/gexp_hi_corr_mu[i][j,ti:],'ko-')

axs[2].set_ylabel(r'$\sigma/\mu$')

for ax in axs:
    ax.axvline(bifts[3], alpha=0.5, color='k', linestyle = '--')

axs[2].set_xlabel('pseudotime')
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
gnm = gnms[cc_grps[i][j]]
axs[0].set_title(gnm)
for ax in axs:
    mf.set_axs_fontsize(ax,20)
    
plt.savefig('{0}/gexp_distr_{1}.jpg'.format(plotdir,gnm),bbox_inches = 'tight')


# In[ ]:





# In[709]:


coeff_vars = [gexp_hi_corr_std[i]/gexp_hi_corr_mu[i] for i in range(len(gexp_hi_corr))]


ti = 101
tf=118
coeff_vars_normd = [((coeff_vars[i][:,ti:tf].T-np.amin(coeff_vars[i][:,ti:tf],axis=1))/
                    (np.amax(coeff_vars[i][:,ti:tf],axis=1)-np.amin(coeff_vars[i][:,ti:tf],axis=1))).T
                   for i in range(len(coeff_vars))]


# In[ ]:


coeff


# In[710]:


coeff_vars[0].shape


# In[715]:


fig,axs=plt.subplots()

for i in range(len(cc_grps[0])):
    #if srt_ord[i] not in idxs:
    axs.plot(np.arange(ti,tf),coeff_vars_normd[0][srt_ord[i]],'o-',
             color=cols[i],label=gnms[cc_grps[0][srt_ord[i]]])

axs.axvline(bifts[3],linestyle='--',color='k')

axs.legend(loc=(1.02,0),ncol=2)

axs.set_xlabel('pseudotime')
axs.set_ylabel(r'$\sigma_{\rm exp}^g\ /\ \mu^g_{\rm exp}$ (norm.)')
mf.set_axs_fontsize(axs,20)
#axs.set_yscale('log')


# In[684]:


cov_maxpos  = np.argmax(coeff_vars_normd[0],axis=1)
cov_bifdist = np.abs(cov_maxpos - bifts[3] + ti)
idxs = np.where(cov_bifdist<=4)[0]


# In[679]:


idxs


# In[666]:





# In[547]:


data = [[ 66386, 174296,  75131, 577908,  32015],
        [ 58230, 381139,  78045,  99308, 160454],
        [ 89135,  80552, 152558, 497981, 603535],
        [ 78415,  81858, 150656, 193263,  69638],
        [139361, 331509, 343164, 781380,  52269]]

columns = ('Freeze', 'Wind', 'Flood', 'Quake', 'Hail')
rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]

values = np.arange(0, 2500, 500)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("Loss in ${0}'s".format(value_increment))
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.title('Loss by Disaster')

plt.show()


# In[ ]:


crv_labs = ['{0}\t[{1:0.3f},{2:0.3f}]'.format(gnms[cc_grps[0][srt_ord[i]]], 
                                             evec1_nn_birng[min_load_idxs[srt_ord[i]],srt_ord[i]],
                                             evec1_nn_birng[max_load_idxs[srt_ord[i]],srt_ord[i]]) 
            for i in range(cc_grps[0].shape[0])]


# In[561]:


columns = ('min', 'max')
rows    = [gnms[cc_grps[0][srt_ord[i]]] for i in range(cc_grps[0].shape[0])]
nrows   = len(rows)
cell_text = []
y_offset = np.zeros(len(columns))
data = np.array([[evec1_nn_birng[min_load_idxs[srt_ord[i]],srt_ord[i]],
         evec1_nn_birng[max_load_idxs[srt_ord[i]],srt_ord[i]]] 
        for i in range(cc_grps[0].shape[0])])
dat_fmt = np.vectorize("%.3f".__mod__)(data)


#cell_text = np.array([['{0:.2f}'.format(k)] for k in j] for j in data]
# for row in range(nrows):
#     y_offset = y_offset + data[row]
#     cell_text.append(['%0.3f' % (x / 1000.0) for x in y_offset])


# In[560]:





# In[584]:


fig,axs=plt.subplots()


table_props=the_table.properties()
table_cells=table_props['child_artists']

table_cells[0].get_text().set_color('white')
# for i in range(nrows):
#     the_table[i,-1].get_text().set_color('white')

# Adjust layout to make room for the table:
#plt.subplots_adjust(left=0.2, bottom=0.2)

#plt.ylabel("Loss in ${0}'s".format(value_increment))
#plt.yticks(values * value_increment, ['%d' % val for val in values])
#plt.xticks([])
#plt.title('Loss by Disaster')

#plt.show()


# In[585]:


len(table_cells)


# In[575]:


dir(table_cells[0])


# In[582]:





# In[ ]:




