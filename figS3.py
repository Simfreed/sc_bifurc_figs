import numpy as np
from sklearn.decomposition import PCA
import myfun as mf
import os
import copy

import matplotlib as mpl
from matplotlib import rc, gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as patches

rc('font', **{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{amssymb}') 



###### load stuff #######
datdir  = 'sn_tmax5'
m2 = 3
m1s       = np.load('{0}/m1s.npy'.format(datdir))
gexp      = np.load('{0}/gexp.npy'.format(datdir))
alphas    = np.load('{0}/alphas.npy'.format(datdir))
drv_idxs  = np.load('{0}/driv_idxs.npy'.format(datdir))

dtraj     = np.load('tc_traj/dtraj.npy'.format(datdir))

#######trajectory processing #######
nm1, ncells, ngenes = gexp.shape
nresp = alphas.shape[0]

nidxs = np.where(alphas<0.5)[0]
pidxs = np.where(alphas>0.5)[0]

xidxs = np.where(drv_idxs==0)[0]
yidxs = np.where(drv_idxs==1)[0]

xnidxs = np.intersect1d(nidxs, xidxs)
xpidxs = np.intersect1d(pidxs, xidxs)
ynidxs = np.intersect1d(nidxs, yidxs)
ypidxs = np.intersect1d(pidxs, yidxs)

gexp_mu  = np.mean(gexp,axis=1)
gexp_qi  = np.quantile(gexp,axis=1,q=[0.025,0.975])
gexp_err = np.std(gexp,axis=1)#/np.sqrt(ncells) #np.abs(gexp_qi-gexp_mu)

###### eigenvalue calculation ########
gexps        = np.array([gexp]) # this is because of the pitchfork clustering
gexps_mu     = np.mean(gexps,axis=2)
gexps_cent   = (gexps.transpose((2,0,1,3))-gexps_mu).transpose((1,2,0,3))
sds          = np.std(gexps_cent, axis=2, ddof=1)
covs         = np.array([[np.cov(gexps_cent[i,t].T) for t in range(nm1)] for i in range(gexps.shape[0])])

cov_eig    = [[np.linalg.eig(cov[t]) for t in np.arange(nm1)] for cov in covs]
cov_evals  = np.array([[cov_eig[i][t][0] for t in np.arange(nm1)] for i in range(len(covs))])
cov_evecs  = np.array([[cov_eig[i][t][1] for t in np.arange(nm1)] for i in range(len(covs))])

cov_evec0      = cov_evecs[...,0]
cov_evec0_sign = mf.nz_sign(mf.maxabs(cov_evec0,axis=2))
cov_evec0_ns   = np.real_if_close((cov_evec0.transpose((2,0,1)) * cov_evec0_sign).transpose((1,2,0)))

##### bifurcation pseudotime ######
bif_idxs = np.argmax(cov_evals[:,:,0],axis=1)

########## resampling gexp ################
print('computing null')
nt, nc, ng = gexp.shape
nsamp = 20
npc = 2
pca = PCA(n_components=npc)
gene_shuf_eig1 = np.zeros([nt, nsamp,npc])
#samp_idxs = np.random.choice(nc, size=(nt, nsamp, ))
for t in range(nt):
    gexpt = gexp[t]
    for i in range(nsamp):
        gexp_shuf = np.array([gexpt[:,g][np.random.choice(nc, nc, replace=True)] for g in range(ng)]).T
        pca.fit(gexp_shuf)
        gene_shuf_eig1[t,i]=pca.explained_variance_
#gexp.shape

null_eval_mu  = np.mean(gene_shuf_eig1,axis=1)
null_eval_err = np.std(gene_shuf_eig1,axis=1)
print('done computing null')
############################################

######################################################
## calculate cos th (angle between cov-v1 and gexp)###
######################################################

v1_bif = cov_evec0[0,bif_idxs[0]]
cos_th = gexps[0].dot(v1_bif)/np.linalg.norm(gexps[0],axis=2)
cos_th_bins = np.linspace(-0.75,0.75,16)
cos_th_hist = np.array([np.histogram(cos_th[i],cos_th_bins,density=True)[0] for i in range(cos_th.shape[0])]).T
cos_th_bin_ctrs = 0.5*(cos_th_bins[1:]+cos_th_bins[:-1])

####################################################
############# jacobian calculations ################
####################################################

# there's def a bunch of extra unused code here....
dxdotdx = -1
dxdotdy = lambda m1,y: -2*m1*y/((1+y*y)**2)
dydotdy = -1
dydotdx = lambda m2,x: dxdotdy(m2,x)
dvdotdr = lambda alpha, r: 2*r*(2*alpha-1)/((1+r*r)**2)

jac_m1 = np.zeros((nm1, ncells, ngenes, ngenes))
for i in range(ngenes):
    jac_m1[:,:,i,i]=-1

jac_m1[:,:,0,1]   = dxdotdy(m1s,gexps[0,:,:,1].T).T
jac_m1[:,:,1,0]   = dxdotdy(m2,gexps[0,:,:,0])

for i in range(nresp):
    jac_m1[:,:,i+2,drv_idxs[i]] = dvdotdr(alphas[i],gexps[0,:,:,drv_idxs[i]])

jac_bif_evals, jac_bif_evecs      = np.linalg.eig(jac_m1)
jac_eig_max_idxs                  = np.argmax(jac_bif_evals,axis=2)
jac_max_eval     = np.max(jac_bif_evals,axis=2)
mu_jac_max_eval  = np.mean(jac_max_eval,axis=1)
std_jac_max_eval = np.std(jac_max_eval,axis=1)/np.sqrt(ncells)

jac_evec0    = np.array([[jac_bif_evecs[i,j,:,jac_eig_max_idxs[i,j]]
                          for j in np.arange(jac_eig_max_idxs.shape[1])]
                         for i in np.arange(jac_eig_max_idxs.shape[0])])

errsp         = np.linalg.norm( jac_evec0.transpose((1,0,2))-cov_evec0_ns[0],axis=2)
errsn         = np.linalg.norm(-jac_evec0.transpose((1,0,2))-cov_evec0_ns[0],axis=2)
jac_evec0_sgn = (np.argmin(np.array([errsn,errsp]),axis=0)*2-1).T
jac_evec0_ns  = ((jac_evec0.transpose((2,0,1))) * jac_evec0_sgn).transpose((1,2,0))

jac_m1_muexp = np.zeros((nm1, ngenes, ngenes))
for i in range(ngenes):
    jac_m1_muexp[:,i,i]=-1

jac_m1_muexp[:,0,1]   = dxdotdy(m1s,gexps_mu[0,:,1].T).T
jac_m1_muexp[:,1,0]   = dxdotdy(m2,gexps_mu[0,:,0])

for i in range(nresp):
    jac_m1_muexp[:,i+2,drv_idxs[i]]   = dvdotdr(alphas[i],gexps_mu[0,:,drv_idxs[i]])

jac_muexp_evals, jac_muexp_evecs      = np.linalg.eig(jac_m1_muexp)
jac_muexp_maxeval_idxs                = np.argmax(jac_muexp_evals,axis=1)
jac_muexp_maxevals                    = np.max(jac_muexp_evals,axis=1)

jac_muexp_evec0     = np.array([jac_muexp_evecs[i,:,jac_muexp_maxeval_idxs[i]] for i in np.arange(nm1)])

errsp               = np.linalg.norm( jac_muexp_evec0-cov_evec0_ns[0],axis=1)
errsn               = np.linalg.norm(-jac_muexp_evec0-cov_evec0_ns[0],axis=1)
jac_muexp_evec0_sgn = (np.argmin(np.array([errsn,errsp]),axis=0)*2-1)

jac_muexp_evec0_ns  = ((jac_muexp_evec0.T) * jac_muexp_evec0_sgn).T
jac_evec_err        = np.linalg.norm(jac_muexp_evec0_ns - cov_evec0_ns[0],axis=1)

##########################################################
############### gene sorting by alpha ####################
##########################################################
xasrt = np.argsort(alphas[xidxs])
yasrt = np.argsort(alphas[yidxs])

sd_sd = np.array([np.outer(sds[0,i],sds[0,i]) for i in range(nm1)])
corrs = covs[0] / sd_sd

xcorrs = corrs[:,0,xidxs[xasrt]+2]
ycorrs = corrs[:,1,yidxs[yasrt]+2]

xalphas = alphas[xidxs[xasrt]]
yalphas = alphas[yidxs[yasrt]]


yastr = ['{0:.2f}'.format(i) for i in yalphas]
xastr = ['{0:.2f}'.format(i) for i in xalphas]


# both together...
asrt     = np.argsort(alphas)
rgidxs   = np.arange(2,ngenes)[asrt]
xycorrs  = np.array([corrs[:,drv_idxs[i-2],i] for i in rgidxs]).T
xyalphas = alphas[asrt]
xyastr   = ['{0:.2f}'.format(i) for i in xyalphas]

############# correlation calculations
#corrsxy  = [xcorrs, ycorrs, xycorrs]
#alphasxy = [xalphas, yalphas, xyalphas]
#astrxy   = [xastr, yastr, xyastr]
#
#ridx,cidx=np.tril_indices(ngenes,-1)
#corr_bins = np.linspace(-1,1,21)
#corr_hists = np.array([np.histogram(corrs[i,ridx,cidx],bins=corr_bins,density=True)[0]
#                       for i in range(corrs.shape[0])])
#corr_bin_ctrs = (corr_bins[1:]+corr_bins[:-1])*0.5
#
gexp_mut = gexp_mu.T
#
l2h_idxs = np.hstack([xpidxs,ynidxs])
h2l_idxs = np.hstack([xnidxs,ypidxs])

l2h_asrt = np.argsort(np.abs(alphas[l2h_idxs]-0.5))
h2l_asrt = np.argsort(np.abs(alphas[h2l_idxs]-0.5))
#
#gexp_mu_sort = np.vstack([gexp_mut[[0]],
#                          gexp_mut[l2h_idxs[np.flip(l2h_asrt)]+2],
#                          gexp_mut[h2l_idxs[h2l_asrt]+2],
#                          gexp_mut[[1]]
#                         ])
#
#gexp_mu_sort_norm = mf.norm0to1(gexp_mu_sort,1)
#
gexp_mu_sort2 = np.vstack([mf.norm0to1(gexp_mut[[0]],1),
                          gexp_mut[l2h_idxs[np.flip(l2h_asrt)]+2],
                          gexp_mut[h2l_idxs[h2l_asrt]+2],
                          mf.norm0to1(gexp_mut[[1]],1)
                         ])

texprelim=r'\setlength{\thinmuskip}{0mu}'+\
r'\setlength{\thickmuskip}{0mu}'+\
r'\setlength{\medmuskip}{0mu}'

#######################################################
######### the figure ################################
#######################################################
bifvarlab = r'$m_1$'
bi = np.argmin(np.abs(m1s-3.0)) # the correct bifurcation point
plt.style.reload_library()
plt.style.use('one_col_fig')

wfrac, hfrac = 2,2
marg_ht = 1
leg_ht  = 1
leg_spc = 1.5
bht = 5
b_rem = bht-leg_spc-leg_ht
bc_spc = 1
c_ht = 5
cd_spc = 1
d_ht = 5

marg_wd = 1
a_wd = 6
ab_spc = 5
b_wd = 9
leg_spc_wd = 0.5
leg_wd = 0.5


# row heights
hts = np.array([
    
    marg_ht,
    leg_ht,
    leg_spc,
    b_rem,
    bc_spc,
    c_ht,
    cd_spc,
    d_ht
    
])

wds = np.array([
    marg_wd,
    a_wd,
    ab_spc,
    b_wd,
    leg_spc_wd,
    leg_wd
])

hts = np.array(wfrac*hts,dtype = int)
wds = np.array(hfrac*wds,dtype = int)

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols 

nr = np.sum(hts)
nc = np.sum(wds)

wid = 8.7/2.54 #
#wid = 11.4/2.54
ht  = wid*nr/nc


fig = plt.figure(figsize=(wid, ht), dpi=100) 

gs = gridspec.GridSpec(nr, nc)

axAL = plt.subplot( gs[rs[0]:rs[1], cs[0]:cs[1]]) # gene expression heat map legend
axA  = plt.subplot( gs[rs[2]:     , cs[0]:cs[1]]) # gene expression
axB  = plt.subplot( gs[rs[0]:rs[3], cs[2]:cs[3]]) # e-value
axC  = plt.subplot( gs[rs[4]:rs[5], cs[2]:cs[3]]) # evec
axCL = plt.subplot( gs[rs[4]:rs[5], cs[4]:cs[5]]) # evec
axD1 = plt.subplot( gs[rs[6]:rs[7], cs[2]:cs[3]]) # truth

caps = ['A',   'B',  'C',  'D']
ri = [rs[0],rs[0],rs[4],rs[6]]
ci = [0,cs[1],    cs[1],cs[1]]
xs = [-2,    1.5,    1.5,    1.5]
ys = [0, -2, -2, -2]

for i in range(len(caps)):
    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i], x=xs[i], y=ys[i],fontsize=14, fontweight='bold')
    cap_ax.axis('off')


#####################################
## A: gene expression              ##
#####################################
im = axA.imshow(gexp_mu_sort2,aspect='auto')

axA.set_xticks(np.arange(nm1))
axA.set_xticklabels(['{0:.1f}'.format(i) if np.mod(i,1)<1e-5else '' for i in m1s])

axA.set_xlabel(bifvarlab)
axA.set_ylabel('gene index')

cbar = fig.colorbar(im, cax=axAL, orientation='horizontal', aspect=1)
cbar.set_label(r'$\langle$gene expression$\rangle$',rotation=0,labelpad=2)

axAL.xaxis.set_label_position('top')


dm1            = np.mean(np.diff(m1s))
taulims        = [np.min(m1s)-dm1,np.max(m1s)+dm1]
costh_tau_lims = [-1,cos_th_hist.shape[1]]


#####################################
## B: covariance eigenvalue         ##
#####################################
cols = ['k','gray']
axB.plot(m1s, np.real(cov_evals[0,:,0]),'o-', color=cols[0],fillstyle='none', label=r'data ($\omega_1$)')
axB.set_ylabel(r'$\omega_1$')
axB.set_ylabel(r'cov. eval. 1', labelpad=5)

axB.set_xticklabels([])
axB.set_yticks(np.arange(0,7.,2.0))
axB.set_yticklabels(['{0:.1f}'.format(i) for i in np.arange(0,7.,2.0)])
axB.set_xlim(*taulims)

#####################################
## E: cos the distribution         ##
#####################################
cos_th_hist_masked = np.ma.masked_where(cos_th_hist == 0, cos_th_hist)
cmap = copy.copy(mpl.cm.get_cmap("viridis"))
cmap.set_bad(color='white')
im=axC.imshow(cos_th_hist_masked,aspect='auto',cmap=cmap)#, origin='upper')
#axC.set_ylabel(r'$\hat{g}^m_i\cdot \vec{s}^1_c$',labelpad=2)
#axC.set_ylabel('cov. evec 1 proj.\n'+r'($\hat{g}^m_i\cdot \vec{s}^1_c$)',labelpad=2)
axC.set_ylabel('cov. evec. proj.',labelpad=2)
#axC.text(s=r'$\hat{g}(m_1)\cdot \vec{s}^1_c$',x=0.05,y=0.8,fontsize=10,transform=axC.transAxes)
#axC.set_xlabel(bifvarlab)
axC.set_yticks(np.arange(cos_th_hist.shape[0]))
axC.set_yticklabels(['{0:.1f}'.format(cos_th_bin_ctrs[i]) if (i+1)%4==0 else '' 
                      for i in range(len(cos_th_bin_ctrs)-1,-1,-1)])
axC.set_xticklabels([])
axC.axvline(bi,color='k', linestyle = '--', alpha=0.5)
axC.set_xlim(*costh_tau_lims)

cbar = fig.colorbar(im, cax=axCL, orientation='vertical', aspect=1)
cbar.set_label(r'frequency',rotation=270, labelpad=6)
axCL.yaxis.set_label_position('right')
axCL.yaxis.set_ticks([1,3,5])

#####################################
## F: jacobian evalue / evec       ##
#####################################

axcols = ['r','b']
axD1.errorbar(m1s,mu_jac_max_eval, yerr=std_jac_max_eval, capsize=5, color=axcols[0], 
              marker='s', fillstyle='none', label=r'$\lambda_d$') 
axD1.set_ylabel('max jac. eval.', color=axcols[0])#, labelpad=0)
axD1.tick_params(axis='y', labelcolor=axcols[0])
axD1.spines['left'].set_color(axcols[0])
axD1.spines['right'].set_color(axcols[1])
axD1.set_yticks(np.arange(-0.6,0,0.2))
axD1.set_xlim(*taulims)

axD1.set_xlabel(bifvarlab)
axD1.set_xlabel('control parameter ({0})'.format(bifvarlab))
axD1.axvline(m1s[bi],color='k', linestyle = '--', alpha=0.5)
leg = axD1.legend(loc=(0,0.65),frameon=False,handlelength=1,handletextpad=0.5)
for text in leg.get_texts():
    text.set_color(axcols[0])
#axD1.set_xticklabels([])

#jacobian eigenvector error
axD2=axD1.twinx()
axD2.plot(m1s, jac_evec_err, 'o-',fillstyle='none',color=axcols[1],label=texprelim+r'$||\vec{p}^d-\vec{s}^1||$')
axD2.set_ylabel('jac. evec. err.',rotation=270,labelpad=10,color=axcols[1])
axD2.tick_params(axis='y', labelcolor=axcols[1])
axD2.spines['left'].set_color(axcols[0])
axD2.spines['right'].set_color(axcols[1])
axD2.set_yticks([0.1,0.3,0.5,0.7,0.9])
leg = axD2.legend(loc=(0.46,0.15),frameon=False,handlelength=1,handletextpad=0)
for text in leg.get_texts():
    text.set_color(axcols[1])

figdir = 'figs'
os.makedirs(figdir, exist_ok=True)
plt.savefig('{0}/figS3_noeq.pdf'.format(figdir), bbox_inches='tight')


