import numpy as np
import os
import grn_sim as sim

from matplotlib import rc, gridspec
import matplotlib.pyplot as plt

rc('font', **{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{amssymb}') 

# dynamics
g1a=lambda g2,m1,m2,tau: m1*tau/(1+g2*g2)
g1b=lambda g2,m1,m2,tau: np.sqrt(m2*tau/g2-1)

g1dot = lambda g1,g2, m1, m2, tau: m1/(1+g2*g2)-g1/tau
g2dot = lambda g1,g2, m1, m2, tau: m2/(1+g1*g1)-g2/tau

#solve for stuff
sn_m1 = np.arange(2,4.1,0.01)
pf_tau = np.arange(0.1,4.2,0.02)
sn_g1 = [sim.get_yss(3, m1, 1, False) for m1 in sn_m1]
pf_g1 = [sim.get_yss(1,1,tau,False) for tau in pf_tau]

sn_zs = [np.sort(z) for z in sn_g1]
sn_n0 = np.array([z[0] for z in sn_zs])
sn_n1 = np.array([z[2] for z in sn_zs if len(z)>1])
sn_s  = np.array([z[1] for z in sn_zs if len(z)>1])
sn_s_m1  = np.array([sn_m1[i] for i in range(sn_m1.shape[0]) if sn_g1[i].shape[0]>1])


pf_zs     = [np.sort(z) for z in pf_g1]
pf_n0     = np.array([z[0] for z in pf_zs])
pf_n1     = np.array([z[2] for z in pf_zs if len(z)>1])
pf_s      = np.array([z[1] for z in pf_zs if len(z)>1])
pf_s_tau  = np.array([pf_tau[i] for i in range(pf_tau.shape[0]) if pf_g1[i].shape[0]>1])

#### the figure ####
### plt.style.reload_library()
plt.style.use('one_col_fig')
# nr = 90
# nc = 45

spc_ht  = 3
tau_ht  = 3
m1_ht   = 7

marg_wd = 3
m1_wd   = 9
spc_wd  = 2
tau_wd  = 9

# row heights
hts = np.array([

    tau_ht,
    tau_ht,
    tau_ht,
    spc_ht,
    m1_ht
])

wds = np.array([
    marg_wd,
    m1_wd,
    spc_wd,
    tau_wd
])

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols

nr = np.sum(hts)
nc = np.sum(wds)

wid = 8.7/2.54
ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=200)

gs = gridspec.GridSpec(nr, nc, hspace=0)

# g1 vs g2
#axA   = plt.subplot( gs[0    :rs[2], cs[0]:cs[1]]) # m1
axA1  = plt.subplot( gs[0    :rs[0], cs[0]:cs[1]]) # m1
axA2  = plt.subplot( gs[rs[0]:rs[1], cs[0]:cs[1]]) # m1
axA3  = plt.subplot( gs[rs[1]:rs[2], cs[0]:cs[1]]) # m1

axB1  = plt.subplot( gs[0    :rs[0], cs[2]:cs[3]]) # tau
axB2  = plt.subplot( gs[rs[0]:rs[1], cs[2]:cs[3]]) # tau
axB3  = plt.subplot( gs[rs[1]:rs[2], cs[2]:cs[3]]) # tau

axC  = plt.subplot( gs[rs[3]:rs[4], cs[0]:cs[1]]) # g1 vs m1
axD  = plt.subplot( gs[rs[3]:rs[4], cs[2]:cs[3]]) # g1 vs m1

caps = ['A','B','C','D']
ri   = [0,0,rs[3],rs[3]]
ci   = [0,cs[1],0,cs[1]]
ys   = [1,1,2.5,2.5]
for i in range(len(caps)):

    cap_ax=plt.subplot(gs[ri[i]:ri[i]+1,ci[i]:ci[i]+1])
    cap_ax.text(s=caps[i],
                x=0,y=ys[i],fontsize=14, verticalalignment='top',horizontalalignment='left')
    cap_ax.axis('off')


#########################################
###### A-B: phase diagrams ##############
#########################################
taus = np.array([[1,1,1],[0.5,2,4]])
m1s  = np.array([[2,3,6],[1,1,1]])
m2s  = np.array([[3,3,3],[1,1,1]])

lss = ['-','--']

axs  = [[axA1,axA2,axA3],[axB1, axB2, axB3]]

ming1 = 0.1
maxg1 = 10
ming2 = 0.03
maxg2 = 5
nv1d  = 20
npts  = 100

# g1s = np.linspace(0.1,4,100)
# g2s = np.linspace(0.1,4,100)

#g1s = np.linspace(ming1,maxg1,npts)
g2s = np.linspace(ming2,maxg2,npts)

xs = np.linspace(ming2, maxg2, nv1d)
ys = np.linspace(ming1, maxg1, nv1d)

g2s = np.logspace(np.log10(ming2),np.log10(maxg2),npts)
xs = np.logspace(np.log10(ming2)-1, np.log10(maxg2)+1, nv1d)
ys = np.logspace(np.log10(ming1)-1, np.log10(maxg1)+1, nv1d)

xx, yy = np.meshgrid(xs,ys)
xxf    = xx.reshape(-1)
yyf    = yy.reshape(-1)
rs     = np.vstack([xxf,yyf]).T
zfills = ['none','none','none']
zmarks = ['o','s','o']
fc = ['gray','none','gray']
for i in range(len(axs)):
    for j in range(len(axs[i])):
        ax = axs[i][j]
        tau = taus[i,j]
        m1 = m1s[i,j]
        m2 = m2s[i,j]
        g2i = np.where(m2*tau/g2s>=1)[0]

        ncg2  = g2s[g2i]
        ncg1a = g1a(ncg2, m1, m2, tau)
        ncg1b = g1b(ncg2, m1, m2, tau)

        ax.plot(ncg2, ncg1a, color = 'r', ls = lss[0],lw=1,zorder=1)
        ax.plot(ncg2, ncg1b, color = 'b', ls = lss[1],lw=1,zorder=1)

        g2zs = np.sort(sim.get_yss(m1,m2,tau,False))
        g1zs = g1a(g2zs, m1, m2, tau)

        # plot the zeros
        for k in range(g2zs.shape[0]):
            ax.plot(g2zs[k],g1zs[k],color='k',marker=zmarks[k],
                    markeredgewidth=1,markersize=4,alpha=1,zorder=2,markerfacecolor=fc[k])#fillstyle=zfills[k],

        # plot the vector field
        uu = g2dot(yy,xx,m1,m2,tau)
        vv = g1dot(yy,xx,m1,m2,tau)
        uvnorm = np.sqrt(uu*uu + vv*vv)
        uuh = uu/uvnorm
        vvh = vv/uvnorm
        ax.quiver(xxf, yyf, uuh, vvh, color = 'k', width=0.004,
                   headwidth=5, headlength=4,alpha=0.55,pivot='tip',scale=20,zorder=0)

        skip   = 10
        ncg1vs = np.array([ncg1a[::skip], ncg1b[::skip]])
        ncg2v  = ncg2[::skip]
        for k in range(ncg1vs.shape[0]):
            ncg1v   = ncg1vs[k]
            uu     = g2dot(ncg1v,ncg2v,m1,m2,tau)
            vv     = g1dot(ncg1v,ncg2v,m1,m2,tau)
            uvnorm = np.sqrt(uu*uu + vv*vv)
            uuh    = uu/uvnorm
            vvh    = vv/uvnorm
            ax.quiver(ncg2v, ncg1v, uuh, vvh, color = 'k', width=0.004,
                   headwidth=5, headlength=4,alpha=1,pivot='tip',scale=20,zorder=3)

        # format
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(ming2,maxg2)
        ax.set_ylim(ming1,maxg1)


        if i==1:
            ax.set_yticklabels([])
            label=r'$k_D=${0:.2f}'.format(1/tau)
        else:
            label=r'$m_1=${0}'.format(m1)

            if j>0:
                ax.set_yticks([0.1,1])
                ax.set_yticklabels(["0.1","1"])
            else:
                ax.set_yticks([0.1,1,10])
                ax.set_yticklabels(["0.1","1","10"])

        props = dict(boxstyle='round,pad=0.01', facecolor='wheat', alpha=0.5,ec='none')
#         ax.text(x=0.98,y=0.95,s=label,transform=ax.transAxes,
#                 verticalalignment='top', horizontalalignment='right',bbox=props,fontsize=6)
        ax.text(x=0.02,y=0.02,s=label,transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',bbox=props,fontsize=6)

        if j==2:
            ax.set_xticks([0.1,1])
            ax.set_xticklabels(["0.1","1"])



axA3.set_xlabel(r'$g_2$',labelpad=-4)
axB3.set_xlabel(r'$g_2$',labelpad=-4)
axA2.set_ylabel(r'$g_1$')

########################################
#######C: saddle node g1################
########################################
axC.plot(sn_m1, sn_n0,'ko',markersize=0.5)
axC.plot(sn_s_m1, sn_n1,'ko', markersize=0.5)
axC.plot(sn_s_m1, sn_s,'k--',fillstyle='none')

axC.set_xlabel(r'$m_1$')
axC.set_ylabel(r'$g_1$')
########################################
#######D: pitchfork g1################
########################################
axD.plot(pf_tau, pf_n0,'ko',markersize=0.5)
axD.plot(pf_s_tau, pf_n1,'ko',markersize=0.5)
axD.plot(pf_s_tau, pf_s,'k--',fillstyle='none')

axD.set_xlabel(r'$1/k_D$')

figdir = 'figs'
os.makedirs(figdir, exist_ok=True)
plt.savefig('{0}/figS1_dyn_syss.pdf'.format(figdir), bbox_inches='tight')
