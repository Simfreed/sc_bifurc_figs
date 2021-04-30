import numpy as np
import scipy

import myfun as mf
import sys
import os

from matplotlib import gridspec, rc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text',usetex=True)


#############################
####### Graphic loading #####
#############################
pngdir    = 'pngs'
figdir    = 'figs'

cell_nms  = ['hmsc_yellow','neutrophil_red','monocyte_blue','lightning_white',
             'lightning_orange', 'lightning_purple_flip','waddington']
cell_pics = [plt.imread('{0}/{1}.png'.format(pngdir,nm)) for nm in cell_nms]

#############################
###### Bifurcation loading ##
#############################
nb_dir = 'no_bifurc'
sn_dir = 'sn1'
pf_dir = 'pf_scale20'
datdirs = [nb_dir, sn_dir, pf_dir]
bif_vars = [np.load('{0}/bvars.npy'.format(ddir)) for ddir in datdirs]
gexps    = [np.load('{0}/gexp.npy'.format(ddir)) for ddir in datdirs]


#############################
##### Manual clustering #####
#############################
clust_poss_nb = np.array([[0.7,1.72],[2,1]]) # for "linear" (hill coeff 1)
clust_poss_sn = np.array([[0,3],[4,0]]) # for saddle node (hill coeff 2)
clust_poss_pf = np.array([[2,2],[0,4],[4,0]]) # for "pitchfork" (hill coeff 2)

clust_poss = [clust_poss_nb, clust_poss_sn, clust_poss_pf]

clust_dists = [np.array([np.linalg.norm(gexps[i][...,0:2]-clust_poss[i][j], axis=2)
                         for j in range(clust_poss[i].shape[0])])
                        for i in range(len(clust_poss))]

clust_labels      = [np.argmin(cd,axis=0) for cd in clust_dists]
clust_label_bools = [np.array(cl,dtype='bool') for cl in clust_labels]

#############################
###### The figure ###########
#############################
plt.style.use('one_col_fig')

frac = 2

marg_wd1 = 0.5
marg_wd2 = 1
marg_wd3 = 1.5
marg_ht  = 0.5

a_ht    = 8

b_wd    = 3
b_ht    = 3
leg_ht  = 1.5

t_ht    = 6
t_wd    = 12


# row heights
hts = np.array([
    t_ht,
    a_ht - t_ht,
    marg_ht,
    b_ht*2/3,
    b_ht*1/3,
    b_ht*1/3,
    b_ht*1/3,
    b_ht*2/3,
    leg_ht
])

wds = np.array([
    marg_wd1,
    b_wd - marg_wd1,
    marg_wd2,
    b_wd,
    marg_wd2,
    b_wd,
    marg_wd3,
    t_wd
])

hts = np.array(frac*hts,dtype = int)
wds = np.array(frac*wds,dtype = int)

rs = np.cumsum(hts) # starting rows
cs = np.cumsum(wds) # starting cols

nr = np.sum(hts)
nc = np.sum(wds)

wid = 11.4/2.54

ht  = wid*nr/nc

fig = plt.figure(figsize=(wid, ht), dpi=200)

gs   = gridspec.GridSpec(nr, nc)
axA   = plt.subplot(gs[   0 :rs[1]  , cs[0]:cs[5]])

axB0  = plt.subplot(gs[rs[3]:rs[6]  ,   0  :cs[1]])
axBL  = plt.subplot(gs[rs[7]:rs[8]  ,   0  :cs[5]])

axB1L   = plt.subplot(gs[rs[2]:rs[3],     0:cs[1]])
axB1ar1 = plt.subplot(gs[rs[3]:rs[4], cs[1]:cs[2]])
axB1a   = plt.subplot(gs[rs[2]:rs[4], cs[2]:cs[3]])
axB1ar2 = plt.subplot(gs[rs[2]:rs[4], cs[3]:cs[4]])
axB1b   = plt.subplot(gs[rs[2]:rs[4], cs[4]:cs[5]])


axB2L   = plt.subplot(gs[rs[6]:rs[7],     0:cs[1]])
axB2ar1 = plt.subplot(gs[rs[5]:rs[6], cs[1]:cs[2]])
axB2a   = plt.subplot(gs[rs[5]:rs[7], cs[2]:cs[3]])
axB2ar2 = plt.subplot(gs[rs[5]:rs[7], cs[3]:cs[4]])
axB2b   = plt.subplot(gs[rs[5]:rs[7], cs[4]:cs[5]])

#rsb  = np.hstack([[0],rs])
rsC = [0, rs[0],rs[4],rs[8]]
axC  = [plt.subplot( gs[rsC[i]:rsC[i+1],cs[6]:cs[7]], projection = '3d') for i in range(3)] # 3d time series

axAc = plt.subplot(gs[0:1,0:1])
axBc = plt.subplot(gs[rs[1]:rs[2],0:1])
axCc = plt.subplot(gs[0:1,cs[5]:cs[6]])

##########################
#### A: waddington #######
##########################
axA.set_xticks([])
axA.set_yticks([])
axA.imshow(cell_pics[6],extent=[-0.05,1.05,0.1,1.08],clip_on=False)
axA.set_xlim(0,1)
axA.set_ylim(0,1)
axA.axis('off')



#########################
#### B: cell pics #######
#########################
axB    = [axB0, axB1a, axB1b, axB2a, axB2b]
ctypes = [[0],[0,1],[1],[0,1,2],[1,2]]
ncs    = [[12],[5,5],[11],[4,3,4],[5,7]]
ncs    = [[70],[45,35],[75],[20,30,25],[35,40]] # ng = 10
ncs    = [[50],[25,30],[53],[15,25,20],[32,28]] # ng = 8
ncs    = [[35],[20,18],[37],[13,10,13],[21,20]] # ng = 7
ncs    = [[20],[12,10],[21],[7,6,8],[13,10]] # ng = 5
ncs    = [[14],[6,8],[14],[4,5,5],[7,7]] # ng = 4

sds    = [6,1,3,5,24]

ngrid = 4
gsz   = 1
imsz  = 0.6
jsz   = gsz-imsz
xmin     = -ngrid/2
xmax     =  ngrid/2
ymin     = -ngrid/2
ymax     =  ngrid/2

xx = np.arange(xmin+gsz/2, xmax, gsz)
yy = np.arange(ymin+gsz/2, ymax, gsz)

bbox_ctrs  = np.array([[[x,y] for x in xx] for y in yy]).reshape(len(xx)*len(yy),2)
bbox_dists = np.linalg.norm(bbox_ctrs,axis=1)
#bbox_dists = np.max(np.abs(bbox_ctrs),axis=1)
bbox_sort  = np.argsort(bbox_dists)
bbox_ctrs  = bbox_ctrs[bbox_sort]
lims       = np.hstack([np.unique(bbox_dists[bbox_sort],return_index=True)[1],len(bbox_dists)])

for i in range(len(axB)):
    np.random.seed(sds[i])
    ax     = axB[i]

    nc     = ncs[i]
    ctps   = ctypes[i]

    ctpsl  = np.hstack([ctps[j]*np.ones(nc[j],dtype='int') for j in range(len(nc))])
    np.random.shuffle(ctpsl)
    nctot  = ctpsl.shape[0]

    nplttd = 0

    for j in range(len(lims)-1):

        ctrs = bbox_ctrs[lims[j]:lims[j+1]]
        nrem = nctot - nplttd

        if nrem <= 0:
            break

        nims   = min(len(ctrs), nrem)
        imctrs = ctrs[np.random.choice(len(ctrs),nims,replace=False)]

        for k in range(nims):
            x0,y0 = imctrs[k] - gsz/2 + np.random.uniform(0,jsz,2)
            ax.imshow(cell_pics[ctpsl[nplttd+k]], extent=[x0,x0+imsz,y0,y0+imsz])
        nplttd += nims

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

############################
###### B : legend ##########
############################

# legends
labs = ['pluripotent','external stimuli','differentiated A', 'differentiated B']
pics = [cell_pics[i] for i in [0,3,1,2]] #cell_pics[3::-1]

axBL.axis('off')
axBL.set_xticks([])
axBL.set_yticks([])
npics = len(pics)

xmax = 21
ymax = 5
axBL.set_xlim(0,xmax)
axBL.set_ylim(0,ymax)

bbox = axBL.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
aspect_ratio = bbox.width / bbox.height

xmarg1 = 0.25
xmarg2 = 0.25
imwd   = 1
imspcx = 1
xmarg3 = 0

ymarg1 = 0
ymarg2 = 0.1
ymarg3 = 0.75
ymarg4 = 0.1
imht   = imwd*1.7 #aspect_ratio #2.5 #imwd*aspect_ratio
imspcy = (ymax - (ymarg1 + ymarg2 + ymarg3 + ymarg4 + imht*2)) / 1

rect = mpatches.FancyBboxPatch((xmarg1, ymarg1),
                               xmax-xmarg3-xmarg1, ymax-ymarg3-ymarg1,
                               facecolor='white',edgecolor='k',clip_on=False,linewidth=0.1,
                               boxstyle=mpatches.BoxStyle("Round",pad=0.5),zorder=0)

valigns=['center']*npics
#x0 = xmarg1 + xmarg2
#y0 = ymarg1 + ymarg2 + (imht + imspcy)*np.arange(npics)
x0 = xmarg1+xmarg2+np.array([0,0,11,11])
y0 = ymarg1 + ymarg2 + (imht + imspcy)*np.array([1,0,1,0])
for i in range(npics):
    axBL.imshow(pics[i],clip_on=True, extent = [x0[i],x0[i]+imwd,y0[i],y0[i]+imht],zorder=1,
                aspect='auto',origin='upper')
    axBL.text(s=labs[i],x=x0[i]+imsz+imspcx, y=y0[i]+imht/2,fontsize=6,verticalalignment=valigns[i],zorder=1)


axBL.add_artist(rect)


axBL.set_zorder(0)

############################
### B : top / bottom annotations #####
############################

txts = ['saddle node\nbifurcation', 'pitchfork\nbifurcation']
txts = ['one-to-one\nfate change', 'one-to-many\nfate change']
pics = cell_pics[4:6]
txtx = [1.1,1]
txty = [1,0.3]
valigns = ['top','bottom']
lightx0 = [1.15,1.45]
lighty0 = [-0.17,1.3]
lightsz = 0.4
axs = [axB1L, axB2L]

for i in range(len(axs)):
    # text
    ax = axs[i]
    ax.text(s=txts[i],x=txtx[i],y=txty[i],fontsize=6,verticalalignment=valigns[i],horizontalalignment='center')

    # lightning
    #tr = transforms.Affine2D().rotate_deg(180)
    ax.imshow(pics[i],clip_on=False, extent=[lightx0[i],lightx0[i]+lightsz,lighty0[i],lighty0[i]+lightsz])

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    ax.axis('off')

######################################
######## middle arrows ###############
######################################
# first arrows
xs  = [0,0,0,0]
ys  = [0.5,0.5,0.5,0.5]
dxs = [1, 1,1,1]
dys = [0.3,-0.3,0,0]
axs = [axB1ar1, axB2ar1, axB1ar2, axB2ar2]
for i in range(len(axs)):
    ax = axs[i]
    ax.annotate("", xy=(xs[i]++dxs[i], ys[i]+dys[i]), xytext=(xs[i], ys[i]),
                arrowprops=dict(headwidth=3, headlength=2, width=0.5, fc='black', clip_on=False), clip_on=False)

    ax.axis('off')
    ax.set_zorder(1)

###########################
##### C: 3d bifurcations ##
###########################

fs = 6
axs = [0,0,0]
cols = ['goldenrod','brown','purple']

labs = ['pluripotent','lineage 1', 'lineage 2']
ms = np.array(['o','s','v'])

#gs.update(hspace=-0.89)

for ii in range(3):
    ax = axC[ii]
    ax.set_aspect('auto')
    ax.set_box_aspect((2.5,1,1))

    xtit_col = 'white' if ii > 0 else 'black'

    gexp = gexps[ii]
    taus = bif_vars[ii]
    clust_label_bool = clust_label_bools[ii]
    clust_label = clust_labels[ii]
    ntau,ncells,ngenes = gexp.shape

    for t in range(ntau):

        for j in range(clust_poss[ii].shape[0]):
            cidxs = np.where(clust_label[t]==j)[0]
            ts = taus[t]*np.ones(cidxs.shape[0])
            ax.scatter(ts, gexp[t,cidxs,0], gexp[t,cidxs,1],
                       s=4, edgecolor=cols[j], facecolor='none',linewidth=0.5,clip_on=False)

    ax.set_ylabel('gene i', fontsize = fs, labelpad=-19.5)
    ax.set_zlabel('gene j', fontsize=fs,labelpad=-19.5)

    #ax.view_init(5, 290)
    ax.view_init(-30, 255)

    ax.tick_params(axis='z',labelsize=fs)

    ax.set_xticks(np.linspace(taus[0],taus[-1],10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    mf.set_axs_fontsize(ax,fs)
    ax.margins(0)
    ax.autoscale_view('tight')
    ax.dist=5.5
    if ii == 0:
        ax.set_xlabel('external stimuli',labelpad=-19,color=xtit_col)

    ax.patch.set_alpha(0)


####### captions #######
cap_fs = 11
axAc.text(s='A',x=0,y=2,fontsize=cap_fs,verticalalignment='top',horizontalalignment='left')
axAc.axis('off')

axBc.text(s='B',x=0,y=0,fontsize=cap_fs,verticalalignment='top',horizontalalignment='left')
axBc.axis('off')

axCc.text(s='C',x=1.25,y=2,fontsize=cap_fs,verticalalignment='top',horizontalalignment='right')
axCc.axis('off')

os.makedirs(figdir, exist_ok=True)
plt.savefig('{0}/fig1_dev_dyn.pdf'.format(figdir), bbox_inches='tight')
