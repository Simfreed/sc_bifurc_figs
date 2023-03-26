library(slingshot)
library(SingleCellExperiment)
library(RcppCNPy)
#library(matrixStats)

set.seed(12345) ## for reproducibility
samp_size = 2000
npcs      = 50

datdir = "/Users/sfreedman/Code/sc_bifurc_figs/sn1"

pcsf   = paste(datdir,"gexp_pcs.npy", sep="/")
metf   = paste(datdir,"metadata.txt", sep="/")

pcs_mat <- npyLoad(pcsf)
met_dat <- read.table(metf, sep=",", header=TRUE)

#cell_idxs = sample.int(dim(pcs_mat)[2], samp_size, replace=FALSE)
cell_idxs = c(1:dim(pcs_mat)[2])
ncells    = length(cell_idxs)
print(ncells)

sim <- SingleCellExperiment(assays = List(counts = pcs_mat[1:npcs, cell_idxs]))

#sim$spring       <- as.matrix(met_dat[cell_idxs,c('pc0','pc1')])
sim$spring       <- as.matrix(met_dat[cell_idxs,c('spring_x','spring_y')])
reducedDims(sim) <- SimpleList(spring = sim$spring, spring2 = sim$spring) # because i'm stupid

type_names<-met_dat[cell_idxs, 'state']
unique_types<-unique(type_names)
clust0      = which(unique_types=="State 0")
diff_clusts = which(unique_types!="State 0")
print(length(clust0))
sim$types<-as.integer(factor(type_names,levels=unique_types))

print("running slingshot")
sim    <- slingshot(sim, clusterLabels = 'types', reducedDim = 'spring', start.clus = clust0, end.clus = diff_clusts, approx_points = samp_size) # really should be index of undifferentiated...
curves = slingCurves(sim)

dist_tots  = array(0,dim=length(curves))

projs      = array(0,dim=c(length(curves),samp_size,2))
ords       = array(0,dim=c(length(curves),samp_size))

psts       = array(0,dim=c(length(curves),ncells))
dists      = array(0,dim=c(length(curves),ncells))
wts        = array(0,dim=c(length(curves),ncells))

print(length(curves))
print(dim(curves$curve1$s))

# i^th element of curves[["curve1"]]$w is the probability that the i^th cell belongs to curve1
# i^th element of curves[["curve1"]]$lambda is the psuedotime of the i^th cell along curve1

for (i in 1:length(curves)){
    cnm          = paste("Lineage",i,sep="") # e.g., ==> "curve1", "curve2", etc.
    dist_tots[i] = curves[[cnm]]$dist
    projs[i,,]   = curves[[cnm]]$s
    psts[i,]     = curves[[cnm]]$lambda
    ords[i,]     = curves[[cnm]]$ord
    dists[i,]    = curves[[cnm]]$dist_ind
    wts[i,]      = curves[[cnm]]$w

}

outdir = paste(datdir, paste("slingshot_", samp_size,"pts_",npcs,"pcs",sep=""), sep="/")
dir.create(outdir, showWarnings = FALSE)

print("saving stuff")
print(length(curves))
npySave(paste(outdir, "dist_tots.npy", sep="/"), dist_tots)
npySave(paste(outdir, "dists.npy", sep="/"), dists)
npySave(paste(outdir, "ords.npy", sep="/"), ords)
npySave(paste(outdir, "wts.npy", sep="/"), wts)
npySave(paste(outdir, "psts.npy", sep="/"), psts)
for (i in 1:length(curves)){
    npySave(paste(outdir, paste("proj",i,".npy", sep=""), sep="/"), projs[i,,])
}

#x = SlingshotDataSet(sim)
#ncurves = length(x)
#all_ptimes = as.matrix(colData(sim)[3:19])
#min_ptimes = rowMins(all_ptimes,na.rm=TRUE)
#plotcol <- colors[cut(min_ptimes, breaks=100)]
#plot(reducedDims(sim)$spring, col=plotcol)
#lines(SlingshotDataSet(sim), lwd=2, col='black')


