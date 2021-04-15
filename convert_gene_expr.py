import scipy

fnm='/Users/simonfreedman/cqub/bifurc/paper_figs/neutrophil_data/gene_expr'
scipy.sparse.save_npz(fnm + '.npz', scipy.io.mmread(fnm + '.mtx.gz'))
