import scipy.io as scio
import scipy.sparse as sp

fnm='neutrophil_data/gene_expr'
sp.save_npz(fnm + '.npz', scio.mmread(fnm + '.mtx.gz'))
