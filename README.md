# sc_bifurc_figs

This is a repo for remaking all the figures in the paper [A dynamical systems treatment of transcriptomic trajectories in hematopoiesis](https://www.biorxiv.org/content/10.1101/2021.05.03.442465v2). 
While the instructions are specific to reproducing the results in the paper, many of the scripts are generic and can be easily applied to other datasets. 

## Minimal Requirements ##
- Latex
- Python 3

While Anaconda is not strictly required, it'll be easiest to have all the correct packages / versions in an Anaconda environment.
Create the environment via:
```
conda env create -f environment.yml
```
and activate it with 
```
conda activate bifurc
```
If not using anaconda, consult environment.yml for package info.


## fig 1 ##
```
python grn_sim_runner.py --dir no_bifurc --nresp 0 --hill 1 --scale 200
python grn_sim_runner.py --dir sn1
python grn_sim_runner.py --dir pf_scale20 --nresp 0 --run_pf
python fig1.py
```
## fig S1 ##
output: model_v_data.pdf
run the notebook: 
```
ipynb/model_v_data.ipynb 
```

## fig 2, S3, S4 -- toy model saddle node ##
outputs: saddle_node.pdf (2), resample.pdf (S3), saddle_node_slingshot (S4)
```
python grn_sim_runner.py --dt_save 0.1 --m1_range 2 5 1 --dir tc_traj
Rscript R/run_slinshot_sim.R
```
run the notebook
```
ipynb/saddle_node.ipynb:
```

## fig S2 -- toy model phase planes##
output: xxx.pdf
```
python phase_planes.py
```
## fig S5 -- toy pitchfork ##
run the notebook:
```
ipynb/pitchfork.ipynb
```
output: xxx.pdf

## fig S6 -- noise induced transition##
run the notebook 
```
ipynb/noise_induced.ipynb
```
output:

## fig S7 -- unequilibrated saddle node##
run the notebook
```
ipynb/unequilibrated.ipynb
```

## Fig S8 -- Small Errors##
ipynb/small_errors.ipynb

## Fig 3,4,S9 ##
download neutrophil files from Klein
```
bash download_neutrophil_data.sh
```
the gene expression matrix loads a lot (~3x) faster as an npz file than an mtx file, so convert it, since we have to read it a few times
```
python convert_gene_expr.py
rm -r neutrophil_data/gene_expr.mtx.gz # optional
```
run the eigenvalue decomposition for the data and nulls -- this takes a few hours I think -- see parallelize directory
```
python pca_gene_resample.py --neval 1 --nsamp 20 --outdir eig
```
run the script to get gene expression trajectories
```
python gexp_trajs.py
```

run the notebook 
```
ipynb/neutrophil.ipynb
```

## Fig S9 -- distributional properties of neutrophil trajectory ##
```
python pca_ncell_sampling.py // ~ 1 hr
python pca_bin_width.py // ~ 0.5 hr
python neut_cov_distr.py
```

## Fig S11 -- slingshot on neutrophil ##

## Fig 5, S12, S13 -- eigenvector and associated calculations ##
run the notebook 
```
ipynb/evec_fig.ipynb
```
outputs: 
- Fig 5 
- Fig S12 -- projection of gene expression onto bifurcation eigenvectors #==> ipynb/evec_fig3.ipynb
- Fig S13 -- distribution of gene expression in different genes #==> ipynb/eigenvector_fig.ipynb
- Fig S14 -- GMM fraction of cells in cluster #==> ipynb/evec_fig2.ipynb has code, but didn't work
