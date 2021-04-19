# sc_bifurc_figs
## fig 1 ##
```
python grn_sim_runner.py --dir no_bifurc --nresp 0 --hill 1 --scale 200
python grn_sim_runner.py --dir sn1
python grn_sim_runner.py --dir pf_scale20 --nresp 0 --run_pf
python fig1.py
```
## fig 2 ##
```
python fig2.py
```
## fig s1 ##
```
python figS1.py
```
## figs 3 and S4##
```
python grn_sim_runner.py --dt_save 0.1 --m1_range 2 5 1 --dir tc_traj
python fig3_S4.py
```

## fig s3 ##
```
python grn_sim_runner.py --tmax 5 --dir sn_tmax5
python figS3.py
```

## fig s2 ##
```
python grn_sim_runner.py --run_pf --scale 200 --dir pf_scale200
python figS2.py
```

## fig 4-5##
download neutrophil files from Klein
```
bash download_neutrophil_data.sh
```

the gene expression matrix loads a lot (~3x) faster as an npz file than an mtx file, so convert it, since we have to read it a few times
```
python convert_gene_expr.py
rm -r gene_expr.mtx.gz # optional
```

run the eigenvalue decomposition for the data and nulls -- this takes a few hours I think...
```
python pca_gene_resample.py --neval 1 --nsamp 20 --dir eig
```

```
python fig4_5.py
```

## fig S5 ##
```
python pca_ncell_sampling.py // ~ 1 hr
python pca_bin_width.py // ~ 0.5 hr
```

## figs 6, S6, S7 ##
running the correlations takes like 1.5 hrs minimally -- nice thing to parallelize (see make_corr_cmds.py)
```
python pcc.py --wdir neutrophil_data/corr --bin_sz 1000 --min_ncells 400 --t0 90 --tf 121 
python fig6_S6_S7.py
```
