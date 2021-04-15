#!/bin/bash
mkdir neutrophil_data
cd neutrophil_data
curl https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_gene_names.txt.gz -o gene_names.txt.gz
curl https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_metadata.txt.gz -o metadata.txt.gz
curl https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_normed_counts.mtx.gz -o gene_expr.mtx.gz
curl https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/stateFate_inVitro_neutrophil_pseudotime.txt.gz -o pseudotime.txt.gz

gunzip gene_names.txt.gz
gunzip metadata.txt.gz
gunzip pseudotime.txt.gz
