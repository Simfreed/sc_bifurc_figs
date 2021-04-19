#!/bin/sh

#SBATCH --output=/projects/p31095/simonf/log/mcmc_%j.out
#SBATCH --error=/projects/p31095/simonf/log/mcmc_%j.err
# SBATCH --time=1:00:00
#SBATCH --account=b1020
#SBATCH --partition=b1020
# SBATCH --exclusive
#SBATCH --ntasks-per-node=16

module load python/anaconda3.6
source activate ml3
