#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=28:00:00
#SBATCH --job-name=mcmcbig
#SBATCH --output=logs/mcmcbig_%j.txt

cd /mnt/parscratch/users/acq25mha/ciphergan
PY=/mnt/parscratch/users/acq25mha/anaconda/.envs/ciphergan/bin/python
$PY -V
for c in w1000 w2000; do
  echo "### $c"
  $PY mcmc_baseline.py --npz_path data/$c/dataset.npz --seeds 3 --out results/mcmc/$c.json
done
