#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --job-name=w500budget
#SBATCH --output=logs/w500budget_%j.txt
cd /mnt/parscratch/users/acq25mha/ciphergan
mkdir -p results/mcmc_budget
/mnt/parscratch/users/acq25mha/anaconda/.envs/ciphergan/bin/python \
  mcmc_baseline.py --npz_path data/w500/dataset.npz \
  --seeds 10 --restarts 3 --steps 1470193 \
  --out results/mcmc_budget/w500_threshold.json
