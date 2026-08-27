#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=difflm
#SBATCH --output=logs/difflm_%j.txt

cd /mnt/parscratch/users/acq25mha/ciphergan
PY=/mnt/parscratch/users/acq25mha/anaconda/.envs/ciphergan/bin/python
mkdir -p models_diffusion

for c in substitution u200; do
  echo "########## $c ##########"
  $PY diffusion_lm.py train --npz_path data/$c/dataset.npz \
    --out models_diffusion/$c.pt --steps 20000
  $PY diffusion_lm.py discriminate --npz_path data/$c/dataset.npz \
    --model models_diffusion/$c.pt --out results/diffusion_$c.json
done
