#!/bin/bash
#SBATCH --job-name=lam --time=01:00:00 --partition=gpu --qos=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G
#SBATCH --array=0-11 --output=logs/%x-%A_%a.out
export SLURM_EXPORT_ENV=ALL
module purge; module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/parscratch/users/$USER/anaconda/.envs/ciphergan
export WANDB_MODE=offline
cd /mnt/parscratch/users/$USER/ciphergan
LAMS=(0.3 0.1 0.03 0.0); SEEDS=(0 1 2)
L=${LAMS[$((SLURM_ARRAY_TASK_ID / 3))]}; S=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
N=lam${L}_s${S}
python train.py --dataroot . --npz_path data/identity/dataset.npz --name $N \
  --model cipher_cycle_gan --pointwise_G --matched_softness --norm none \
  --gan_mode wgangp --lambda_gp 10 --n_critic 5 --lr 1e-4 --beta1 0.0 \
  --ngf 64 --ndf 64 --embed_dim 24 --n_layers_D 1 --batch_size 128 \
  --lambda_A $L --lambda_B $L --pair_seed $S \
  --n_epochs 60 --n_epochs_decay 20 --tau_start 0.5 --tau_end 0.1 \
  --print_freq 100000 --num_threads 8
python eval_cipher.py --dataroot . --npz_path data/identity/dataset.npz --name $N \
  --model cipher_cycle_gan --pointwise_G --matched_softness --norm none \
  --ngf 64 --ndf 64 --embed_dim 24 --n_layers_D 1
