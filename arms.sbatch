#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=arms
#SBATCH --array=0-1
#SBATCH --output=logs/arm_%A_%a.txt
#SBATCH --error=logs/arm_err_%A_%a.txt

cd /mnt/parscratch/users/acq25mha/ciphergan

COMMON="--dataroot . --npz_path data/identity/dataset.npz \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 --n_epochs 50 --n_epochs_decay 50 \
  --pair_seed 0 --print_freq 100 --num_threads 8"

case $SLURM_ARRAY_TASK_ID in
  0) python train.py $COMMON --name arm_nogp \
       --gan_mode lsgan ;;
  1) python train.py $COMMON --name arm_wgangp \
       --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 ;;
esac
