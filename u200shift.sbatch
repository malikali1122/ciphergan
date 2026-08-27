#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=u200shift
#SBATCH --array=0-4%5
#SBATCH --output=logs/u200shift_%A_%a.txt

cd /mnt/parscratch/users/acq25mha/ciphergan
PY=/mnt/parscratch/users/acq25mha/anaconda/.envs/ciphergan/bin/python
SEED=$SLURM_ARRAY_TASK_ID

$PY train.py --dataroot . --npz_path data/u200shift/dataset.npz \
  --name ex_u200shift_s${SEED} \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 50 --n_epochs_decay 50 --save_epoch_freq 5 \
  --pair_seed ${SEED} --print_freq 100 --num_threads 8
