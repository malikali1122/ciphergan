#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00          # Set for 24 hours (adjust based on your needs)
#SBATCH --job-name=CipherGAN
#SBATCH --output=logs/full_train_%j.txt  # %j is replaced by the job ID
#SBATCH --error=logs/full_train_err_%j.txt

# Navigate to your directory
cd /mnt/parscratch/users/acq25mha/ciphergan

# Run the full training command
python train.py --dataroot . --npz_path data/identity/dataset.npz \
  --name cipher_full \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --gan_mode lsgan --use_gp --lambda_gp 10 --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 100 --n_epochs_decay 100 \
  --pair_seed 0 --print_freq 100 --num_threads 8

