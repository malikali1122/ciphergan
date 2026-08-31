#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=ident_loop
#SBATCH --array=0-1
#SBATCH --output=logs/identloop_%A_%a.txt
#SBATCH --error=logs/identloop_err_%A_%a.txt

# Identity-cipher control for the conditional diffusion generator.
#
# Task 0: --netG_kind conv       does the PATCHED file still train normally?
# Task 1: --netG_kind diffusion  does the reverse chain train at all?
#
# Identity is the right control: the correct mapping is the identity
# permutation, so any failure here is the plumbing, not the cryptanalysis.
# Task 0 must reproduce your existing identity numbers. If it does not, the
# patch broke something and task 1 tells you nothing.
#
# 20 epochs, not 100. This is a go/no-go on convergence, not a result.
# At 1.5 min/epoch measured on substitution, task 0 is ~30 min and task 1
# ~2h at the 4.1x CPU-measured ratio. Compare the two Elapsed times with
# sacct afterwards to get the REAL GPU ratio, which is the number that
# decides whether substitution is affordable before 1 September.
#
# Flags below are copied from subst.sbatch unchanged, including
# --pointwise_G, so the only difference between the two tasks is the
# generator paradigm.

cd /mnt/parscratch/users/acq25mha/ciphergan_diff

# Both arms are diffusion at k=1. The only difference is loopholing, so any
# gap is attributable to the latent pathway and nothing else.
# Reference points from the previous run, same flags, same 20 epochs:
#   ident_conv   (feed-forward, k=1)  accuracy 0.8590
#   ident_diff4  (diffusion,    k=1)  accuracy 0.3540   <- task 0 repeats this
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  KIND_ARGS="--netG_kind diffusion --n_steps 4"
  RUN_NAME="ident_diff4_rerun"
else
  KIND_ARGS="--netG_kind diffusion --n_steps 4 --loopholing"
  RUN_NAME="ident_diff4_loop"
fi

echo "=== ${RUN_NAME} : ${KIND_ARGS} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train.py --dataroot . --npz_path data/identity/dataset.npz \
  --name ${RUN_NAME} \
  --model cipher_cycle_gan ${KIND_ARGS} \
  --pointwise_G --share_embedding --matched_softness \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 10 --n_epochs_decay 10 \
  --save_epoch_freq 5 \
  --pair_seed 0 --print_freq 100 --num_threads 8
