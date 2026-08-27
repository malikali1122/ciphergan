#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=pos
#SBATCH --array=0-34%10
#SBATCH --output=logs/pos_%A_%a.txt
#SBATCH --error=logs/pos_err_%A_%a.txt
#
# Positional encoding on every position-dependent condition, plus poly1 as the
# control. Identical to blitz.sbatch in every other respect, so each run pairs
# directly with its --pos_dim 0 counterpart.
#
# THE PREDICTION THIS TESTS
# Without positional encoding, recovery sits at 1/p: 0.9978, 0.6478, 0.5009,
# 0.2451, 0.1425 at periods 1, 2, 3, 5, 7. A supervised capacity probe confirms
# that ceiling is representational rather than adversarial - given labels and
# unlimited gradient, a pos_dim=0 generator reaches 0.3521 on period 3 against
# a 1/p of 0.3333, while pos_dim=32 reaches 1.0000.
#
# So the architecture can now express the map. Whether the ADVERSARIAL signal
# can find it is the open question, and is what this array answers. Three
# outcomes are all informative:
#   recovery rises to ~1 at every period   -> the limitation was representational
#                                             and is now removed
#   recovery rises but decays with period  -> a real difficulty gradient in p,
#                                             visible for the first time
#   recovery stays at 1/p                  -> the representational account is
#                                             incomplete; something else is wrong
#
# poly1 is the control: it has no position dependence, so adding a positional
# embedding should change nothing. If poly1 degrades, the extra input width is
# hurting the monoalphabetic case and the comparison is confounded.
#
# Checkpoints are NOT compatible with the --pos_dim 0 runs: the flag changes
# the input width of the first convolution. Hence the pos_ name prefix.

cd /mnt/parscratch/users/acq25mha/ciphergan

JOBS=(
  poly1:0  poly1:1  poly1:2  poly1:3  poly1:4
  poly2:0  poly2:1  poly2:2  poly2:3  poly2:4
  poly3:0  poly3:1  poly3:2  poly3:3  poly3:4
  poly5:0  poly5:1  poly5:2  poly5:3  poly5:4
  poly7:0  poly7:1  poly7:2  poly7:3  poly7:4
  vig3:0   vig3:1   vig3:2   vig3:3   vig3:4
  vig7:0   vig7:1   vig7:2   vig7:3   vig7:4
)

ENTRY=${JOBS[$SLURM_ARRAY_TASK_ID]}
COND=${ENTRY%%:*}
SEED=${ENTRY##*:}

echo "task $SLURM_ARRAY_TASK_ID -> condition $COND seed $SEED, positional encoding ON"

if [ ! -f data/$COND/dataset.npz ]; then
  echo "MISSING data/$COND/dataset.npz"
  exit 1
fi

# --max_len must exceed the sample length of 64. 128 leaves headroom without
# adding meaningful parameters: the table is max_len x pos_dim = 4096 weights.
python train.py --dataroot . --npz_path data/$COND/dataset.npz \
  --name pos_${COND}_s${SEED} \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --pos_dim 32 --max_len 128 \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 50 --n_epochs_decay 50 \
  --save_epoch_freq 5 \
  --pair_seed ${SEED} --print_freq 100 --num_threads 8
