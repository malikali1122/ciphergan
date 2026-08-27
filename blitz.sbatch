#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=blitz
#SBATCH --array=0-69%10
#SBATCH --output=logs/bz_%A_%a.txt
#SBATCH --error=logs/bz_err_%A_%a.txt
#
# Three experiments in one array. Configuration identical to frontier.sbatch
# throughout - same 50+50 schedule, so the temperature anneal is on the same
# clock and every condition stays comparable to the existing results.
#
# A. PERIOD-1 FAMILY BLITZ (30 runs, 6 families x 5 seeds)
#    atbash, affine, keyword, composed, plus shift and substitution which are
#    already done and are not repeated here.
#
#    All six are single permutations applied identically at every position, so
#    they are elements of one group. Key spaces are 1, 486, ~10^28 and ~10^28
#    against 27 for shift and 27! for substitution - twenty-eight orders of
#    magnitude. The prediction is that performance is INDISTINGUISHABLE across
#    all of them. If it holds, key space is excluded as a difficulty axis by
#    construction rather than by correlation.
#
# B. PERIOD SWEEP (25 runs, 5 periods x 5 seeds)
#    polysub at periods 1, 2, 3, 5, 7, with arbitrary permutations per row
#    rather than Vigenere's rotations. Period 1 is the control: it must match
#    the group A families, since a period-1 polysub IS a substitution. Any
#    drop from period 1 to period 2 locates the boundary.
#
# C. CORPUS SIZE SWEEP (25 runs, 5 sizes x 5 seeds)
#    Substitution at 100k, 300k, 1M, 3M and the full ~6.1M characters. Tests
#    whether the roughly one-in-two convergence rate is data-limited. Smaller
#    corpora train faster, so these are cheap.
#
# Build the datasets first with make_blitz_data.sh.

cd /mnt/parscratch/users/acq25mha/ciphergan

JOBS=(
  # A: period-1 families
  atbash:0    atbash:1    atbash:2    atbash:3    atbash:4
  affine:0    affine:1    affine:2    affine:3    affine:4
  keyword:0   keyword:1   keyword:2   keyword:3   keyword:4
  composed:0  composed:1  composed:2  composed:3  composed:4
  # B: period sweep
  poly1:0     poly1:1     poly1:2     poly1:3     poly1:4
  poly2:0     poly2:1     poly2:2     poly2:3     poly2:4
  poly3:0     poly3:1     poly3:2     poly3:3     poly3:4
  poly5:0     poly5:1     poly5:2     poly5:3     poly5:4
  poly7:0     poly7:1     poly7:2     poly7:3     poly7:4
  # C: corpus size
  c100k:0     c100k:1     c100k:2     c100k:3     c100k:4
  c300k:0     c300k:1     c300k:2     c300k:3     c300k:4
  c1m:0       c1m:1       c1m:2       c1m:3       c1m:4
  c3m:0       c3m:1       c3m:2       c3m:3       c3m:4
  cfull:0     cfull:1     cfull:2     cfull:3     cfull:4
)

ENTRY=${JOBS[$SLURM_ARRAY_TASK_ID]}
COND=${ENTRY%%:*}
SEED=${ENTRY##*:}

echo "task $SLURM_ARRAY_TASK_ID -> condition $COND seed $SEED"

if [ ! -f data/$COND/dataset.npz ]; then
  echo "MISSING data/$COND/dataset.npz - run make_blitz_data.sh first"
  exit 1
fi

python train.py --dataroot . --npz_path data/$COND/dataset.npz \
  --name bz_${COND}_s${SEED} \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 50 --n_epochs_decay 50 \
  --save_epoch_freq 5 \
  --pair_seed ${SEED} --print_freq 100 --num_threads 8
