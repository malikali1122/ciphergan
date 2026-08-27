#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=frontier
#SBATCH --array=0-44%10
#SBATCH --output=logs/fr_%A_%a.txt
#SBATCH --error=logs/fr_err_%A_%a.txt

# Frontier runs. Identical configuration to subst.sbatch in every respect
# except the dataset and the seed: same 50+50 schedule, so the tau anneal is
# on the same clock and the conditions stay comparable.
#
#   vig3, vig7        10 seeds each. The paper's own claim, and the hard rungs.
#   shift             5 seeds. Ladder completeness; expected easy.
#   a7, a10, a14, a20 5 seeds each. The alphabet frontier.
#
# a27 is omitted: it is the same task as substitution, which is already done
# at 10 seeds. Read that row into the frontier plot as the V=27 point.
#
# %10 throttles to ten concurrent tasks so the array does not monopolise the
# partition. Drop it if the queue is empty and you want them all at once.

cd /mnt/parscratch/users/acq25mha/ciphergan

JOBS=(
  vig3:0  vig3:1  vig3:2  vig3:3  vig3:4
  vig3:5  vig3:6  vig3:7  vig3:8  vig3:9
  vig7:0  vig7:1  vig7:2  vig7:3  vig7:4
  vig7:5  vig7:6  vig7:7  vig7:8  vig7:9
  shift:0 shift:1 shift:2 shift:3 shift:4
  a7:0    a7:1    a7:2    a7:3    a7:4
  a10:0   a10:1   a10:2   a10:3   a10:4
  a14:0   a14:1   a14:2   a14:3   a14:4
  a20:0   a20:1   a20:2   a20:3   a20:4
)

ENTRY=${JOBS[$SLURM_ARRAY_TASK_ID]}
COND=${ENTRY%%:*}
SEED=${ENTRY##*:}

echo "task $SLURM_ARRAY_TASK_ID -> condition $COND seed $SEED"

if [ ! -f data/$COND/dataset.npz ]; then
  echo "MISSING data/$COND/dataset.npz - build it with make_data.py first"
  exit 1
fi

python train.py --dataroot . --npz_path data/$COND/dataset.npz \
  --name fr_${COND}_s${SEED} \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 50 --n_epochs_decay 50 \
  --save_epoch_freq 5 \
  --pair_seed ${SEED} --print_freq 100 --num_threads 8
