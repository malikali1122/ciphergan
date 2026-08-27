#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=expt
#SBATCH --array=0-49%10
#SBATCH --output=logs/ex_%A_%a.txt
#SBATCH --error=logs/ex_err_%A_%a.txt
#
# TWO BREAKING-POINT EXPERIMENTS
#
# D. SYMBOL COUNT (30 runs, 6 vocabularies x 5 seeds)
#    Word-level substitution at 50 to 2,000 types. The character alphabet caps
#    at 27, so this is the only way to extend the frontier of Section 6.1.
#    Positional encoding is OFF: these are period-1 ciphers.
#
#    A confound to report rather than hide. Bigram density falls as 1/V^2:
#    70 counts per cell at 50 types, 6.0 at 200, and 0.31 at 1,000. Below one
#    count per cell the language signal itself is exhausted, so a failure at
#    large V is a property of the corpus as much as of the method. The
#    classical baseline is run on every condition for exactly this reason - if
#    it fails too, the limit is the task.
#
# E. KEY LENGTH (20 runs, 4 periods x 5 seeds)
#    Polysub at periods 11 to 31, with positional encoding ON, picking up where
#    Section 6.4 stopped at period 7. At period 31 each key row appears at
#    roughly two positions per 64-token sample, so a limit is expected
#    somewhere in this range.
#
# Everything else matches Section 5, so both sweeps are comparable to the
# existing results.

cd /mnt/parscratch/users/acq25mha/ciphergan

JOBS=(
  w50:0    w50:1    w50:2    w50:3    w50:4
  w100:0   w100:1   w100:2   w100:3   w100:4
  w200:0   w200:1   w200:2   w200:3   w200:4
  w500:0   w500:1   w500:2   w500:3   w500:4
  w1000:0  w1000:1  w1000:2  w1000:3  w1000:4
  w2000:0  w2000:1  w2000:2  w2000:3  w2000:4
  poly11:0 poly11:1 poly11:2 poly11:3 poly11:4
  poly15:0 poly15:1 poly15:2 poly15:3 poly15:4
  poly21:0 poly21:1 poly21:2 poly21:3 poly21:4
  poly31:0 poly31:1 poly31:2 poly31:3 poly31:4
)

ENTRY=${JOBS[$SLURM_ARRAY_TASK_ID]}
COND=${ENTRY%%:*}
SEED=${ENTRY##*:}

if [ ! -f data/$COND/dataset.npz ]; then
  echo "MISSING data/$COND/dataset.npz - run make_expt_data.sh first"
  exit 1
fi

# Word-level conditions are period 1 and need no positional encoding; the
# period sweep needs it, since Section 6.4 established that without it a
# position-dependent key cannot be represented at all.
case $COND in
  w*)    POS="" ;            NAME=ex_${COND} ;;
  poly*) POS="--pos_dim 32 --max_len 128" ; NAME=ex_${COND}_pos ;;
esac

echo "task $SLURM_ARRAY_TASK_ID -> $COND seed $SEED  ${POS:-(no positional encoding)}"

python train.py --dataroot . --npz_path data/$COND/dataset.npz \
  --name ${NAME}_s${SEED} \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  $POS \
  --norm layer --gan_mode wgangp --use_gp --lambda_gp 10 --n_critic 5 \
  --cycle_loss simplex_l1 \
  --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 --n_layers_D 3 --kw_D 4 \
  --lambda_A 1 --lambda_B 1 --lr 2e-4 --beta1 0.0 --beta2 0.9 \
  --warmup_steps 2500 --batch_size 64 \
  --n_epochs 50 --n_epochs_decay 50 \
  --save_epoch_freq 5 \
  --pair_seed ${SEED} --print_freq 100 --num_threads 8
