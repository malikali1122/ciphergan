#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=expt
#SBATCH --array=0-4%5
#SBATCH --output=logs/ex_%A_%a.txt
#SBATCH --error=logs/ex_err_%A_%a.txt
#
cd /mnt/parscratch/users/acq25mha/ciphergan

JOBS=(u200:0 u200:1 u200:2 u200:3 u200:4)

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
  w*|u*)    POS="" ;            NAME=ex_${COND} ;;
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
