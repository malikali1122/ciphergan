#!/usr/bin/env bash
# ============================================================================
# run_training.sh - long training runs and sweeps for the cipher task
#
#   ./run_training.sh data      # build dataset from the corpus
#   ./run_training.sh long      # single long run, resumable
#   ./run_training.sh resume    # continue an interrupted long run
#   ./run_training.sh sweep     # short runs over the settings that matter
#   ./run_training.sh curve     # accuracy-vs-epoch curve from saved checkpoints
#
# Run from the root of the pytorch-CycleGAN-and-pix2pix fork.
# ============================================================================
set -euo pipefail

CORPUS=${CORPUS:-corpus.txt}       # plain-text corpus
CIPHER=${CIPHER:-substitution}
LEN=${LEN:-128}                    # sample_length
DATA=${DATA:-data/${CIPHER}_${LEN}}
NAME=${NAME:-${CIPHER}_${LEN}_cyc}
EPOCHS=${EPOCHS:-120}
DECAY=${DECAY:-40}

# Architecture. Scale ngf/ndf up only after a small model has been shown to
# learn - a bigger network hides optimisation problems rather than fixing them.
ARCH="--ngf 96 --ndf 96 --embed_dim 32 --n_blocks_G 3 --n_layers_D 3"
OPTIM="--batch_size 128 --lr 2e-4 --lambda_A 10 --lambda_B 10 --gan_mode lsgan"

case "${1:-long}" in

data)
  python make_data.py \
    --corpus "$CORPUS" \
    --cipher "$CIPHER" \
    --sample_length "$LEN" \
    --split_mode disjoint \
    --eval_fraction 0.1 \
    --out_dir "$DATA"
  echo
  echo "Check stats.json before training. For a monoalphabetic cipher,"
  echo "unigram_profile_sym_kl_bits should be near 0 and mutual_information_bits"
  echo "equal to plain_entropy_bits. If not, the data pipeline is wrong and"
  echo "no amount of training will fix it."
  ;;

long)
  python train.py \
    --dataroot . --npz_path "$DATA/dataset.npz" \
    --name "$NAME" --model cipher_cycle_gan \
    $ARCH $OPTIM \
    --n_epochs "$EPOCHS" --n_epochs_decay "$DECAY" \
    --tau_start 0.5 --tau_end 0.05 \
    --save_epoch_freq 5 --save_latest_freq 5000 \
    --print_freq 2000 --display_freq 10000 \
    --num_threads 4 \
    2>&1 | tee -a "logs/${NAME}.log"
  ;;

resume)
  # Set FROM to the last completed epoch (see checkpoints/$NAME/).
  FROM=${FROM:?set FROM=<last completed epoch>}
  python train.py \
    --dataroot . --npz_path "$DATA/dataset.npz" \
    --name "$NAME" --model cipher_cycle_gan \
    $ARCH $OPTIM \
    --n_epochs "$EPOCHS" --n_epochs_decay "$DECAY" \
    --tau_start 0.5 --tau_end 0.05 \
    --save_epoch_freq 5 --save_latest_freq 5000 \
    --print_freq 2000 --display_freq 10000 \
    --num_threads 4 \
    --continue_train --epoch_count "$FROM" \
    2>&1 | tee -a "logs/${NAME}.log"
  ;;

sweep)
  # Short runs. The point is to find which settings move the needle before
  # committing GPU-days, not to reach final accuracy.
  mkdir -p logs
  for cfg in \
    "base:--tau_start 1.0 --tau_end 0.1" \
    "sharp:--tau_start 0.5 --tau_end 0.05" \
    "sharper:--tau_start 0.2 --tau_end 0.02" \
    "gumbel:--tau_start 1.0 --tau_end 0.1 --gumbel" \
    "lam20:--tau_start 0.5 --tau_end 0.05 --lambda_A 20 --lambda_B 20" \
    "sepdom:--tau_start 0.5 --tau_end 0.05" \
  ; do
    tag="${cfg%%:*}"; flags="${cfg#*:}"
    echo "=== $tag ==="
    python train.py \
      --dataroot . --npz_path "$DATA/dataset.npz" \
      --name "sweep_${tag}" --model cipher_cycle_gan \
      $ARCH $OPTIM $flags \
      --n_epochs 15 --n_epochs_decay 5 \
      --save_epoch_freq 20 --save_latest_freq 2000 \
      --print_freq 5000 --display_freq 20000 --num_threads 4 \
      > "logs/sweep_${tag}.log" 2>&1
    python eval_cipher.py \
      --dataroot . --npz_path "$DATA/dataset.npz" \
      --name "sweep_${tag}" --model cipher_cycle_gan $ARCH \
      2>&1 | grep -E "character accuracy|key recovery"
  done
  ;;

curve)
  # Accuracy against training time. This is the figure for the results
  # chapter; a single final number is much weaker evidence.
  echo "epoch,char_accuracy,key_recovery"
  for ckpt in checkpoints/"$NAME"/*_net_G_A.pth; do
    ep=$(basename "$ckpt" _net_G_A.pth)
    [ "$ep" = "latest" ] && continue
    python eval_cipher.py \
      --dataroot . --npz_path "$DATA/dataset.npz" \
      --name "$NAME" --model cipher_cycle_gan $ARCH --epoch "$ep" \
      2>/dev/null | awk -v e="$ep" '
        /character accuracy/ {a=$3}
        /key recovery/       {k=$3}
        END {print e "," a "," k}'
  done
  ;;

*) echo "unknown mode: $1"; exit 1 ;;
esac

# ============================================================================
# SLURM template - save separately as train.sbatch and submit with
#   sbatch train.sbatch
# Partition and account names are cluster-specific.
# ============================================================================
: <<'SLURM'
#!/bin/bash
#SBATCH --job-name=ciphergan
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

module load Anaconda3
source activate ciphergan

export NAME=substitution_128_cyc
export DATA=data/substitution_128
export EPOCHS=120 DECAY=40

# Checkpoint every 5 epochs so a walltime kill costs at most 5 epochs.
# Resubmit with FROM=<last epoch> ./run_training.sh resume
srun ./run_training.sh long
SLURM
