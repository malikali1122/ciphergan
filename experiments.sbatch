#!/bin/bash
# ===========================================================================
# experiments.sbatch - the full experimental grid for the results chapter.
#
#   1. build the datasets (login node, CPU, a few minutes):
#        bash experiments.sbatch data
#
#   2. submit the grid:
#        sbatch --array=0-29 experiments.sbatch
#
#   3. collect tables and figures once complete:
#        python collect.py
#
# 30 runs: a cipher-complexity ladder and an alphabet-size frontier, each with
# three seeds. Three seeds is not optional here - an earlier sweep showed
# run-to-run variance exceeding every between-condition difference, so single
# runs are not evidence.
# ===========================================================================
#SBATCH --job-name=cipher-grid
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# --------------------------------------------------------------- the grid --
# Ladder: cipher complexity at the full 27-symbol alphabet.
LADDER=(identity shift substitution vig3 vig7)
# Frontier: substitution at decreasing alphabet size. The permutation space
# falls from 26! to 6!, isolating scale from every other variable.
FRONTIER=(a7 a10 a14 a20 a27)
SEEDS=(0 1 2)

ALPHA_a7="abcdef "
ALPHA_a10="abcdefghi "
ALPHA_a14="abcdefghijklm "
ALPHA_a20="abcdefghijklmnopqrs "
ALPHA_a27="abcdefghijklmnopqrstuvwxyz "

ARCH="--pointwise_G --matched_softness --straight_through --norm layer \
      --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 32 --n_layers_D 5 --kw_D 15"
OPT="--gan_mode lsgan --use_gp --lambda_gp 10 --lambda_A 1 --lambda_B 1 \
     --lr 2e-4 --beta1 0.0 --batch_size 64 --tau_start 0.5 --tau_end 0.1"
EPOCHS="--n_epochs 40 --n_epochs_decay 10"

# ------------------------------------------------------- dataset building --
if [ "${1:-}" = "data" ]; then
  for c in identity shift substitution; do
    python make_data.py --corpus mydata.txt --cipher $c \
      --sample_length 64 --out_dir data/$c --no_plot
  done
  python make_data.py --corpus mydata.txt --cipher vigenere --key 345 \
    --sample_length 64 --out_dir data/vig3 --no_plot
  python make_data.py --corpus mydata.txt --cipher vigenere --key 3451627 \
    --sample_length 64 --out_dir data/vig7 --no_plot
  for a in a7 a10 a14 a20 a27; do
    v="ALPHA_$a"
    python make_data.py --corpus mydata.txt --cipher substitution \
      --alphabet "${!v}" --sample_length 64 --out_dir data/$a --no_plot
  done
  echo
  echo "Datasets built. Check each stats.json before training:"
  echo "  unigram_profile_sym_kl_bits near 0, and"
  echo "  mutual_information_bits equal to plain_entropy_bits."
  echo "If those do not hold, the data is wrong and training cannot fix it."
  exit 0
fi

# ------------------------------------------------------------ environment --
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/parscratch/users/$USER/anaconda/.envs/ciphergan
export WANDB_MODE=offline WANDB_SILENT=true
cd /mnt/parscratch/users/$USER/ciphergan
mkdir -p logs

# ----------------------------------------------------------- dispatch one --
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -lt 15 ]; then
  DATA=${LADDER[$((I / 3))]}
else
  J=$((I - 15))
  DATA=${FRONTIER[$((J / 3))]}
fi
S=${SEEDS[$((I % 3))]}
NAME="${DATA}_s${S}"

echo "=== task $I : dataset=$DATA seed=$S ==="
nvidia-smi --query-gpu=name --format=csv,noheader

# --print_freq 100 is deliberate. The loss curves are the primary diagnostic;
# a large print_freq leaves loss_log.txt effectively empty.
python train.py --dataroot . --npz_path "data/$DATA/dataset.npz" --name "$NAME" \
  --model cipher_cycle_gan $ARCH $OPT $EPOCHS \
  --pair_seed "$S" --print_freq 100 --display_freq 20000 \
  --save_epoch_freq 50 --num_threads 8

python eval_cipher.py --dataroot . --npz_path "data/$DATA/dataset.npz" \
  --name "$NAME" --model cipher_cycle_gan $ARCH

python plot_curves.py --name "$NAME" || true
