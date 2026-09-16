#!/bin/bash
# ============================================================================
# train.sbatch - long cipher-GAN training run on Sheffield's Stanage cluster
#
#   sbatch train.sbatch
#
# Resume after a walltime kill (checkpoints are written every 5 epochs):
#   FROM=45 sbatch train.sbatch
# ============================================================================
#SBATCH --job-name=ciphergan
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Stanage's own Python guidance: export this before loading Anaconda.
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2024.02-1
source activate ciphergan

# util/visualizer.py imports wandb unconditionally, even though --use_wandb is
# opt-in. The package must be installed, but compute nodes have no outbound
# network, so force it offline or it will hang trying to reach the API.
export WANDB_MODE=offline
export WANDB_SILENT=true

# Home directories are quota-limited. Keep the working set on parscratch.
export PROJECT=/mnt/parscratch/users/$USER/ciphergan
cd "$PROJECT"
mkdir -p logs

# --------------------------------------------------------------------------
export CORPUS=${CORPUS:-mydata.txt}
export CIPHER=${CIPHER:-substitution}
export LEN=${LEN:-128}
export DATA=${DATA:-data/${CIPHER}_${LEN}}
export NAME=${NAME:-${CIPHER}_${LEN}_cyc}
export EPOCHS=${EPOCHS:-120}
export DECAY=${DECAY:-40}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$DATA/dataset.npz" ]; then
  echo "no dataset found - building it"
  ./run_training.sh data
fi

if [ -n "${FROM:-}" ]; then
  echo "resuming from epoch $FROM"
  ./run_training.sh resume
else
  ./run_training.sh long
fi

echo "=== final evaluation ==="
python eval_cipher.py \
  --dataroot . --npz_path "$DATA/dataset.npz" \
  --name "$NAME" --model cipher_cycle_gan \
  --ngf 96 --ndf 96 --embed_dim 32 --n_blocks_G 3

echo "=== learning curve ==="
./run_training.sh curve | tee "logs/${NAME}_curve.csv"

# ============================================================================
# sweep.sbatch - the six-config sweep as an array job, six GPUs in parallel.
# Save the block below as a separate file and submit with:
#   sbatch --array=0-5 sweep.sbatch
# On the cluster the whole sweep finishes
# in the wall-clock time of one run.
# ============================================================================
: <<'SWEEP'
#!/bin/bash
#SBATCH --job-name=cipher-sweep
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2024.02-1
source activate ciphergan
export WANDB_MODE=offline WANDB_SILENT=true
cd /mnt/parscratch/users/$USER/ciphergan

DATA=data/substitution_128
ARCH="--ngf 96 --ndf 96 --embed_dim 32 --n_blocks_G 3 --n_layers_D 3"
OPTIM="--batch_size 128 --lr 2e-4 --lambda_A 10 --lambda_B 10 --gan_mode lsgan"

TAGS=(base sharp sharper gumbel lam20 nlayers1)
FLAGS=(
  "--tau_start 1.0 --tau_end 0.1"
  "--tau_start 0.5 --tau_end 0.05"
  "--tau_start 0.2 --tau_end 0.02"
  "--tau_start 1.0 --tau_end 0.1 --gumbel"
  "--tau_start 0.5 --tau_end 0.05 --lambda_A 20 --lambda_B 20"
  "--tau_start 0.5 --tau_end 0.05 --n_layers_D 1"
)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
EXTRA=${FLAGS[$SLURM_ARRAY_TASK_ID]}

python train.py --dataroot . --npz_path "$DATA/dataset.npz" \
  --name "sweep_$TAG" --model cipher_cycle_gan $ARCH $OPTIM $EXTRA \
  --n_epochs 30 --n_epochs_decay 10 \
  --save_epoch_freq 10 --print_freq 5000 --display_freq 20000 --num_threads 8

python eval_cipher.py --dataroot . --npz_path "$DATA/dataset.npz" \
  --name "sweep_$TAG" --model cipher_cycle_gan $ARCH
SWEEP
