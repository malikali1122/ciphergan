#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=evalall
#SBATCH --dependency=afterany:11134220
#SBATCH --output=logs/evalall_%j.txt

cd /mnt/parscratch/users/acq25mha/ciphergan
mkdir -p results/arms

FLAGS="--dataroot . --npz_path data/identity/dataset.npz \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 \
  --n_layers_D 3 --kw_D 4 --phase test --num_test 10000"

for arm in arm_nogp arm_wgangp; do
  for e in 5 10 20 40 60 80 100 latest; do
    f=checkpoints/$arm/${e}_net_G_A.pth
    [ -f "$f" ] || continue
    echo "===== $arm epoch $e ====="
    python eval_cipher.py $FLAGS --name $arm --epoch $e
    cp checkpoints/$arm/eval_metrics.json results/arms/${arm}_e${e}.json
  done
done

echo "===== SUMMARY ====="
for j in results/arms/*.json; do
  python -c "import json,sys; d=json.load(open('$j')); \
print(f\"{'$j'.split('/')[-1][:-5]:24s} char {d['character_accuracy']:.4f}  key {d['key_recovery_accuracy']:.4f}\")"
done
