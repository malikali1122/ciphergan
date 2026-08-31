#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=evalloop
#SBATCH --output=logs/evalloop_%j.txt
#SBATCH --error=logs/evalloop_err_%j.txt

# Accuracy for the two loopholing arms, plus the four earlier ones re-read
# so the whole table lands in one place.
#
# NOTE: --loopholing must be passed at eval time for ident_diff4_loop. The
# flag adds a LayerNorm (loop_norm) to the generator, so without it the
# architecture will not match the saved state dict and the load fails.

cd /mnt/parscratch/users/acq25mha/ciphergan_diff
mkdir -p results/diffusion_arms

BASE="--dataroot . --npz_path data/identity/dataset.npz \
  --model cipher_cycle_gan --share_embedding --matched_softness \
  --norm layer --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 \
  --n_layers_D 3 --kw_D 4 --phase test --num_test 10000"

run_arm () {
  local name="$1"; shift
  local f="checkpoints/${name}/latest_net_G_A.pth"
  if [ ! -f "$f" ]; then
    echo "===== ${name}: NO CHECKPOINT - skipped ====="
    return
  fi
  echo ""
  echo "===== ${name} ====="
  python eval_cipher.py $BASE --name "${name}" --epoch latest "$@"
  [ -f "checkpoints/${name}/eval_metrics_latest.json" ] && \
    cp "checkpoints/${name}/eval_metrics_latest.json" \
       "results/diffusion_arms/${name}.json"
}

run_arm ident_diff4_rerun --pointwise_G --netG_kind diffusion --n_steps 4
run_arm ident_diff4_loop  --pointwise_G --netG_kind diffusion --n_steps 4 --loopholing

echo ""
echo "===== FULL TABLE ====="
python - <<'PY'
import json, os
arms = [("ident_conv",        "feed-forward  k=1"),
        ("ident_diff4",       "diffusion     k=1"),
        ("ident_diff4_rerun", "diffusion     k=1  (rerun)"),
        ("ident_diff4_loop",  "diffusion     k=1  + loopholing"),
        ("ident_conv_k3",     "feed-forward  k=3"),
        ("ident_diff4_k3",    "diffusion     k=3")]
for name, label in arms:
    p = f"checkpoints/{name}/eval_metrics_latest.json"
    if not os.path.exists(p):
        print(f"  {label:34s}    (not evaluated)"); continue
    d = json.load(open(p))
    print(f"  {label:34s} acc {d['character_accuracy']:.4f}"
          f"   chance {d.get('chance_accuracy', float('nan')):.4f}")
PY
