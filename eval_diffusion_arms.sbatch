#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval4
#SBATCH --output=logs/eval4_%j.txt
#SBATCH --error=logs/eval4_err_%j.txt

# Accuracy for the four identity-control arms of the diffusion experiment.
#
#                  k=1 (pointwise)      k=3 (context)
#   conv           ident_conv           ident_conv_k3
#   diffusion      ident_diff4          ident_diff4_k3
#
# The claim this table supports: iterative denoising needs lateral context to
# refine against, but unpaired cipher cracking needs a context-free generator
# for cycle consistency to identify the correct mapping. If that is right,
# only the top-left cell should be high, and the two k=3 cells should fail
# TOGETHER - the conv baseline failing at k=3 is what makes this a statement
# about the task rather than about the diffusion implementation.
#
# Base flags copied verbatim from eval_all.sbatch. Per-arm flags differ only
# where the arms were trained differently.

cd /mnt/parscratch/users/acq25mha/ciphergan_diff
mkdir -p results/diffusion_arms

BASE="--dataroot . --npz_path data/identity/dataset.npz \
  --model cipher_cycle_gan --share_embedding --matched_softness \
  --norm layer --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 \
  --n_layers_D 3 --kw_D 4 --phase test --num_test 10000"

run_arm () {
  local name="$1"; shift
  local extra="$*"
  local f="checkpoints/${name}/latest_net_G_A.pth"
  if [ ! -f "$f" ]; then
    echo "===== ${name}: NO CHECKPOINT (${f}) - skipped ====="
    return
  fi
  echo ""
  echo "===== ${name} ====="
  echo "      extra flags: ${extra}"
  python eval_cipher.py $BASE --name "${name}" --epoch latest ${extra}
  if [ -f "checkpoints/${name}/eval_metrics.json" ]; then
    cp "checkpoints/${name}/eval_metrics.json" \
       "results/diffusion_arms/${name}.json"
  else
    echo "WARNING: no eval_metrics.json written for ${name}."
    echo "         eval_cipher.py may need patch_dump_pred.py applied."
  fi
}

run_arm ident_conv        --pointwise_G
run_arm ident_diff4       --pointwise_G --netG_kind diffusion --n_steps 4
run_arm ident_conv_k3
run_arm ident_diff4_k3    --netG_kind diffusion --n_steps 4

echo ""
echo "===== SUMMARY ====="
python - <<'PY'
import glob, json, os
rows = []
for p in sorted(glob.glob("results/diffusion_arms/*.json")):
    name = os.path.basename(p)[:-5]
    try:
        d = json.load(open(p))
    except Exception as e:
        rows.append((name, f"unreadable: {e}")); continue
    acc = next((d[k] for k in ("accuracy", "acc", "token_accuracy")
                if k in d), None)
    rows.append((name, f"{acc:.4f}" if isinstance(acc, float)
                 else f"keys: {sorted(d)[:6]}"))
if not rows:
    print("No metrics written. Read the per-arm output above instead.")
for n, v in rows:
    print(f"  {n:22s} {v}")
PY
