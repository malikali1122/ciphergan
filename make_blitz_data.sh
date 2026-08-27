#!/bin/bash
# make_blitz_data.sh
#
#   bash make_blitz_data.sh
#
# Builds every dataset the blitz array needs. CPU only, a couple of minutes.
# Safe to re-run; existing datasets are skipped.

cd /mnt/parscratch/users/acq25mha/ciphergan
set -e

COMMON="--corpus mydata.txt --sample_length 64 --no_plot"

build () {  # build <out_dir> <extra args...>
  local out=$1; shift
  if [ -f "data/$out/dataset.npz" ]; then
    echo "  data/$out exists, skipping"
    return
  fi
  echo "  building data/$out"
  python make_data.py $COMMON --out_dir "data/$out" "$@" > /dev/null
}

echo "A: period-1 families"
build atbash    --cipher atbash
build affine    --cipher affine --affine_a 5 --affine_b 8
build keyword   --cipher keyword --keyword cryptogam
build composed  --cipher composed --cipher_seed 0 --shift 3

echo "B: period sweep (arbitrary permutation per position, not rotations)"
for p in 1 2 3 5 7; do
  build poly$p --cipher polysub --period $p --cipher_seed 0
done

echo "C: corpus size sweep, substitution throughout"
build c100k --cipher substitution --max_chars 100000
build c300k --cipher substitution --max_chars 300000
build c1m   --cipher substitution --max_chars 1000000
build c3m   --cipher substitution --max_chars 3000000
build cfull --cipher substitution

echo
echo "summary:"
python - <<'PY'
import glob, json, os
import numpy as np
rows = []
for p in sorted(glob.glob("data/*/dataset.npz")):
    d = np.load(p, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    c = m["cipher"]
    rows.append((os.path.basename(os.path.dirname(p)), c["name"], c["period"],
                 c.get("key", ""), d["train_plain"].shape[0],
                 d["eval_plain"].shape[0]))
print(f"  {'dataset':12s} {'cipher':14s} {'p':>2s} {'key':16s} "
      f"{'train':>8s} {'eval':>7s}")
for r in rows:
    print(f"  {r[0]:12s} {r[1]:14s} {r[2]:2d} {str(r[3]):16s} "
          f"{r[4]:8d} {r[5]:7d}")
PY

echo
echo "now: sbatch blitz.sbatch"
