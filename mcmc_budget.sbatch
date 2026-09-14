#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=mcmcbudget
#SBATCH --output=logs/mcmcbudget_%j.txt
#SBATCH --error=logs/mcmcbudget_err_%j.txt

# Does the word-level decline come from bigram sparsity, or from a search
# budget that grows linearly in V while the transposition space grows as V^2?
#
# The default budget is max(20000, 800*V) steps per position class. Connor
# (2003, section 9) argues a chain must run long enough for every
# transposition to be proposed at least once, which is a coupon-collector
# bound of k*ln(k) for k = C(V,2).
#
#   cond    V    default    k*ln(k)   ratio
#   w200  201     160800     199161    0.81
#   u200  202     161600     201354    0.80
#
# A single point at the threshold would not separate the two explanations,
# because 1.2x is a small change. This runs a ladder instead: the default,
# the threshold, and four times the threshold. Flat across all three means
# the budget was never the constraint and the bigram-sparsity account in
# discussion 1.4 stands. Rising means it was, and 1.4 and 2.3 need rewriting.
#
# Every other flag matches the reported runs, so the default rung should
# reproduce the published figure (w200 0.7650, u200 0.6020) and acts as the
# control on that.
#
# Roughly 50 minutes in total.

cd /mnt/parscratch/users/acq25mha/ciphergan
PY=/mnt/parscratch/users/acq25mha/anaconda/.envs/ciphergan/bin/python
mkdir -p results/mcmc_budget

$PY - <<'PY'
import math
for name, V in [("w200", 201), ("u200", 202)]:
    k = V * (V - 1) // 2
    print(f"{name}: V={V}  C(V,2)={k}  k*ln(k)={math.ceil(k*math.log(k))}  "
          f"default={max(20000, 800*V)}")
PY

run () {
  local cond=$1 steps=$2 tag=$3
  echo ""
  echo "===== ${cond}  steps=${steps}  (${tag}) ====="
  $PY mcmc_baseline.py \
      --npz_path data/${cond}/dataset.npz \
      --seeds 10 --restarts 3 --steps ${steps} \
      --out results/mcmc_budget/${cond}_${tag}.json
}

# w200: default 160800, threshold 199161, 4x threshold 796644
run w200 160800 default
run w200 199161 threshold
run w200 796644 four_x

# u200: default 161600, threshold 201354, 4x threshold 805416
run u200 161600 default
run u200 201354 threshold
run u200 805416 four_x

echo ""
echo "===== SUMMARY ====="
$PY - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("results/mcmc_budget/*.json")):
    d = json.load(open(p))
    cond, tag = os.path.basename(p)[:-5].split("_", 1)
    rows.append((cond, tag, d["steps"], d["key_recovery_best"],
                 d["key_recovery_mean"], d["character_accuracy_best"],
                 d["n_solved"], len(d["seeds"])))
print(f"{'cond':6} {'budget':10} {'steps':>9} {'key best':>9} "
      f"{'key mean':>9} {'acc best':>9} {'solved':>8}")
for c, t, st, kb, km, ab, ns, n in rows:
    print(f"{c:6} {t:10} {st:>9} {kb:>9.4f} {km:>9.4f} {ab:>9.4f} "
          f"{str(ns)+'/'+str(n):>8}")
print()
print("Flat across the three rungs -> budget was not the constraint;")
print("discussion 1.4 stands as written.")
print("Rising -> the vocabulary axis is confounded by search budget and")
print("1.4 and 2.3 both need rewriting.")
PY
