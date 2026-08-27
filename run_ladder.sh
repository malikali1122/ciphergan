#!/usr/bin/env bash
# Cipher-family ladder: 27-symbol alphabet, ten seeds per cipher.
#
# identity and substitution already have ten seeds each (checkpoints/identity_s*_gp1
# and checkpoints/a27_s*_gp1), so only these three are outstanding.
#
# The decisive comparison is shift against substitution. Both leak identically -
# same index of coincidence, same unigram profile, same mutual information - but
# their key spaces differ by twenty-five orders of magnitude. If shift wins, key
# space governs difficulty. If they tie, leakage governs.
set -euo pipefail
for d in shift vig3 vig7; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    sbatch --export=ALL,LAMBDA_GP=1,SEED=$s,DATA_OVERRIDE=$d \
           --job-name=L${d}_$s --array=0-0 experiments.sbatch
  done
done
echo "30 jobs submitted. Track with: squeue -u \$USER"
