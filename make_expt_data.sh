#!/bin/bash
# make_expt_data.sh
#
#   bash make_expt_data.sh
#
# Builds datasets for the two breaking-point experiments. CPU only, minutes.
# Safe to re-run; existing datasets are skipped.

cd /mnt/parscratch/users/acq25mha/ciphergan
set -e

build () {
  local out=$1; shift
  if [ -f "data/$out/dataset.npz" ]; then echo "  data/$out exists"; return; fi
  echo "  building data/$out"
  python make_data.py --corpus mydata.txt --no_plot --out_dir "data/$out" "$@" \
    | grep -E "word-level|train_plain" || true
}

# --- D: symbol count -------------------------------------------------------
# Word-level tokens lift the 27-symbol cap. Sample length is 32 rather than 64
# to keep the sample count near the character-level runs; a word carries far
# more information than a character, so 32 word tokens is not a shorter sample
# in any meaningful sense.
echo "D: symbol count (word-level substitution)"
for n in 50 100 200 500 1000 2000; do
  build w$n --cipher substitution --token_level word --vocab_words $n \
        --sample_length 32
done

# --- E: key length ---------------------------------------------------------
# The period IS the key length for a polyalphabetic cipher. Periods up to 7 are
# already solved once positional encoding is supplied (Section 6.4), so this
# sweep starts where that one stopped. At period 31 with 64-token samples each
# key row is seen at only two positions per sample, which is where a limit is
# expected.
echo "E: key length (polysub period sweep)"
for p in 11 15 21 31; do
  build poly$p --cipher polysub --period $p --cipher_seed 0 --sample_length 64
done

echo
echo "classical baseline on the new conditions (CPU, scales with vocabulary):"
for c in w50 w100 w200 w500 w1000 w2000 poly11 poly15 poly21 poly31; do
  [ -f data/$c/dataset.npz ] || continue
  printf "  %-8s " $c
  python mcmc_baseline.py --npz_path data/$c/dataset.npz --seeds 3 \
    --out results/mcmc/$c.json 2>&1 | grep -oE "solved \(key = 1.0\)  [0-9]+/[0-9]+|wall clock +[0-9.]+ s" | tr '\n' ' '
  echo
done

echo
echo "now: sbatch expt.sbatch"
