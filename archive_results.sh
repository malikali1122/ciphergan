#!/bin/bash
# archive_results.sh
#
#   bash archive_results.sh
#
# Collects everything the results chapter needs into results_bundle/ and packs
# it as a single tar.gz for download. Re-runnable; the bundle
# directory is rebuilt from scratch each time.
#
# Raw checkpoints (.pth) are deliberately NOT included - they are gigabytes and
# no reported result needs them. Everything derived from them is here.

cd /mnt/parscratch/users/acq25mha/ciphergan
B=results_bundle
rm -rf $B
mkdir -p $B/{tables,selection,metrics,mcmc,curves,logs,samples,code}

echo "== regenerating tables =="
python collect_results.py  > $B/tables/collect_stdout.txt 2>&1
python frontier_table.py   > $B/tables/frontier_stdout.txt 2>&1
cp -r results/tables/* $B/tables/ 2>/dev/null

echo "== selection records =="
# one selection.json per run: which checkpoint was chosen, why, and every
# candidate's score. This is the evidence for the blind-selection protocol.
for f in checkpoints/*/selection.json; do
  [ -f "$f" ] || continue
  run=$(basename "$(dirname "$f")")
  cp "$f" $B/selection/${run}.json
done

echo "== per-epoch metrics =="
for f in checkpoints/*/eval_metrics_*.json; do
  [ -f "$f" ] || continue
  run=$(basename "$(dirname "$f")")
  ep=$(basename "$f" .json); ep=${ep#eval_metrics_}
  cp "$f" $B/metrics/${run}__e${ep}.json
done
cp -r results/arms $B/metrics/arms 2>/dev/null

echo "== classical baseline =="
cp results/mcmc/*.json results/mcmc/*.txt $B/mcmc/ 2>/dev/null

echo "== training logs =="
# loss_log.txt carries the diagnostic columns (Dadv, gp_A/B, pr/pf) that the
# mechanistic claims rest on. Small enough to keep in full.
for f in checkpoints/*/loss_log.txt; do
  [ -f "$f" ] || continue
  cp "$f" $B/logs/$(basename "$(dirname "$f")").txt
done
cp results/frontier_sweep.txt results/subst_selection.txt $B/logs/ 2>/dev/null

echo "== curves =="
for r in arm_nogp arm_wgangp subst_s0 subst_s8 fr_vig3_s0 fr_vig7_s0; do
  [ -d checkpoints/$r ] || continue
  python plot_curves.py --name $r --out $B/curves/${r}.png \
    > $B/curves/${r}_summary.txt 2>&1
done

echo "== qualitative samples =="
# The cipher-in / decrypted / truth triples printed by eval_cipher.py. Worth a
# figure: a solved run and a failed run side by side says more than a number.
for f in $B/logs/*.txt; do :; done
grep -h -A 3 "^cipher in" logs/*.txt 2>/dev/null | head -400 \
  > $B/samples/decryption_examples.txt

echo "== code =="
for f in mcmc_baseline.py select_checkpoint.py collect_results.py \
         frontier_table.py sweep_frontier.sh frontier.sbatch subst.sbatch \
         arms.sbatch eval_all.sbatch make_data.py eval_cipher.py \
         models/cipher_cycle_gan_model.py models/networks_seq.py \
         cipher_engine.py; do
  [ -f "$f" ] && cp "$f" $B/code/$(basename "$f")
done

echo "== manifest =="
{
  echo "CipherGAN results bundle"
  echo "built $(date -u '+%Y-%m-%d %H:%M UTC') on $(hostname)"
  echo
  echo "git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
  echo "git status:"
  git status --porcelain 2>/dev/null | head -30
  echo
  echo "runs with a selection record: $(ls $B/selection | wc -l)"
  echo "per-epoch metric files:       $(ls $B/metrics | wc -l)"
  echo "baseline conditions:          $(ls $B/mcmc/*.json 2>/dev/null | wc -l)"
  echo "training logs:                $(ls $B/logs | wc -l)"
  echo
  echo "datasets:"
  for d in data/*/dataset.npz; do
    [ -f "$d" ] || continue
    python - "$d" <<'PY'
import sys, json, numpy as np
p = sys.argv[1]
d = np.load(p, allow_pickle=True)
m = json.loads(str(d["meta"]))
c = m["cipher"]
print(f"  {p:44s} cipher={c['name']:12s} key={c.get('key',''):8s} "
      f"period={c['period']} V={c['vocab_size']} "
      f"train={d['train_plain'].shape} eval={d['eval_plain'].shape} "
      f"symbols={''.join(str(s) for s in d['symbols'][1:])!r}")
PY
  done
} > $B/MANIFEST.txt 2>&1

tar czf results_bundle.tar.gz $B
echo
echo "bundle: $(du -sh results_bundle.tar.gz | cut -f1)"
cat $B/MANIFEST.txt
echo
echo "bring it down with:"
echo "  scp acq25mha@stanage.shef.ac.uk:/mnt/parscratch/users/acq25mha/ciphergan/results_bundle.tar.gz ."
