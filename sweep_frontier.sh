#!/bin/bash
# sweep_frontier.sh
#
#   bash sweep_frontier.sh 2>&1 | tee results/frontier_sweep.txt
#
# For every finished fr_<cond>_s<seed> run: evaluate each saved checkpoint,
# then choose one with select_checkpoint.py using ciphertext only. Skips runs
# that have not finished and checkpoints already evaluated, so it is safe to
# re-run while the array is still going.
#
# Requires patch_dump_pred.py to have been applied, otherwise eval_cipher.py
# writes no pred_<epoch>.npz and selection has nothing to read.

cd /mnt/parscratch/users/acq25mha/ciphergan

EPOCHS="5 10 15 20 25 30 40 50 60 80 100"

# A run is finished when its FINAL epoch checkpoint exists. latest_net_G_A.pth
# is not a completion marker: train.py rewrites it every --save_latest_freq
# iterations while training is still going, so testing for it evaluates
# half-trained models and reports them as results.
FINAL_EPOCH=100

# "bash sweep_frontier.sh partial" includes unfinished runs, clearly marked.
# Useful for watching progress; never for a number that goes in the write-up.
PARTIAL=${1:-}
ARCH="--model cipher_cycle_gan --pointwise_G --share_embedding \
  --matched_softness --norm layer --embed_dim 256 --ngf 128 --n_blocks_G 5 \
  --ndf 64 --n_layers_D 3 --kw_D 4 --phase test --num_test 10000"

shopt -s nullglob
for dir in checkpoints/fr_*/; do
  run=$(basename "$dir")
  cond=${run#fr_}; cond=${cond%_s*}

  if [ ! -f "$dir/${FINAL_EPOCH}_net_G_A.pth" ]; then
    if [ "$PARTIAL" != "partial" ]; then
      last=$(ls "$dir" 2>/dev/null | grep -o '^[0-9]*_net_G_A.pth' \
             | cut -d_ -f1 | sort -n | tail -1)
      echo "##### $run: still training (latest saved epoch ${last:-none}), skipping"
      continue
    fi
    echo "##### $run: PARTIAL - not yet at epoch $FINAL_EPOCH, do not report"
  fi
  if [ ! -f "data/$cond/dataset.npz" ]; then
    echo "##### $run: no data/$cond/dataset.npz, skipping"
    continue
  fi

  for e in $EPOCHS; do
    [ -f "$dir/${e}_net_G_A.pth" ] || continue
    [ -f "$dir/pred_${e}.npz" ] && continue      # already evaluated
    python eval_cipher.py --dataroot . --npz_path "data/$cond/dataset.npz" \
      --name "$run" $ARCH --epoch $e > /dev/null 2>&1
  done

  echo "##### $run #####"
  python select_checkpoint.py --name "$run" \
    --npz_path "data/$cond/dataset.npz" --show_accuracy | tail -4
done

echo
echo "building frontier table"
python frontier_table.py
