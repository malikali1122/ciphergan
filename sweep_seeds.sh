#!/bin/bash
cd /mnt/parscratch/users/acq25mha/ciphergan
FLAGS="--dataroot . --npz_path data/substitution/dataset.npz \
  --model cipher_cycle_gan --pointwise_G --share_embedding --matched_softness \
  --norm layer --embed_dim 256 --ngf 128 --n_blocks_G 5 --ndf 64 \
  --n_layers_D 3 --kw_D 4 --phase test --num_test 10000"

for s in 0 1 2 3 4 5 6 7 8 9; do
  if [ ! -f checkpoints/subst_s${s}/latest_net_G_A.pth ]; then
    echo "##### seed $s: not finished, skipping"
    continue
  fi
  for e in 5 10 15 20 25 30 40 50 60 80 100; do
    [ -f checkpoints/subst_s${s}/${e}_net_G_A.pth ] || continue
    python eval_cipher.py $FLAGS --name subst_s${s} --epoch $e > /dev/null 2>&1
  done
  echo "##### seed $s #####"
  python select_checkpoint.py --name subst_s${s} \
    --npz_path data/substitution/dataset.npz --show_accuracy | tail -6
done
