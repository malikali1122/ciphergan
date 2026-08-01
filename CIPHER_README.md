# CipherGAN-style cipher cracking on `pytorch-CycleGAN-and-pix2pix`

Four new files, **zero modifications to any existing repo file**. The whole diff
is additive, which keeps the "what did you change" section of the report short
and precise.

```
data/cipher_dataset.py             --dataset_mode cipher
models/networks_seq.py             1D generator, 1D PatchGAN, soft embedding
models/cipher_cycle_gan_model.py   --model cipher_cycle_gan
models/cipher_gan_model.py         --model cipher_gan   (adversarial only)
eval_cipher.py                     CER + unsupervised key recovery
```

Plus the data engine (`cipher_engine.py`, `cipher_stats.py`, `make_data.py`,
`run_ablation.py`) at the repo root.

## Setup

```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix
git checkout 2a7afba          # PIN THIS - see warning below
pip install torch torchvision dominate wandb nltk numpy matplotlib
```

**Pin the commit.** `master` has been refactored for distributed training:
`--gpu_ids` no longer exists, device comes from `opt.device` via `init_ddp()`,
and `BaseModel.setup()` now calls `networks.init_net` itself. The repo's own
`docs/overview.md` still documents the *older* layout. Everything here was
tested against `2a7afba`. Record the commit hash in your methodology chapter.

## Run

```bash
# 1. data + Phase 1 vulnerability analysis
python make_data.py --corpus corpus.txt --cipher substitution \
    --sample_length 64 --out_dir data/brown

# 2. train  (--vocab_size is inferred; do not pass it)
python train.py --dataroot . --npz_path data/brown/dataset.npz \
    --name sub_cyc --model cipher_cycle_gan \
    --lambda_A 10 --lambda_B 10 --batch_size 128 \
    --ngf 64 --ndf 64 --embed_dim 24 --n_blocks_G 2 \
    --n_epochs 40 --n_epochs_decay 10 --tau_start 1.0 --tau_end 0.1 \
    --save_epoch_freq 1 --num_threads 0

# 3. evaluate on the held-out paired split
python eval_cipher.py --dataroot . --npz_path data/brown/dataset.npz \
    --name sub_cyc --model cipher_cycle_gan \
    --ngf 64 --ndf 64 --embed_dim 24 --n_blocks_G 2

# adversarial-only control
python train.py ... --model cipher_gan --name sub_gan
```

Resume with `--continue_train --epoch_count N`. Temperature annealing is
resume-aware (it seeds from `epoch_count`, not zero).

## How it hooks in without touching the repo

| need | hook used |
|---|---|
| vocab size | dataset writes `opt.vocab_size`; `train.py` builds data before model |
| temperature anneal | overrides `update_learning_rate()`, already called per epoch |
| text logging | overrides `compute_visuals()`, already called at `--display_freq` |
| no visdom | `display_id=0` default; `visual_names=[]` since `tensor2im` can't render tokens |

Inherited unchanged: `backward_D_basic`, `optimize_parameters`, `GANLoss`
(`vanilla|lsgan|wgangp`), `ImagePool`, checkpointing, LR schedulers.

## Findings so far (preliminary — CPU, undertrained)

### λ (cycle weight) — my initial advice was wrong

Reasoning that `lambda_A=10` was tuned for pixel-L1 and would swamp the
adversarial term under cross-entropy, I suggested lowering it. **The experiment
says the opposite.** Identical network, batch, and data:

| run | λ | epochs | char accuracy | key recovery |
|---|---:|---:|---:|---:|
| `fast`   | 1  | 4 | 0.051 | 0.037 (= chance) |
| `fast10` | 10 | 2 | 0.219 | 0.074 |

λ=10 beat λ=1 at **half** the training steps. Chance is 0.037.

In hindsight the mechanism is clear and worth stating in the report: cycle-
consistency alone is satisfied by *any* bijection, including the identity, so it
cannot pick the right key on its own. The adversarial term is what selects a
fluent output. But with the cycle term weak, the generator satisfies the
discriminator by collapsing onto generic English regardless of input — fluent
and wrong. The λ=1 samples show exactly that: readable fragments bearing no
relation to the ciphertext. **The cycle term is what forces information
preservation; the adversarial term then chooses among information-preserving
maps.** Keep λ high.

Caveat: the two runs differ in epoch count, and neither is near convergence.
Treat this as a pilot that motivates a proper λ sweep, not a result.

### Discriminator receptive field

Measured by gradient support, not derived:

| `--n_layers_D` | context per score |
|---:|---:|
| 1 | 10 characters |
| 2 | 22 characters |
| 3 | 46 characters (default) |
| 4 | 94 characters |

Reproduce with:

```python
d = SeqDiscriminator(28, 8, 8, n_layers, norm_layer=get_norm_layer_1d("none")).eval()
x = torch.zeros(1, 400, 28, requires_grad=True)
out = d(x); out[0, 0, out.shape[-1] // 2].backward()
nz = (x.grad.abs().sum(-1)[0] > 0).nonzero().flatten()
print(int(nz[-1] - nz[0]) + 1)
```

This gives a falsifiable prediction for the methods chapter: substitution should
crack even at 10 characters, since Phase 1 showed unigram and bigram statistics
survive intact. Vigenère accuracy should collapse once key length approaches the
receptive field.

## Status and open questions

Best result so far is **26.7% character accuracy** (chance 3.7%) after ~1300
optimizer steps on CPU. The task is not solved. Outputs are already
English-like well before they are correct, so **read the samples, do not trust
the losses** — `checkpoints/<name>/samples.txt` gets a decoded block at every
`--display_freq`.

Not yet resolved:

- **Does it converge at all?** Untested beyond ~4 epochs. Run 40+ on a GPU
  before drawing any conclusion. Every number here is from an undertrained model.
- **Temperature schedule.** `1.0 → 0.1` is a guess. `--gumbel` (straight-through)
  is wired up as an ablation and untested.
- **`--cycle_loss l1`** reproduces CycleGAN's original objective for comparison
  against the cross-entropy default. Untested.
- **Embedding initialisation.** `networks.init_weights` matches on `Conv` and
  `Linear`, so `nn.Embedding` keeps PyTorch's default N(0,1) — large relative to
  the initialised conv weights. Worth checking whether scaling it down helps.
