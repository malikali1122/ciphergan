Code submission — unsupervised cipher cracking with CycleGANs

Repository: github.com/malikali1122/ciphergan
  branch cipher-gan, commit 687162c   — adversarial results
  branch diffusion,  commit 582c911   — second paradigm (Methodology section 9)

Fork of github.com/junyanz/pytorch-CycleGAN-and-pix2pix at commit 2a7afba.

WRITTEN FOR THIS PROJECT
  cipher_engine.py            cipher construction and key tables
  make_data.py                dataset generation from the Brown corpus
  data/cipher_dataset.py      paired/unpaired loading and splits
  models/cipher_cycle_gan_model.py   the cipher-specific model
  models/networks_seq.py      sequence generator and discriminator
  models/networks_diffusion.py       diffusion generator (section 9)
  models/test_contract.py     generator interface tests
  diffusion_lm.py             plaintext diffusion LM and scoring (section 9.2)
  mcmc_baseline.py            classical baseline (section 8)
  eval_cipher.py              metrics and decryption dumps (section 7)
  select_checkpoint.py        label-free selection (section 7.3)
  collect_*.py, *_table.py, summarise_mcmc.py    result collection
  make_figures.py, make_slide_figures.py         figures
  *.sbatch, *.sh              cluster job scripts
  patch_*.py                  incremental fixes applied during development,
                              retained for provenance; all are folded into
                              the committed source

INHERITED AND UNMODIFIED
  train.py, test.py, options/, util/, models/base_model.py,
  models/networks.py, data/base_dataset.py and the remaining dataset classes

Datasets are not included. Regenerate with make_data.py; seeds are fixed, so
every condition in Methodology Table 1 reconstructs exactly.
