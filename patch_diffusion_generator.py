"""Add --netG_kind to CipherCycleGANModel so the generator paradigm is a flag.

Run from the repo root of ciphergan_diff:

    python patch_diffusion_generator.py --check     # report only, no writes
    python patch_diffusion_generator.py             # apply

Idempotent: applying twice is a no-op. Writes models/cipher_cycle_gan_model.py
only if every anchor matched, so a partial patch cannot land.

What it changes, and nothing else:
  1. imports build_generator from models.networks_diffusion
  2. registers --netG_kind {conv,diffusion} and --n_steps
  3. routes the netG_A / netG_B construction through build_generator

--netG_kind conv is the default and reproduces current behaviour exactly:
build_generator("conv", ...) constructs the same SeqGenerator with the same
arguments. That is the point of the check mode below - it verifies the
default path is byte-identical in behaviour before any diffusion run.
"""

import argparse
import os
import sys

TARGET = os.path.join("models", "cipher_cycle_gan_model.py")

OLD_IMPORT = ("from models.networks_seq import SeqGenerator, SeqDiscriminator, "
              "get_norm_layer_1d")
NEW_IMPORT = (OLD_IMPORT
              + "\nfrom models.networks_diffusion import build_generator")

OLD_OPT = '''        parser.add_argument("--straight_through", action="store_true")'''
NEW_OPT = '''        parser.add_argument("--straight_through", action="store_true")
        parser.add_argument("--netG_kind", type=str, default="conv",
                            choices=["conv", "diffusion"],
                            help="generator paradigm. conv = feed-forward "
                                 "SeqGenerator (the default, and what every "
                                 "result before this flag used). diffusion = "
                                 "conditional absorbing-state reverse chain.")
        parser.add_argument("--n_steps", type=int, default=4,
                            help="reverse diffusion steps; --netG_kind "
                                 "diffusion only. n_steps=1 collapses the "
                                 "chain to a single denoising pass, which is "
                                 "NOT a distinct paradigm - use >= 4.")'''

OLD_G = '''        self.netG_A = networks.init_net(
            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False),
                         pos_dim=getattr(opt, "pos_dim", 0),
                         max_len=getattr(opt, "max_len", 512)),
            opt.init_type, opt.init_gain).to(self.device)
        self.netG_B = networks.init_net(
            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False),
                         pos_dim=getattr(opt, "pos_dim", 0),
                         max_len=getattr(opt, "max_len", 512)),
            opt.init_type, opt.init_gain).to(self.device)'''

NEW_G = '''        _kind = getattr(opt, "netG_kind", "conv")
        _gkw = dict(vocab_size=V, embed_dim=E, ngf=opt.ngf,
                    n_blocks=opt.n_blocks_G, norm_layer=norm_layer,
                    pointwise=getattr(opt, "pointwise_G", False),
                    pos_dim=getattr(opt, "pos_dim", 0),
                    max_len=getattr(opt, "max_len", 512))
        if _kind == "diffusion":
            _gkw["n_steps"] = getattr(opt, "n_steps", 4)
            print(f"[cipher] generator: conditional diffusion, "
                  f"{_gkw['n_steps']} reverse steps")
            if _gkw["n_steps"] < 2:
                print("[cipher] WARNING: n_steps < 2 is a single denoising "
                      "pass, not a diffusion chain. Results from this setting "
                      "must not be reported as a second paradigm.")
        self.netG_A = networks.init_net(
            build_generator(_kind, **_gkw),
            opt.init_type, opt.init_gain).to(self.device)
        self.netG_B = networks.init_net(
            build_generator(_kind, **_gkw),
            opt.init_type, opt.init_gain).to(self.device)'''

EDITS = [("import", OLD_IMPORT, NEW_IMPORT),
         ("options", OLD_OPT, NEW_OPT),
         ("generator construction", OLD_G, NEW_G)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"ERROR: {TARGET} not found. Run this from the repo root.")

    src = open(TARGET, encoding="utf-8").read()
    if "\r\n" in src:
        sys.exit(f"ERROR: {TARGET} has CRLF line endings. "
                 f"Run: sed -i 's/\\r$//' {TARGET}")

    if "build_generator" in src:
        print("Already patched - no changes needed.")
        return

    out, failed = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            failed.append(f"  {name}: found {n} matches, expected exactly 1")
            continue
        out = out.replace(old, new)
        print(f"  OK   {name}")

    if failed:
        print("\nPATCH ABORTED - the file differs from what this patch "
              "expects:\n" + "\n".join(failed))
        print("\nNothing was written. Send the file for inspection rather "
              "than editing by hand.")
        sys.exit(1)

    if args.check:
        print("\n--check: all 3 anchors matched. Nothing written.")
        return

    open(TARGET + ".bak_prediff", "w", encoding="utf-8").write(src)
    open(TARGET, "w", encoding="utf-8").write(out)
    print(f"\nWrote {TARGET}")
    print(f"Backup at {TARGET}.bak_prediff")


if __name__ == "__main__":
    main()
