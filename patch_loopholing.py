"""Add --loopholing to CipherCycleGANModel.

Run from the repo root of ciphergan_diff, AFTER patch_diffusion_generator.py:

    python patch_loopholing.py --check
    python patch_loopholing.py

Idempotent. Aborts without writing if the anchors do not match.
"""

import argparse
import os
import sys

TARGET = os.path.join("models", "cipher_cycle_gan_model.py")

OLD_OPT = '''        parser.add_argument("--n_steps", type=int, default=4,'''
NEW_OPT = '''        parser.add_argument("--loopholing", action="store_true",
                            help="carry the denoiser hidden state between "
                                 "reverse steps on a deterministic latent "
                                 "pathway (Jo et al., ICLR 2026). Without "
                                 "it, each step re-embeds the previous "
                                 "step's softmax and discards everything "
                                 "else the denoiser computed. "
                                 "--netG_kind diffusion only.")
        parser.add_argument("--n_steps", type=int, default=4,'''

OLD_G = '''            _gkw["n_steps"] = getattr(opt, "n_steps", 4)
            print(f"[cipher] generator: conditional diffusion, "
                  f"{_gkw['n_steps']} reverse steps")'''
NEW_G = '''            _gkw["n_steps"] = getattr(opt, "n_steps", 4)
            _gkw["loopholing"] = getattr(opt, "loopholing", False)
            print(f"[cipher] generator: conditional diffusion, "
                  f"{_gkw['n_steps']} reverse steps, "
                  f"loopholing {'ON' if _gkw['loopholing'] else 'off'}")'''

EDITS = [("options", OLD_OPT, NEW_OPT), ("generator kwargs", OLD_G, NEW_G)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"ERROR: {TARGET} not found. Run from the repo root.")
    src = open(TARGET, encoding="utf-8").read()

    if "build_generator" not in src:
        sys.exit("ERROR: run patch_diffusion_generator.py first.")
    if "--loopholing" in src:
        print("Already patched - no changes needed.")
        return

    out, failed = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            failed.append(f"  {name}: found {n} matches, expected 1")
            continue
        out = out.replace(old, new)
        print(f"  OK   {name}")

    if failed:
        print("\nPATCH ABORTED - nothing written:\n" + "\n".join(failed))
        sys.exit(1)
    if args.check:
        print("\n--check: anchors matched. Nothing written.")
        return

    open(TARGET + ".bak_preloop", "w", encoding="utf-8").write(src)
    open(TARGET, "w", encoding="utf-8").write(out)
    print(f"\nWrote {TARGET}\nBackup at {TARGET}.bak_preloop")


if __name__ == "__main__":
    main()
