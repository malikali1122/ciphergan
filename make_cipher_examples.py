"""
make_cipher_examples.py
=======================

    python make_cipher_examples.py

Generates the worked-example column for the Methodology section 3.1 table,
so the seven period-1 ciphers are shown rather than only named.

Everything comes from cipher_engine itself, so the examples cannot drift
from the implementation. Notably this includes how each cipher treats the
space symbol, which is part of the 27-symbol alphabet and is enciphered
like any other.

No GPU and no data needed; runs on a login node in under a second.
"""

import numpy as np

import cipher_engine as ce

ALPHA = ce.DEFAULT_ALPHABET          # "abcdefghijklmnopqrstuvwxyz "
V = len(ALPHA) + ce.CROP_AMOUNT      # 28: the alphabet plus the pad reserve
MSG = "attack at dawn"

# (label for the table, factory, kwargs) - defaults match the reported runs
CONDITIONS = [
    ("identity",     ce.make_identity_cipher,     {}),
    ("atbash",       ce.make_atbash_cipher,       {}),
    ("shift",        ce.make_shift_cipher,        dict(shift=3)),
    ("affine",       ce.make_affine_cipher,       dict(a=5, b=8)),
    ("keyword",      ce.make_keyword_cipher,      {}),
    ("substitution", ce.make_substitution_cipher, dict(seed=0)),
    ("composed",     ce.make_composed_cipher,     dict(seed=0, shift=3)),
]


def encode(text):
    """Text to indices. CROP_AMOUNT reserves index 0 for padding."""
    return np.array([[ALPHA.index(ch) + ce.CROP_AMOUNT for ch in text]])


def decode(idx):
    return "".join(ALPHA[i - ce.CROP_AMOUNT] for i in idx)


def main():
    plain = encode(MSG)
    print(f"alphabet {ALPHA!r}  ({len(ALPHA)} symbols, V={V})")
    print(f"plaintext: {MSG!r}\n")

    rows = []
    for label, factory, kw in CONDITIONS:
        cipher = factory(V, **kw)
        out = decode(cipher.encrypt(plain)[0])
        # round-trip check: a wrong table would show up here, not in the table
        back = decode(cipher.decrypt(cipher.encrypt(plain))[0])
        rows.append((label, out, cipher.key_repr, back == MSG))

    w = max(len(r[0]) for r in rows)
    print(f"{'cipher'.ljust(w)}  {'attack at dawn becomes'.ljust(16)}  "
          f"key            round trip")
    for label, out, key, ok in rows:
        print(f"{label.ljust(w)}  {out.ljust(16)}  {key[:14].ljust(14)} "
              f"{'ok' if ok else 'FAILED'}")

    print("\nmarkdown for the report:\n")
    print("| cipher | `attack at dawn` becomes |")
    print("|---|---|")
    for label, out, _, _ in rows:
        print(f"| {label} | `{out}` |")

    if any(not r[3] for r in rows):
        print("\nWARNING: a round trip failed; do not use the table above.")


if __name__ == "__main__":
    main()
