"""
run_ablation.py
===============

Produces the "cipher complexity vs statistical leakage" table: a result that
needs no trained network, and the baseline every later accuracy number is
read against.

    python run_ablation.py --corpus corpus.txt --out ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import math

from cipher_engine import Vocab, build_dataset
import cipher_stats as cs
from make_data import load_corpus

CONFIGS = [
    ("identity",     dict(cipher_name="identity")),
    ("shift-3",      dict(cipher_name="shift", shift=3)),
    ("substitution", dict(cipher_name="substitution", cipher_seed=0)),
    ("vigenere-3",   dict(cipher_name="vigenere", key=[3, 4, 5])),
    ("vigenere-7",   dict(cipher_name="vigenere", key=list(range(1, 8)))),
    ("vigenere-15",  dict(cipher_name="vigenere", key=list(range(1, 16)))),
]

FIELDS = ["config", "key_space_log10", "cipher_ioc", "cipher_entropy_bits",
          "unigram_profile_sym_kl_bits", "bigram_profile_sym_kl_bits",
          "mutual_information_bits"]


def key_space_log10(name: str, n: int) -> float:
    """Cryptographic key-space size. Deliberately contrasted with leakage:
    substitution has a vastly larger key space than a shift yet leaks just as
    much."""
    if name == "identity":
        return 0.0
    if name.startswith("shift"):
        return math.log10(n - 1)
    if name == "substitution":
        return sum(math.log10(i) for i in range(1, n))
    if name.startswith("vigenere"):
        return int(name.split("-")[1]) * math.log10(n)
    return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=None)
    p.add_argument("--sample_length", type=int, default=100)
    p.add_argument("--min_chars", type=int, default=400_000)
    p.add_argument("--out", default="ablation.csv")
    args = p.parse_args()

    vocab = Vocab()
    V = len(vocab)
    text = load_corpus(args.corpus, args.min_chars)

    rows = []
    for name, kw in CONFIGS:
        ds = build_dataset(text, sample_length=args.sample_length,
                           vocab=vocab, **kw)
        r = cs.profile_report(ds.train_plain, ds.train_cipher, V)
        rows.append({
            "config": name,
            "key_space_log10": round(key_space_log10(name, vocab.n_usable), 2),
            "cipher_ioc": round(r["cipher_ioc"], 5),
            "cipher_entropy_bits": round(r["cipher_entropy_bits"], 4),
            "unigram_profile_sym_kl_bits": round(
                r["unigram_profile_sym_kl_bits"], 6),
            "bigram_profile_sym_kl_bits": round(
                r["bigram_profile_sym_kl_bits"], 6),
            "mutual_information_bits": round(
                cs.mutual_information(ds.eval_plain, ds.eval_cipher, V), 4),
        })

    ref = cs.profile_report(
        build_dataset(text, cipher_name="identity", vocab=vocab).train_plain,
        build_dataset(text, cipher_name="identity", vocab=vocab).train_cipher, V)
    print(f"plaintext IoC = {ref['plain_ioc']:.5f}   "
          f"H(X) = {ref['plain_entropy_bits']:.4f} bits\n")

    w = max(len(f) for f in FIELDS)
    hdr = "  ".join(f"{f:>{w}}" for f in FIELDS)
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print("  ".join(f"{str(row[f]):>{w}}" for f in FIELDS))

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
