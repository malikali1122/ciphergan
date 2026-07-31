"""
make_data.py
============

Command-line front end for the cipher engine.

    python make_data.py --corpus corpus.txt --cipher substitution \
        --sample_length 100 --out_dir data/sub-27

    # cipher-complexity ablation, one command per row of your results table
    python make_data.py --corpus corpus.txt --cipher shift        --shift 3
    python make_data.py --corpus corpus.txt --cipher substitution --cipher_seed 0
    python make_data.py --corpus corpus.txt --cipher vigenere     --key 345

Outputs, per run, into --out_dir:
    dataset.npz     unpaired train banks + paired eval set + true key table
    vocab.txt       CipherGAN-format vocabulary
    key.json        the true key (for evaluation and for the appendix)
    stats.json      the vulnerability metrics
    frequency_invariance.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cipher_engine import Vocab, build_dataset, DEFAULT_ALPHABET
import cipher_stats as cs


SAMPLE_TEXT = (
    "it is a truth universally acknowledged that a single man in possession "
    "of a good fortune must be in want of a wife however little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood this truth is so well fixed in the minds of the "
    "surrounding families that he is considered as the rightful property of "
    "some one or other of their daughters "
)


def load_corpus(path: str | None, min_chars: int) -> str:
    if path:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    # fallback so the pipeline is runnable before you have a corpus wired up
    reps = 1 + min_chars // len(SAMPLE_TEXT)
    return SAMPLE_TEXT * reps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=None, help="path to a plain text file")
    p.add_argument("--cipher", default="substitution",
                   choices=["identity", "shift", "substitution", "vigenere"])
    p.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    p.add_argument("--sample_length", type=int, default=100)
    p.add_argument("--shift", type=int, default=3)
    p.add_argument("--key", default="345", help="vigenere key, digits")
    p.add_argument("--cipher_seed", type=int, default=0)
    p.add_argument("--shuffle_seed", type=int, default=1234)
    p.add_argument("--separate_domains", action="store_true")
    p.add_argument("--split_mode", default="disjoint",
                   choices=["disjoint", "parity"])
    p.add_argument("--eval_fraction", type=float, default=0.1)
    p.add_argument("--min_chars", type=int, default=200_000)
    p.add_argument("--out_dir", default="data/run")
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vocab = Vocab(tuple(args.alphabet))
    text = load_corpus(args.corpus, args.min_chars)

    ds = build_dataset(
        text,
        cipher_name=args.cipher,
        sample_length=args.sample_length,
        vocab=vocab,
        separate_domains=args.separate_domains,
        split_mode=args.split_mode,
        eval_fraction=args.eval_fraction,
        shift=args.shift,
        key=[int(c) for c in args.key],
        cipher_seed=args.cipher_seed,
        shuffle_seed=args.shuffle_seed,
    )
    print(ds.summary())

    # --- vulnerability analysis on the UNPAIRED banks (what the model sees) --
    stats = cs.profile_report(ds.train_plain, ds.train_cipher, len(vocab))
    # ...and mutual information on the PAIRED eval split, where alignment exists
    stats["mutual_information_bits"] = cs.mutual_information(
        ds.eval_plain, ds.eval_cipher, len(vocab))
    stats["cipher"] = ds.cipher.name
    stats["key"] = ds.cipher.key_repr
    stats["period"] = ds.cipher.period

    ds.save_npz(os.path.join(args.out_dir, "dataset.npz"))
    vocab.save(os.path.join(args.out_dir, "vocab.txt"), args.separate_domains)
    with open(os.path.join(args.out_dir, "key.json"), "w") as fh:
        json.dump(ds.cipher.describe(), fh, indent=2)
    with open(os.path.join(args.out_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    if not args.no_plot:
        try:
            cs.plot_invariance(
                ds.train_plain, ds.train_cipher, len(vocab),
                symbols=vocab.symbols,
                title=f"{ds.cipher.name} (key={ds.cipher.key_repr})",
                path=os.path.join(args.out_dir, "frequency_invariance.png"))
        except ImportError:
            print("matplotlib not installed; skipping figure")

    print("\nvulnerability metrics")
    for k, v in stats.items():
        print(f"  {k:34s} {v}")
    print(f"\nwrote -> {args.out_dir}")

    # eyeball check
    print("\nexample plaintext :", vocab.decode(ds.eval_plain[0][:60]))
    if not args.separate_domains:
        print("example ciphertext:", vocab.decode(ds.eval_cipher[0][:60]))
    else:
        print("example ciphertext (indices):", ds.eval_cipher[0][:20])


if __name__ == "__main__":
    main()
