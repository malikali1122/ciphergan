"""
cipher_engine.py
================

Framework-agnostic cipher engine for unsupervised cipher cracking.

Design notes
------------
Index conventions are deliberately identical to Cohere-Labs-Community/CipherGAN
(`data/data_generators/cipher_generator.py`) so that output from this module can
be consumed either by that TensorFlow repo or by a PyTorch pipeline:

  * index 0 is reserved for ``<pad>``  (their ``_CROP_AMOUNT = 1``)
  * ciphers permute only indices ``1 .. V-1``
  * ``separate_domains=True`` adds ``V - 1`` to every ciphertext index, pushing
    the cipher alphabet into a disjoint index range. This stops the
    discriminator from getting a free win / free loss on token identity alone.

Every cipher is expressed as a *key table* of shape ``[period, V]`` mapping a
plaintext index to a ciphertext index, given the position modulo ``period``:

    ciphertext[i, j] = key_table[j % period, plaintext[i, j]]

  * shift (Caesar)      -> period 1, table is a rotation
  * substitution (mono) -> period 1, table is an arbitrary permutation
  * vigenere (poly)     -> period len(key), table row k is a rotation by key[k]

This single abstraction is what makes the "cipher complexity" ablation in your
project plan cheap: only ``period`` and the permutation structure change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PAD_TOKEN = "<pad>"
CROP_AMOUNT = 1  # number of reserved indices at the front of the vocab
DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz "


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    """Character-level vocabulary with a reserved <pad> at index 0."""

    def __init__(self, symbols: Sequence[str] = tuple(DEFAULT_ALPHABET)):
        self.symbols: List[str] = [PAD_TOKEN] + list(symbols)
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.itos: Dict[int, str] = {i: s for s, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def n_usable(self) -> int:
        """Number of non-reserved symbols (the size of the cipher alphabet)."""
        return len(self.symbols) - CROP_AMOUNT

    def clean(self, text: str) -> str:
        """Lowercase and drop anything outside the alphabet, collapsing spaces."""
        allowed = set(self.symbols[CROP_AMOUNT:])
        out, prev_space = [], False
        for ch in text.lower():
            if ch.isspace():
                ch = " "
            if ch not in allowed:
                continue
            if ch == " ":
                if prev_space:
                    continue
                prev_space = True
            else:
                prev_space = False
            out.append(ch)
        return "".join(out).strip()

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[c] for c in text], dtype=np.int64)

    def decode(self, indices: Sequence[int], strip_pad: bool = True) -> str:
        chars = []
        for i in indices:
            tok = self.itos.get(int(i), "?")
            if tok == PAD_TOKEN:
                if strip_pad:
                    continue
                tok = "_"
            chars.append(tok)
        return "".join(chars)

    def save(self, path: str, separate_domains: bool = False) -> None:
        """Write a vocab file in the CipherGAN one-token-per-line format."""
        tokens = list(self.symbols)
        if separate_domains:
            # cipher-domain copies of every usable symbol, in index order
            tokens += [f"{s}#c" for s in self.symbols[CROP_AMOUNT:]]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(tokens) + "\n")


# --------------------------------------------------------------------------- #
# Ciphers
# --------------------------------------------------------------------------- #
@dataclass
class Cipher:
    """A keyed cipher represented as a position-dependent index permutation."""

    name: str
    key_table: np.ndarray  # [period, V] int64; row k = mapping at position k
    vocab_size: int
    period: int
    key_repr: str = ""
    separate_domains: bool = False

    @property
    def domain_offset(self) -> int:
        return (self.vocab_size - CROP_AMOUNT) if self.separate_domains else 0

    @property
    def total_vocab_size(self) -> int:
        """Vocab size the model must span (both domains if separated)."""
        return self.vocab_size + self.domain_offset

    def encrypt(self, plain: np.ndarray) -> np.ndarray:
        """Encrypt an int array of shape [n_samples, length] (or [length])."""
        plain = np.atleast_2d(plain)
        n, length = plain.shape
        pos = np.arange(length) % self.period
        cipher = self.key_table[pos[None, :].repeat(n, axis=0), plain]
        # never shift the pad symbol out of index 0
        pad_mask = plain == 0
        cipher = cipher + self.domain_offset
        cipher[pad_mask] = 0
        return cipher.astype(np.int64)

    def decrypt(self, cipher: np.ndarray) -> np.ndarray:
        """Ground-truth decryption. For evaluation only - never for training."""
        cipher = np.atleast_2d(cipher).copy()
        pad_mask = cipher == 0
        cipher = cipher - self.domain_offset
        n, length = cipher.shape
        inverse = self.inverse_table()
        pos = np.arange(length) % self.period
        cipher = np.clip(cipher, 0, self.vocab_size - 1)
        plain = inverse[pos[None, :].repeat(n, axis=0), cipher]
        plain[pad_mask] = 0
        return plain.astype(np.int64)

    def inverse_table(self) -> np.ndarray:
        inv = np.zeros_like(self.key_table)
        for k in range(self.period):
            inv[k, self.key_table[k]] = np.arange(self.vocab_size)
        return inv

    def describe(self) -> Dict:
        return {
            "name": self.name,
            "key": self.key_repr,
            "period": self.period,
            "vocab_size": self.vocab_size,
            "separate_domains": self.separate_domains,
            "domain_offset": self.domain_offset,
            "key_table": self.key_table.tolist(),
        }


def _identity_row(V: int) -> np.ndarray:
    return np.arange(V, dtype=np.int64)


def _rotation_row(V: int, shift: int) -> np.ndarray:
    """Rotate only the usable indices; index 0 (<pad>) maps to itself."""
    row = _identity_row(V)
    usable = np.arange(CROP_AMOUNT, V)
    n = len(usable)
    row[usable] = usable[(np.arange(n) + shift) % n]
    return row


def make_shift_cipher(V: int, shift: int = 3, **kw) -> Cipher:
    """Caesar / shift cipher. Weakest: only |alphabet| possible keys."""
    return Cipher("shift", _rotation_row(V, shift)[None, :], V, 1,
                  key_repr=str(shift), **kw)


def make_substitution_cipher(V: int, seed: Optional[int] = 0, **kw) -> Cipher:
    """Monoalphabetic substitution: an arbitrary permutation of the alphabet.

    Key space is (V-1)! - astronomically larger than a shift - but the unigram
    and n-gram *frequency profile* is completely preserved, which is the
    statistical vulnerability the GAN exploits.
    """
    rng = np.random.default_rng(seed)
    row = _identity_row(V)
    usable = np.arange(CROP_AMOUNT, V)
    row[usable] = rng.permutation(usable)
    key_repr = "perm" + ("" if seed is None else f"-seed{seed}")
    return Cipher("substitution", row[None, :], V, 1, key_repr=key_repr, **kw)


def make_vigenere_cipher(V: int, key: Sequence[int] = (3, 4, 5), **kw) -> Cipher:
    """Polyalphabetic: a different rotation per position. Flattens unigram stats."""
    table = np.stack([_rotation_row(V, int(k)) for k in key])
    return Cipher("vigenere", table, V, len(key),
                  key_repr="".join(str(int(k)) for k in key), **kw)


def make_identity_cipher(V: int, **kw) -> Cipher:
    """Sanity-check control: the task is trivially solvable if the code is right."""
    return Cipher("identity", _identity_row(V)[None, :], V, 1, key_repr="0", **kw)


CIPHER_FACTORIES = {
    "identity": make_identity_cipher,
    "shift": make_shift_cipher,
    "substitution": make_substitution_cipher,
    "vigenere": make_vigenere_cipher,
}


def build_cipher(name: str, vocab_size: int, separate_domains: bool = False,
                 shift: int = 3, key: Sequence[int] = (3, 4, 5),
                 seed: Optional[int] = 0) -> Cipher:
    if name == "shift":
        return make_shift_cipher(vocab_size, shift=shift,
                                 separate_domains=separate_domains)
    if name == "substitution":
        return make_substitution_cipher(vocab_size, seed=seed,
                                        separate_domains=separate_domains)
    if name == "vigenere":
        return make_vigenere_cipher(vocab_size, key=key,
                                    separate_domains=separate_domains)
    if name == "identity":
        return make_identity_cipher(vocab_size, separate_domains=separate_domains)
    raise ValueError(f"unknown cipher {name!r}; choose from {list(CIPHER_FACTORIES)}")


# --------------------------------------------------------------------------- #
# Corpus -> fixed-length samples
# --------------------------------------------------------------------------- #
def chunk(indices: np.ndarray, length: int, stride: Optional[int] = None
          ) -> np.ndarray:
    """Cut a flat index stream into [n, length] non-overlapping samples."""
    stride = stride or length
    n = 1 + (len(indices) - length) // stride if len(indices) >= length else 0
    if n <= 0:
        raise ValueError("corpus is shorter than one sample")
    starts = np.arange(n) * stride
    return np.stack([indices[s:s + length] for s in starts])


# --------------------------------------------------------------------------- #
# The unpaired dataset
# --------------------------------------------------------------------------- #
@dataclass
class CipherDataset:
    """Unpaired plaintext / ciphertext banks plus a held-out paired eval set."""

    vocab: Vocab
    cipher: Cipher
    train_plain: np.ndarray      # [n_a, L] domain X, plaintext only
    train_cipher: np.ndarray     # [n_b, L] domain Y, ciphertext only
    eval_plain: np.ndarray       # [n_e, L] paired - EVALUATION ONLY
    eval_cipher: np.ndarray      # [n_e, L] paired - EVALUATION ONLY
    split_mode: str = "disjoint"
    meta: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"cipher={self.cipher.name} key={self.cipher.key_repr} "
            f"period={self.cipher.period}\n"
            f"vocab={len(self.vocab)} usable={self.vocab.n_usable} "
            f"total(model)={self.cipher.total_vocab_size} "
            f"separate_domains={self.cipher.separate_domains}\n"
            f"train_plain={self.train_plain.shape} "
            f"train_cipher={self.train_cipher.shape} "
            f"eval={self.eval_plain.shape} split={self.split_mode}"
        )

    def save_npz(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            train_plain=self.train_plain,
            train_cipher=self.train_cipher,
            eval_plain=self.eval_plain,
            eval_cipher=self.eval_cipher,
            key_table=self.cipher.key_table,
            symbols=np.array(self.vocab.symbols, dtype=object),
            meta=np.array(json.dumps(self.meta), dtype=object),
        )


def build_dataset(text: str,
                  cipher_name: str = "substitution",
                  sample_length: int = 100,
                  vocab: Optional[Vocab] = None,
                  separate_domains: bool = False,
                  split_mode: str = "disjoint",
                  eval_fraction: float = 0.1,
                  shift: int = 3,
                  key: Sequence[int] = (3, 4, 5),
                  cipher_seed: int = 0,
                  shuffle_seed: int = 1234,
                  deduplicate: bool = True) -> CipherDataset:
    """Turn a raw text corpus into unpaired training banks.

    split_mode:
      "disjoint" - the corpus is cut in half; half A is only ever seen as
                   plaintext, half B is only ever seen as ciphertext. No sample
                   exists in both domains. This is the strict reading of
                   "unpaired" and is the one to defend in the write-up.
      "parity"   - CipherGAN's scheme: every sample is enciphered, then even
                   indices contribute their X and odd indices their Y. Cheaper
                   on data; the underlying content overlaps in distribution but
                   no individual pair is ever presented together.
    """
    vocab = vocab or Vocab()
    V = len(vocab)
    cipher = build_cipher(cipher_name, V, separate_domains=separate_domains,
                          shift=shift, key=key, seed=cipher_seed)

    cleaned = vocab.clean(text)
    samples = chunk(vocab.encode(cleaned), sample_length)

    n_raw = len(samples)
    if deduplicate:
        # A repetitive corpus can put byte-identical chunks in both halves,
        # which would silently re-introduce paired supervision. Drop them.
        samples = np.unique(samples, axis=0)

    rng = np.random.default_rng(shuffle_seed)
    samples = samples[rng.permutation(len(samples))]

    n_eval = max(1, int(round(eval_fraction * len(samples))))
    eval_plain = samples[:n_eval]
    train_pool = samples[n_eval:]
    eval_cipher = cipher.encrypt(eval_plain)

    if split_mode == "disjoint":
        mid = len(train_pool) // 2
        train_plain = train_pool[:mid]
        train_cipher = cipher.encrypt(train_pool[mid:])
    elif split_mode == "parity":
        train_plain = train_pool[0::2]
        train_cipher = cipher.encrypt(train_pool[1::2])
    else:
        raise ValueError("split_mode must be 'disjoint' or 'parity'")

    meta = {
        "cipher": cipher.describe(),
        "sample_length": sample_length,
        "split_mode": split_mode,
        "n_train_plain": int(len(train_plain)),
        "n_train_cipher": int(len(train_cipher)),
        "n_eval": int(len(eval_plain)),
        "n_samples_raw": int(n_raw),
        "n_samples_deduplicated": int(len(samples)),
    }
    return CipherDataset(vocab, cipher, train_plain, train_cipher,
                         eval_plain, eval_cipher, split_mode, meta)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def character_accuracy(pred: np.ndarray, truth: np.ndarray,
                       ignore_pad: bool = True) -> float:
    pred, truth = np.atleast_2d(pred), np.atleast_2d(truth)
    mask = (truth != 0) if ignore_pad else np.ones_like(truth, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    return float((pred[mask] == truth[mask]).mean())


def character_error_rate(pred: np.ndarray, truth: np.ndarray) -> float:
    return 1.0 - character_accuracy(pred, truth)


def mapping_accuracy(predicted_table: np.ndarray, cipher: Cipher,
                     ignore_unobserved: bool = True) -> Tuple[float, float]:
    """Fraction of the true key recovered, plus the coverage it was scored on.

    Symbols the model never saw are marked -1 by ``infer_mapping_from_outputs``
    and excluded by default: scoring them would conflate "got it wrong" with
    "the corpus contains no z".
    """
    true = cipher.inverse_table()[:, CROP_AMOUNT:]
    pred = np.atleast_2d(predicted_table)[:, :true.shape[1] + CROP_AMOUNT]
    pred = pred[:, CROP_AMOUNT:]
    mask = (pred >= 0) if ignore_unobserved else np.ones_like(pred, dtype=bool)
    coverage = float(mask.mean())
    if mask.sum() == 0:
        return 0.0, coverage
    return float((pred[mask] == true[mask]).mean()), coverage


def infer_mapping_from_outputs(cipher_in: np.ndarray, plain_out: np.ndarray,
                               vocab_size: int, period: int = 1,
                               min_support: int = 1) -> np.ndarray:
    """Read off the model's implied key by majority vote over its own outputs.

    Lets you report a recovered key table without ever giving the model labels.
    Entries with fewer than ``min_support`` observations are set to -1.
    """
    table = np.full((period, vocab_size), -1, dtype=np.int64)
    cipher_in, plain_out = np.atleast_2d(cipher_in), np.atleast_2d(plain_out)
    length = cipher_in.shape[1]
    pos = np.arange(length) % period
    for k in range(period):
        cols = pos == k
        counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
        np.add.at(counts, (cipher_in[:, cols].ravel(),
                           plain_out[:, cols].ravel()), 1)
        seen = counts.sum(axis=1) >= min_support
        table[k, seen] = counts[seen].argmax(axis=1)
    return table
