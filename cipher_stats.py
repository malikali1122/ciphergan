"""
cipher_stats.py
===============

Phase 1 of the project plan: *prove* the cipher leaks before training anything.

The claim your report needs to support is that a monoalphabetic substitution
cipher is a relabelling of the alphabet, not a change to the *shape* of the
distribution. These functions produce the numbers and the figure for that.

Metrics
-------
unigram_profile      sorted frequency vector; identical under any monoalphabetic
                     cipher, distorted by a polyalphabetic one
index_of_coincidence classic cryptanalytic statistic. English ~0.066, uniform
                     random over 26 letters ~0.038. Invariant under
                     substitution; drops toward random as Vigenere key grows
mutual_information   I(plain; cipher) in bits. Equals the plaintext entropy for
                     a deterministic monoalphabetic cipher (total leakage);
                     falls as the mapping becomes position-dependent
bigram_divergence    symmetric KL between sorted bigram distributions - shows
                     that higher-order structure survives too
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).ravel()
    return x[x != 0]  # drop <pad>


def unigram_counts(seq: np.ndarray, vocab_size: int) -> np.ndarray:
    return np.bincount(_flat(seq), minlength=vocab_size).astype(np.float64)


def unigram_profile(seq: np.ndarray, vocab_size: int) -> np.ndarray:
    """Frequencies sorted descending - the cipher-invariant 'fingerprint'."""
    c = unigram_counts(seq, vocab_size)
    p = c / max(c.sum(), 1)
    return np.sort(p)[::-1]


def bigram_counts(seq: np.ndarray, vocab_size: int) -> np.ndarray:
    seq = np.atleast_2d(seq)
    a, b = seq[:, :-1].ravel(), seq[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    m = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(m, (a[keep], b[keep]), 1.0)
    return m


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def index_of_coincidence(seq: np.ndarray, vocab_size: int) -> float:
    c = unigram_counts(seq, vocab_size)
    n = c.sum()
    if n < 2:
        return 0.0
    return float((c * (c - 1)).sum() / (n * (n - 1)))


def mutual_information(plain: np.ndarray, cipher: np.ndarray,
                       vocab_size: int) -> float:
    """I(X;Y) in bits over aligned plaintext/ciphertext symbols.

    Requires an aligned pair, so run this on the held-out EVAL split only.
    """
    a, b = np.asarray(plain), np.asarray(cipher)
    if a.shape != b.shape:
        raise ValueError(
            f"mutual_information needs ALIGNED pairs, got {a.shape} vs "
            f"{b.shape}. Call it on the eval split, not the unpaired banks.")
    a, b = a.ravel(), b.ravel()
    keep = (a != 0) & (b != 0)
    a, b = a[keep], b[keep]
    hi = max(vocab_size, int(b.max()) + 1)
    joint = np.zeros((vocab_size, hi), dtype=np.float64)
    np.add.at(joint, (a, b), 1.0)
    joint /= max(joint.sum(), 1)
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p / max(p.sum(), eps) + eps
    q = q / max(q.sum(), eps) + eps
    return float((p * np.log2(p / q)).sum() + (q * np.log2(q / p)).sum())


def profile_report(plain: np.ndarray, cipher: np.ndarray,
                   vocab_size: int) -> Dict[str, float]:
    """One row of the vulnerability table in your report."""
    up, uc = unigram_profile(plain, vocab_size), unigram_profile(cipher, vocab_size)
    n = min(len(up), len(uc))
    bp = np.sort(bigram_counts(plain, vocab_size).ravel())[::-1]
    bc_full = bigram_counts(cipher, max(vocab_size, int(np.max(cipher)) + 1))
    bc = np.sort(bc_full.ravel())[::-1]
    m = min(len(bp), len(bc))
    return {
        "unigram_profile_max_abs_diff": float(np.abs(up[:n] - uc[:n]).max()),
        "unigram_profile_sym_kl_bits": symmetric_kl(up[:n], uc[:n]),
        "bigram_profile_sym_kl_bits": symmetric_kl(bp[:m], bc[:m]),
        "plain_entropy_bits": entropy(up),
        "cipher_entropy_bits": entropy(uc),
        "plain_ioc": index_of_coincidence(plain, vocab_size),
        "cipher_ioc": index_of_coincidence(cipher, vocab_size),
        # MI needs aligned pairs; this runs on the unpaired banks.
        # Callers compute it on the held-out eval split instead.
    }


def plot_invariance(plain: np.ndarray, cipher: np.ndarray, vocab_size: int,
                    symbols: Optional[Sequence[str]] = None,
                    title: str = "", path: str = "frequency_invariance.png"):
    """Two-panel figure: raw frequencies differ, sorted profiles coincide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cp = unigram_counts(plain, vocab_size)
    cp = cp / max(cp.sum(), 1)
    hi = max(vocab_size, int(np.max(cipher)) + 1)
    cc = unigram_counts(cipher, hi)
    cc = cc / max(cc.sum(), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    idx = np.arange(1, vocab_size)
    ax1.bar(idx - 0.2, cp[1:vocab_size], width=0.4, label="plaintext")
    ax1.bar(idx + 0.2, cc[1:vocab_size], width=0.4, label="ciphertext")
    ax1.set_title("Per-symbol frequency (relabelled)")
    ax1.set_xlabel("symbol index")
    ax1.set_ylabel("relative frequency")
    if symbols is not None:
        ax1.set_xticks(idx)
        ax1.set_xticklabels([symbols[i] for i in idx], fontsize=7)
    ax1.legend()

    up, uc = unigram_profile(plain, vocab_size), unigram_profile(cipher, hi)
    n = min(len(up), len(uc))
    ax2.plot(up[:n], "o-", label="plaintext profile")
    ax2.plot(uc[:n], "s--", label="ciphertext profile")
    ax2.set_title("Rank-ordered frequency profile (invariant)")
    ax2.set_xlabel("rank")
    ax2.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
