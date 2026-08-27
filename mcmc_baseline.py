"""
mcmc_baseline.py
================

Classical Metropolis-Hastings key recovery, as the control the adversarial
method is measured against.

    python mcmc_baseline.py --npz_path data/substitution/dataset.npz --seeds 10

WHY THIS EXISTS
---------------
The frontier and ladder curves only mean something if the tasks on them are
solvable. A method that predates deep learning by decades, runs on a CPU in
seconds, and sees exactly the same inputs establishes that. Where the classical
attack succeeds and the GAN does not, the curve is a property of the GAN.

WHAT IT SEES
------------
Exactly what the GAN sees, and no more:

  * ``train_cipher``  the unpaired ciphertext bank, used to search for the key.
  * ``train_plain``   the unpaired plaintext bank, used ONLY to fit the bigram
                      language model. Never aligned with any ciphertext.

It never touches ``eval_plain`` until scoring, and under
``--split_mode disjoint`` the eval samples come from a slice of the corpus that
appears in neither training bank. ``--audit`` prints the check.

METHOD
------
State is a decryption table ``dec[k, c] -> plaintext symbol``, one permutation
per position class ``k`` (period 1 for identity/shift/substitution, 3 or 7 for
the Vigenere conditions). Score is the log-likelihood of the decrypted
ciphertext under the plaintext bigram model. Proposals swap two symbols within
one position class; acceptance is Metropolis with a geometric annealing
schedule.

The score is computed from bigram COUNTS rather than by decrypting the corpus
each step:

    score = sum_k  sum_{a,b}  C_k[a,b] * logP[ dec[k][a], dec[k+1][b] ]

``C_k[a,b]`` counts ciphertext symbol ``a`` at a position congruent to k
followed by ``b``. A proposal touching class k changes only the terms k and
k-1, so each step costs two V x V gathers regardless of corpus size. This is
what makes ten seeds on the full Brown corpus a matter of seconds.

REPORTING
---------
Metrics come from the same ``cipher_engine`` functions ``eval_cipher.py`` uses,
computed on the same eval split, so the numbers drop straight into the results
table beside the GAN's.

Note that ``chance_accuracy`` in the JSON is the uniform figure 1/(V-1), which
is what ``eval_cipher.py`` also reports. It is not the right reference point: a
model that emits nothing but spaces already scores about 0.175 on the 27-symbol
condition. ``majority_class_accuracy`` is computed here as well; read results
against that one.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cipher_engine import (Cipher, CROP_AMOUNT, character_accuracy,
                           infer_mapping_from_outputs, mapping_accuracy)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    c = meta["cipher"]
    cipher = Cipher(name=c["name"],
                    key_table=np.array(c["key_table"], dtype=np.int64),
                    vocab_size=int(c["vocab_size"]),
                    period=int(c["period"]),
                    key_repr=c.get("key", ""),
                    separate_domains=bool(c["separate_domains"]))
    return {
        "train_plain": d["train_plain"],
        "train_cipher": d["train_cipher"],
        "eval_plain": d["eval_plain"],
        "eval_cipher": d["eval_cipher"],
        "symbols": list(d["symbols"]),
        "cipher": cipher,
        "meta": meta,
    }


def audit(ds) -> dict:
    """Confirm the language model cannot have seen the evaluation plaintext."""
    def rows(a):
        return set(map(bytes, np.ascontiguousarray(a, dtype=np.int64)))

    tp, ep, tc = rows(ds["train_plain"]), rows(ds["eval_plain"]), None
    overlap_plain = len(tp & ep)

    # The ciphertext bank decrypted under the true key: does any of it coincide
    # with the plaintext bank? Under --split_mode parity it will, by design.
    true_plain_of_bank = ds["cipher"].decrypt(ds["train_cipher"])
    tc = rows(true_plain_of_bank)
    overlap_banks = len(tp & tc)

    return {
        "eval_plain_in_train_plain": overlap_plain,
        "train_cipher_content_in_train_plain": overlap_banks,
        "split_mode": ds["meta"].get("split_mode"),
        "clean": overlap_plain == 0,
    }


# --------------------------------------------------------------------------- #
# Language model, fitted on the unpaired plaintext bank only
# --------------------------------------------------------------------------- #
def bigram_logprobs(plain: np.ndarray, V: int, alpha: float = 0.5) -> np.ndarray:
    counts = np.zeros((V, V), dtype=np.float64)
    a, b = plain[:, :-1].ravel(), plain[:, 1:].ravel()
    keep = (a != 0) & (b != 0)
    np.add.at(counts, (a[keep], b[keep]), 1.0)
    counts += alpha
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def class_bigram_counts(cipher_text: np.ndarray, V: int, period: int,
                        offset: int = 0) -> list:
    """C[k][a, b]: symbol a at position = k (mod period), followed by b."""
    x = np.asarray(cipher_text, dtype=np.int64)
    if offset:
        x = np.where(x == 0, 0, x - offset)
    length = x.shape[1]
    out = []
    for k in range(period):
        cols = np.arange(length - 1)[np.arange(length - 1) % period == k]
        a, b = x[:, cols].ravel(), x[:, cols + 1].ravel()
        keep = (a != 0) & (b != 0)
        C = np.zeros((V, V), dtype=np.float64)
        np.add.at(C, (a[keep], b[keep]), 1.0)
        out.append(C)
    return out


# --------------------------------------------------------------------------- #
# Metropolis-Hastings
# --------------------------------------------------------------------------- #
def _term(C, L, dec_from, dec_to) -> float:
    return float(np.sum(C * L[dec_from[:, None], dec_to[None, :]]))


def _freq_init(C_list, L, V, rng, perturb: int = 0) -> np.ndarray:
    """Rank-match ciphertext unigram frequency to plaintext unigram frequency.

    A classical opening move, and a fair one: the frequency profile is
    invariant under a substitution cipher, which is precisely the leakage the
    vulnerability analysis measures.

    ``perturb`` applies that many random swaps per position class afterwards.
    Without it every restart in a seed would begin from the same point and the
    extra chains would buy nothing.
    """
    period = len(C_list)
    usable = np.arange(CROP_AMOUNT, V)
    plain_freq = np.exp(L).sum(axis=1)
    plain_order = np.argsort(-plain_freq[CROP_AMOUNT:]) + CROP_AMOUNT
    dec = np.zeros((period, V), dtype=np.int64)
    for k in range(period):
        cfreq = C_list[k].sum(axis=1) + C_list[k - 1].sum(axis=0)
        cfreq = cfreq[CROP_AMOUNT:] + rng.random(V - CROP_AMOUNT) * 1e-9
        cipher_order = np.argsort(-cfreq) + CROP_AMOUNT
        dec[k, cipher_order] = plain_order
        for _ in range(perturb):
            i, j = usable[rng.integers(len(usable), size=2)]
            dec[k, i], dec[k, j] = dec[k, j], dec[k, i]
    return dec


def mh_solve(C_list, L, V, rng, steps=30000, t0=0.02, t1=2e-4,
             init="freq", perturb=0):
    period = len(C_list)
    usable = np.arange(CROP_AMOUNT, V)

    if init == "freq":
        dec = _freq_init(C_list, L, V, rng, perturb=perturb)
    else:
        dec = np.zeros((period, V), dtype=np.int64)
        for k in range(period):
            dec[k, usable] = rng.permutation(usable)

    n_bigrams = max(1.0, sum(C.sum() for C in C_list))
    terms = [_term(C_list[k], L, dec[k], dec[(k + 1) % period])
             for k in range(period)]
    score = sum(terms)
    best_dec, best_score = dec.copy(), score

    ratio = (t1 / t0) ** (1.0 / max(1, steps - 1))
    T = t0
    for _ in range(steps):
        k = int(rng.integers(period))
        i, j = usable[rng.integers(len(usable), size=2)]
        if i == j:
            T *= ratio
            continue

        dec[k, i], dec[k, j] = dec[k, j], dec[k, i]
        affected = {k, (k - 1) % period}
        new_terms = {m: _term(C_list[m], L, dec[m], dec[(m + 1) % period])
                     for m in affected}
        new_score = score + sum(new_terms[m] - terms[m] for m in affected)

        # Deltas are normalised per bigram so the schedule does not have to be
        # retuned when the corpus size changes.
        delta = (new_score - score) / n_bigrams
        if delta >= 0 or rng.random() < np.exp(delta / T):
            score = new_score
            terms = [new_terms.get(m, terms[m]) for m in range(period)]
            if score > best_score:
                best_score, best_dec = score, dec.copy()
        else:
            dec[k, i], dec[k, j] = dec[k, j], dec[k, i]
        T *= ratio

    return best_dec, best_score / n_bigrams


def apply_key(dec: np.ndarray, cipher_text: np.ndarray,
              offset: int = 0) -> np.ndarray:
    x = np.asarray(cipher_text, dtype=np.int64)
    pad = x == 0
    if offset:
        x = np.where(pad, 0, x - offset)
    period = dec.shape[0]
    pos = np.arange(x.shape[1]) % period
    out = dec[pos[None, :].repeat(x.shape[0], axis=0), np.clip(x, 0, dec.shape[1] - 1)]
    out[pad] = 0
    return out.astype(np.int64)


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", required=True)
    p.add_argument("--seeds", type=int, default=10,
                   help="independent attempts; ten is the project minimum")
    p.add_argument("--restarts", type=int, default=3,
                   help="chains per seed; best-scoring chain is kept")
    p.add_argument("--steps", type=int, default=0,
                   help="0 = auto: 20000 x period. A period-7 Vigenere is "
                        "seven coupled permutations and needs the budget; "
                        "20000 is ample for a monoalphabetic cipher.")
    p.add_argument("--t0", type=float, default=0.02, help="initial temperature")
    p.add_argument("--t1", type=float, default=2e-4, help="final temperature")
    p.add_argument("--alpha", type=float, default=0.5, help="add-alpha smoothing")
    p.add_argument("--init", default="freq", choices=["freq", "random"])
    p.add_argument("--solve_samples", type=int, default=0,
                   help="cap ciphertext rows used for the search (0 = all)")
    p.add_argument("--audit", action="store_true",
                   help="print the train/eval separation check")
    p.add_argument("--out", default=None, help="path for the metrics JSON")
    args = p.parse_args()

    ds = load(args.npz_path)
    cipher = ds["cipher"]
    V, period = cipher.vocab_size, cipher.period
    offset = cipher.domain_offset

    if args.audit:
        a = audit(ds)
        print("\nleakage audit")
        for k, v in a.items():
            print(f"  {k:38s} {v}")
        if not a["clean"]:
            print("  WARNING: eval plaintext appears in the LM training bank")

    bank = ds["train_cipher"]
    if args.solve_samples:
        bank = bank[:args.solve_samples]

    L = bigram_logprobs(ds["train_plain"], V, alpha=args.alpha)
    C_list = class_bigram_counts(bank, V, period, offset=offset)

    # Budget scales with BOTH the period and the alphabet, since the state is
    # p permutations of V symbols and the number of distinct transpositions
    # grows as V^2. A fixed budget silently handicaps the baseline at large
    # vocabularies: on a 200-word cipher, 60,000 steps recovers 0.37 of the key
    # and 400,000 recovers 0.89.
    steps = args.steps if args.steps > 0 else max(20000, 800 * V) * period

    truth = ds["eval_plain"]
    maj = float(np.bincount(truth[truth != 0].ravel(),
                            minlength=V).max() / (truth != 0).sum())

    rows, t_start = [], time.time()
    for s in range(args.seeds):
        rng = np.random.default_rng(s)
        best = None
        for r in range(max(1, args.restarts)):
            # chain 0 takes the clean frequency match; later chains are
            # perturbed away from it so the extra budget buys exploration
            dec, sc = mh_solve(C_list, L, V, rng, steps=steps,
                               t0=args.t0, t1=args.t1, init=args.init,
                               perturb=0 if r == 0 else 4 * r)
            if best is None or sc > best[1]:
                best = (dec, sc)
        dec, sc = best

        pred = apply_key(dec, ds["eval_cipher"], offset=offset)
        acc = character_accuracy(pred, truth)
        # same call eval_cipher.py makes, so the two tables are comparable
        table = infer_mapping_from_outputs(ds["eval_cipher"], pred, V,
                                           period=period)
        key_acc, coverage = mapping_accuracy(table, cipher)
        rows.append({"seed": s, "character_accuracy": float(acc),
                     "key_recovery_accuracy": float(key_acc),
                     "key_recovery_coverage": float(coverage),
                     "logprob_per_bigram": float(sc)})
        print(f"  seed {s:2d}  key {key_acc:.4f}  char {acc:.4f}  "
              f"logP/bigram {sc:+.4f}")

    elapsed = time.time() - t_start
    keys = np.array([r["key_recovery_accuracy"] for r in rows])
    chars = np.array([r["character_accuracy"] for r in rows])

    print("\n" + "=" * 62)
    print(f"cipher              {cipher.name} (key={cipher.key_repr}) "
          f"period={period}")
    print(f"alphabet            {V - CROP_AMOUNT} usable symbols")
    print(f"eval samples        {truth.shape[0]} x {truth.shape[1]}")
    print(f"key recovery        best {keys.max():.4f}  mean {keys.mean():.4f}"
          f"  sd {keys.std(ddof=1) if len(keys) > 1 else 0:.4f}")
    print(f"character accuracy  best {chars.max():.4f}  mean {chars.mean():.4f}")
    print(f"majority-class      {maj:.4f}   (uniform chance {1/(V-1):.4f})")
    print(f"solved (key = 1.0)  {(keys >= 0.999).sum()}/{len(keys)}")
    print(f"wall clock          {elapsed:.1f} s on CPU for {len(keys)} seeds")
    print("=" * 62)

    out = args.out or os.path.join(os.path.dirname(args.npz_path) or ".",
                                   "mcmc_metrics.json")
    with open(out, "w") as fh:
        json.dump({
            "method": "metropolis-hastings, bigram LM on unpaired plaintext",
            "cipher": cipher.name,
            "key": cipher.key_repr,
            "period": period,
            "vocab_size": V,
            "steps": steps,
            "restarts": args.restarts,
            "init": args.init,
            "seeds": rows,
            "key_recovery_best": float(keys.max()),
            "key_recovery_mean": float(keys.mean()),
            "character_accuracy_best": float(chars.max()),
            "character_accuracy_mean": float(chars.mean()),
            "n_solved": int((keys >= 0.999).sum()),
            "majority_class_accuracy": maj,
            "chance_accuracy": float(1 / (V - 1)),
            "seconds": elapsed,
        }, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
