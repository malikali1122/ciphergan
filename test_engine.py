"""Correctness tests. Run: python test_engine.py"""
import numpy as np

from cipher_engine import (Vocab, build_cipher, build_dataset, CROP_AMOUNT,
                           character_accuracy, mapping_accuracy,
                           infer_mapping_from_outputs)
import cipher_stats as cs
from make_data import SAMPLE_TEXT

TEXT = SAMPLE_TEXT * 600
V = len(Vocab())
ok = lambda name: print(f"  PASS  {name}")


def test_roundtrip():
    v = Vocab()
    x = v.encode(v.clean(SAMPLE_TEXT))[None, :]
    for name in ["identity", "shift", "substitution", "vigenere"]:
        for sep in [False, True]:
            c = build_cipher(name, V, separate_domains=sep)
            y = c.encrypt(x)
            assert np.array_equal(c.decrypt(y), x), (name, sep)
            if sep:
                nz = y[y != 0]
                assert nz.min() >= V - CROP_AMOUNT + 1, name
                assert y.max() < c.total_vocab_size
            ok(f"roundtrip {name} separate_domains={sep}")


def test_bijectivity():
    for name in ["shift", "substitution", "vigenere"]:
        c = build_cipher(name, V)
        for k in range(c.period):
            row = c.key_table[k, CROP_AMOUNT:]
            assert len(set(row.tolist())) == len(row), name
            assert c.key_table[k, 0] == 0
        ok(f"bijective key table {name}")


def test_pad_preserved():
    c = build_cipher("substitution", V)
    x = np.array([[0, 1, 2, 0, 3]])
    y = c.encrypt(x)
    assert y[0, 0] == 0 and y[0, 3] == 0
    ok("pad index 0 is never enciphered")


def test_unpaired_disjoint():
    ds = build_dataset(TEXT, "substitution", sample_length=50,
                       split_mode="disjoint")
    plain_rows = {r.tobytes() for r in ds.train_plain}
    recovered = {r.tobytes() for r in ds.cipher.decrypt(ds.train_cipher)}
    assert not (plain_rows & recovered), "a training pair leaked across domains"
    assert not (plain_rows & {r.tobytes() for r in ds.eval_plain})
    ok("disjoint split: no plaintext sample appears in both banks")


def test_monoalphabetic_invariance():
    ds = build_dataset(TEXT, "substitution", sample_length=100)
    r = cs.profile_report(ds.train_plain, ds.train_cipher, V)
    assert r["unigram_profile_sym_kl_bits"] < 1e-3, r
    assert abs(r["plain_ioc"] - r["cipher_ioc"]) < 5e-3, r
    mi = cs.mutual_information(ds.eval_plain, ds.eval_cipher, V)
    h = cs.entropy(cs.unigram_profile(ds.eval_plain, V))
    assert abs(mi - h) < 1e-6, (mi, h)
    ok(f"substitution leaks fully: I(X;Y)={mi:.3f} == H(X)={h:.3f} bits")


def test_vigenere_degrades():
    base = build_dataset(TEXT, "substitution", sample_length=100)
    ioc_sub = cs.index_of_coincidence(base.train_cipher, V)
    prev = ioc_sub
    for klen in [3, 7, 15]:
        ds = build_dataset(TEXT, "vigenere", sample_length=100,
                           key=list(range(1, klen + 1)))
        ioc = cs.index_of_coincidence(ds.train_cipher, V)
        assert ioc < prev + 1e-6, (klen, ioc, prev)
        prev = ioc
        print(f"        vigenere key_len={klen:2d} IoC={ioc:.4f}")
    ok(f"IoC falls monotonically with key length (substitution={ioc_sub:.4f})")


def test_key_recovery_metric():
    ds = build_dataset(TEXT, "substitution", sample_length=100)
    # simulate a perfect model: decrypt with the true key
    perfect = ds.cipher.decrypt(ds.eval_cipher)
    assert character_accuracy(perfect, ds.eval_plain) == 1.0
    table = infer_mapping_from_outputs(ds.eval_cipher, perfect,
                                       ds.cipher.total_vocab_size, period=1)
    acc, cov = mapping_accuracy(table, ds.cipher)
    assert acc == 1.0, (acc, cov)
    print(f"        key recovery acc={acc:.3f} on coverage={cov:.3f}")
    # simulate a useless model
    rng = np.random.default_rng(0)
    junk = rng.integers(1, V, size=ds.eval_plain.shape)
    assert character_accuracy(junk, ds.eval_plain) < 0.15
    ok("CER and key-recovery metrics behave at both extremes")


def test_parity_mode():
    ds = build_dataset(TEXT, "substitution", sample_length=50,
                       split_mode="parity")
    assert abs(len(ds.train_plain) - len(ds.train_cipher)) <= 1
    ok("parity split matches CipherGAN's scheme")


if __name__ == "__main__":
    print("cipher engine tests")
    for fn in [test_roundtrip, test_bijectivity, test_pad_preserved,
               test_unpaired_disjoint, test_monoalphabetic_invariance,
               test_vigenere_degrades, test_key_recovery_metric,
               test_parity_mode]:
        fn()
    print("\nall tests passed")
