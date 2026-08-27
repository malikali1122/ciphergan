"""
patch_families.py

Run from the repository root. Safe to run twice.

WHAT THIS ADDS
--------------
1. --max_chars on make_data.py, so a corpus can be truncated to a given size.
2. Five cipher families, chosen to separate axes that are currently confounded.

THE DESIGN, AND WHY THESE FAMILIES
----------------------------------
Every period-1 cipher in this study is an element of the symmetric group on
the alphabet. Identity, shift, atbash, affine, keyword and random substitution
differ enormously in key space - 1, 27, 1, 486, and 27! respectively - but they
are all a single permutation applied identically at every position. From the
generator's point of view they are the same function class, and composing two
of them yields a third element of the same group rather than anything new.

That predicts something falsifiable: the adversarial method should perform
IDENTICALLY on all of them, and key space should not matter. If it holds, key
space is ruled out as a difficulty axis by construction rather than by
correlation, and the cipher "tree" collapses to a single node for everything
position-independent.

Position dependence is the axis that leaves the group. A period-p cipher is p
permutations indexed by position, and a translation-equivariant generator
cannot represent it at all. The period sweep tests whether difficulty scales
with p or whether the boundary is sharp at p > 1.

  atbash     reversal, a<->z. An involution, so it is its own inverse. Key
             space 1, same as identity, but the permutation is non-trivial.
             Separates "key space is small" from "the map is easy".

  affine     x -> (ax + b) mod n, a coprime to n. Key space 486 for a
             27-symbol alphabet: far larger than a shift, far smaller than a
             full permutation, and structured rather than random. Fills the
             gap between shift and substitution.

  keyword    A permutation generated from a keyword: the keyword's distinct
             letters first, then the remaining alphabet in order. A classical
             human-usable key, and non-uniform in a way a random permutation
             is not.

  polysub    Period p, but each row is an ARBITRARY permutation rather than a
             rotation. Vigenere's rows are rotations, so a failure on Vigenere
             could in principle be about rotation structure rather than about
             position. This separates the two: if polysub and Vigenere fail
             identically, position dependence is the cause.

  composed   substitution followed by a shift. Demonstrates group closure
             directly: the resulting key table is a single permutation, and
             the model cannot distinguish it from any other substitution.

WHAT IS NOT ADDED
-----------------
Ciphers that change sequence length, or that are not position-wise bijective
(homophonic, polygraphic, transposition). The entire pipeline assumes an
input-length-preserving symbol-to-symbol map, and these would require changes
to the metrics as well as the model.
"""

import ast

CE = "cipher_engine.py"
MD = "make_data.py"
changed = []

# ------------------------------------------------------------ cipher engine ---
s = open(CE).read()

if "def make_atbash_cipher" not in s:
    anchor = """def make_identity_cipher(V: int, **kw) -> Cipher:"""
    if anchor not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: make_identity_cipher\n"
            "Run this and send the output:\n"
            "  grep -n 'def make_identity_cipher' cipher_engine.py\n")

    new = '''def _reversal_row(V: int) -> np.ndarray:
    """a <-> z, b <-> y, ... over the usable indices only."""
    row = _identity_row(V)
    usable = np.arange(CROP_AMOUNT, V)
    row[usable] = usable[::-1]
    return row


def make_atbash_cipher(V: int, **kw) -> Cipher:
    """Atbash: reverse the alphabet. An involution, so it is self-inverse.

    Key space is 1, the same as the identity cipher, but the map is not the
    identity. Any difference in difficulty between the two therefore cannot be
    attributed to key space.
    """
    return Cipher("atbash", _reversal_row(V)[None, :], V, 1,
                  key_repr="reverse", **kw)


def _affine_row(V: int, a: int, b: int) -> np.ndarray:
    """x -> (a*x + b) mod n over the usable indices. Requires gcd(a, n) = 1."""
    usable = np.arange(CROP_AMOUNT, V)
    n = len(usable)
    from math import gcd
    if gcd(a, n) != 1:
        raise ValueError(f"affine multiplier {a} is not coprime to {n}")
    row = _identity_row(V)
    row[usable] = usable[(a * np.arange(n) + b) % n]
    return row


def make_affine_cipher(V: int, a: int = 5, b: int = 8, **kw) -> Cipher:
    """Affine cipher, a structured subgroup of the permutations.

    Key space is phi(n) * n, which for a 27-symbol alphabet is 486: three
    orders of magnitude above a shift and twenty-six below a full permutation.
    It sits between the two on key space while preserving the unigram
    frequency profile exactly, as every period-1 permutation does.
    """
    return Cipher("affine", _affine_row(V, a, b)[None, :], V, 1,
                  key_repr=f"a{a}b{b}", **kw)


def _keyword_row(V: int, keyword: str, alphabet: str) -> np.ndarray:
    """Classical keyword cipher: distinct keyword letters, then the rest."""
    seen, order = set(), []
    for ch in list(keyword) + list(alphabet):
        if ch in alphabet and ch not in seen:
            seen.add(ch)
            order.append(alphabet.index(ch))
    row = _identity_row(V)
    usable = np.arange(CROP_AMOUNT, V)
    if len(order) != len(usable):
        raise ValueError(f"keyword cipher covered {len(order)} of "
                         f"{len(usable)} symbols")
    row[usable] = np.array(order, dtype=np.int64) + CROP_AMOUNT
    return row


def make_keyword_cipher(V: int, keyword: str = "cryptogam",
                        alphabet: str = None, **kw) -> Cipher:
    """A permutation a human could memorise, rather than a uniform random one.

    Included because a random permutation is a very particular kind of key:
    it has no structure for a model to exploit or be misled by. A keyword key
    leaves a long ordered tail, which is structure of exactly the sort a
    language model prior might latch onto.
    """
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET
    return Cipher("keyword", _keyword_row(V, keyword, alphabet)[None, :], V, 1,
                  key_repr=keyword, **kw)


def make_polysub_cipher(V: int, period: int = 3, seed: int = 0, **kw) -> Cipher:
    """Period-p with an arbitrary permutation per position, not a rotation.

    Vigenere's rows are rotations of one another. That makes two explanations
    for a failure on Vigenere indistinguishable: the model cannot handle
    position dependence, or it cannot handle rotation structure. Replacing the
    rotations with independent random permutations removes the second, so a
    failure here isolates position dependence as the cause.
    """
    rng = np.random.default_rng(seed)
    usable = np.arange(CROP_AMOUNT, V)
    rows = []
    for _ in range(period):
        row = _identity_row(V)
        row[usable] = rng.permutation(usable)
        rows.append(row)
    return Cipher("polysub", np.stack(rows), V, period,
                  key_repr=f"p{period}-seed{seed}", **kw)


def make_composed_cipher(V: int, seed: int = 0, shift: int = 3, **kw) -> Cipher:
    """A substitution followed by a shift.

    The composition of two period-1 permutations is a period-1 permutation.
    The resulting key table is indistinguishable in kind from any other
    substitution key, so any theory under which composition depth predicts
    difficulty has to explain why this cipher is not harder than its parts.
    """
    sub = make_substitution_cipher(V, seed=seed).key_table[0]
    rot = _rotation_row(V, shift)
    return Cipher("composed", rot[sub][None, :], V, 1,
                  key_repr=f"sub{seed}+shift{shift}", **kw)


''' + anchor
    s = s.replace(anchor, new, 1)

    # register the new factories
    old_reg = '''CIPHER_FACTORIES = {
    "identity": make_identity_cipher,
    "shift": make_shift_cipher,
    "substitution": make_substitution_cipher,
    "vigenere": make_vigenere_cipher,
}'''
    if old_reg not in s:
        raise SystemExit("\nANCHOR NOT FOUND: CIPHER_FACTORIES\n"
                         "  grep -n 'CIPHER_FACTORIES' cipher_engine.py\n")
    s = s.replace(old_reg, '''CIPHER_FACTORIES = {
    "identity": make_identity_cipher,
    "shift": make_shift_cipher,
    "substitution": make_substitution_cipher,
    "vigenere": make_vigenere_cipher,
    # period-1 families: all elements of the same permutation group
    "atbash": make_atbash_cipher,
    "affine": make_affine_cipher,
    "keyword": make_keyword_cipher,
    "composed": make_composed_cipher,
    # period-p with arbitrary rows, to isolate position dependence
    "polysub": make_polysub_cipher,
}''', 1)

    old_bc = '''    if name == "identity":
        return make_identity_cipher(vocab_size, separate_domains=separate_domains)
    raise ValueError'''
    if old_bc not in s:
        raise SystemExit("\nANCHOR NOT FOUND: build_cipher identity branch\n"
                         "  grep -n 'def build_cipher' -A 20 cipher_engine.py\n")
    s = s.replace(old_bc, '''    if name == "identity":
        return make_identity_cipher(vocab_size, separate_domains=separate_domains)
    if name == "atbash":
        return make_atbash_cipher(vocab_size, separate_domains=separate_domains)
    if name == "affine":
        return make_affine_cipher(vocab_size, a=affine_a, b=affine_b,
                                  separate_domains=separate_domains)
    if name == "keyword":
        return make_keyword_cipher(vocab_size, keyword=keyword,
                                   alphabet=alphabet,
                                   separate_domains=separate_domains)
    if name == "composed":
        return make_composed_cipher(vocab_size, seed=seed, shift=shift,
                                    separate_domains=separate_domains)
    if name == "polysub":
        return make_polysub_cipher(vocab_size, period=period, seed=seed,
                                   separate_domains=separate_domains)
    raise ValueError''', 1)

    old_sig = '''def build_cipher(name: str, vocab_size: int, separate_domains: bool = False,
                 shift: int = 3, key: Sequence[int] = (3, 4, 5),
                 seed: Optional[int] = 0) -> Cipher:'''
    if old_sig not in s:
        raise SystemExit("\nANCHOR NOT FOUND: build_cipher signature\n"
                         "  grep -n 'def build_cipher' cipher_engine.py\n")
    s = s.replace(old_sig, '''def build_cipher(name: str, vocab_size: int, separate_domains: bool = False,
                 shift: int = 3, key: Sequence[int] = (3, 4, 5),
                 seed: Optional[int] = 0, affine_a: int = 5, affine_b: int = 8,
                 keyword: str = "cryptogam", alphabet: str = None,
                 period: int = 3) -> Cipher:''', 1)

    # thread the new arguments through build_dataset
    old_bd = """                  cipher_seed: int = 0,
                  shuffle_seed: int = 1234,
                  deduplicate: bool = True) -> CipherDataset:"""
    if old_bd not in s:
        raise SystemExit("\nANCHOR NOT FOUND: build_dataset signature\n"
                         "  grep -n 'def build_dataset' -A 12 cipher_engine.py\n")
    s = s.replace(old_bd, """                  cipher_seed: int = 0,
                  shuffle_seed: int = 1234,
                  affine_a: int = 5,
                  affine_b: int = 8,
                  keyword: str = "cryptogam",
                  period: int = 3,
                  deduplicate: bool = True) -> CipherDataset:""", 1)

    old_call = """    cipher = build_cipher(cipher_name, V, separate_domains=separate_domains,"""
    if old_call not in s:
        raise SystemExit("\nANCHOR NOT FOUND: build_cipher call in build_dataset\n"
                         "  grep -n 'cipher = build_cipher' -A 3 cipher_engine.py\n")
    s = s.replace(old_call, """    cipher = build_cipher(cipher_name, V, separate_domains=separate_domains,
                          affine_a=affine_a, affine_b=affine_b,
                          keyword=keyword, period=period,
                          alphabet="".join(vocab.symbols[CROP_AMOUNT:]),""", 1)

    ast.parse(s)
    open(CE, "w").write(s)
    changed.append("cipher_engine: atbash, affine, keyword, composed, polysub")

# --------------------------------------------------------------- make_data ---
t = open(MD).read()

if "--max_chars" not in t:
    a = '    p.add_argument("--min_chars", type=int, default=200_000)'
    if a not in t:
        raise SystemExit("\nANCHOR NOT FOUND: --min_chars\n"
                         "  grep -n 'min_chars' make_data.py\n")
    t = t.replace(a, a + '''
    p.add_argument("--max_chars", type=int, default=0,
                   help="truncate the corpus to this many characters; "
                        "0 uses all of it. For the corpus-size sweep.")
    p.add_argument("--affine_a", type=int, default=5)
    p.add_argument("--affine_b", type=int, default=8)
    p.add_argument("--keyword", default="cryptogam")
    p.add_argument("--period", type=int, default=3,
                   help="period for the polysub cipher")''', 1)

    old_choices = ('''    p.add_argument("--cipher", default="substitution",
                   choices=["identity", "shift", "substitution", "vigenere"])''')
    if old_choices not in t:
        raise SystemExit("\nANCHOR NOT FOUND: --cipher choices\n"
                         "  grep -n 'choices=' make_data.py\n")
    t = t.replace(old_choices, '''    p.add_argument("--cipher", default="substitution",
                   choices=["identity", "shift", "substitution", "vigenere",
                            "atbash", "affine", "keyword", "composed",
                            "polysub"])''', 1)

    old_load = "    text = load_corpus(args.corpus, args.min_chars)"
    if old_load not in t:
        raise SystemExit("\nANCHOR NOT FOUND: load_corpus call\n"
                         "  grep -n 'load_corpus' make_data.py\n")
    t = t.replace(old_load, '''    text = load_corpus(args.corpus, args.min_chars)
    if args.max_chars:
        # Truncate from the start, so a smaller corpus is a strict prefix of a
        # larger one and the sweep varies quantity alone.
        text = text[:args.max_chars]
        print(f"[cipher] corpus truncated to {len(text)} characters")''', 1)

    old_pass = """        cipher_seed=args.cipher_seed,
        shuffle_seed=args.shuffle_seed,
    )"""
    if old_pass not in t:
        raise SystemExit("\nANCHOR NOT FOUND: build_dataset call in make_data\n"
                         "  grep -n 'build_dataset' -A 14 make_data.py\n")
    t = t.replace(old_pass, """        cipher_seed=args.cipher_seed,
        shuffle_seed=args.shuffle_seed,
        affine_a=args.affine_a,
        affine_b=args.affine_b,
        keyword=args.keyword,
        period=args.period,
    )""", 1)

    ast.parse(t)
    open(MD, "w").write(t)
    changed.append("make_data: --max_chars and the new cipher choices")

print("applied:\n  " + "\n  ".join(changed) if changed
      else "nothing to do (already patched)")
