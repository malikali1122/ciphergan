"""
patch_wordlevel.py

Run from the repository root. Safe to run twice.

WHY
---
The character alphabet caps at 27 symbols, so the frontier of Section 6 cannot
be pushed further without changing what a symbol is. Word-level tokens remove
the cap: a vocabulary of 2,000 words is a substitution cipher over 2,000
symbols, with a key space of 2000! and a majority-class rate near 0.05.

This is also the setting Gomez et al. (2018) report, at 200 words and at
Brown-W, so the results become directly comparable to the published ones
rather than only to the character-level figures.

WHAT IT ADDS
------------
    --token_level {char,word}   default char, so nothing existing changes
    --vocab_words N             keep the N most frequent word types

Tokens outside the vocabulary are dropped, exactly as the character pipeline
drops punctuation and digits. The alternative, mapping them to a shared <unk>,
would give the cipher a symbol whose plaintext distribution is a mixture of
thousands of words, which is a different task.

WHAT TO WATCH
-------------
Sample count falls with word-level tokens: the Brown corpus is roughly 6.1M
characters but only about 1.1M word tokens, so 64-token samples yield ~17,000
rather than ~40,000. Section 6.3 found no seed converging below roughly 20,000
samples per bank. Word-level runs at length 64 therefore sit at or below that
threshold, and a failure could be a data limit rather than a vocabulary limit.
Use --sample_length 32 to restore the sample count, and report the length
alongside the vocabulary size so the two are not confounded.
"""

import ast

CE = "cipher_engine.py"
MD = "make_data.py"
changed = []

s = open(CE).read()

if "class WordVocab" not in s:
    anchor = "class Vocab:"
    if anchor not in s:
        raise SystemExit("\nANCHOR NOT FOUND: class Vocab\n"
                         "  grep -n 'class Vocab' cipher_engine.py\n")

    word_vocab = '''class WordVocab:
    """Word-level vocabulary, interface-compatible with Vocab.

    Symbols are word types rather than characters, which lets the alphabet
    exceed the 27 available at character level. Tokens outside the vocabulary
    are dropped rather than mapped to <unk>, mirroring how the character
    pipeline discards punctuation: a shared <unk> would be a single cipher
    symbol standing for thousands of distinct words, and the model would be
    learning something else.
    """

    def __init__(self, words):
        self.symbols = [PAD_TOKEN] + list(words)
        self.stoi = {s: i for i, s in enumerate(self.symbols)}
        self.itos = {i: s for s, i in self.stoi.items()}

    @staticmethod
    def from_corpus(text: str, n_words: int):
        """The n_words most frequent types, ties broken alphabetically."""
        from collections import Counter
        counts = Counter(WordVocab._tokenise(text))
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return WordVocab([w for w, _ in ranked[:n_words]])

    @staticmethod
    def _tokenise(text: str):
        """Lowercase alphabetic tokens. Punctuation and digits are dropped."""
        out, cur = [], []
        for ch in text.lower():
            if ch.isalpha():
                cur.append(ch)
            elif cur:
                out.append("".join(cur))
                cur = []
        if cur:
            out.append("".join(cur))
        return out

    def __len__(self):
        return len(self.symbols)

    @property
    def n_usable(self) -> int:
        return len(self.symbols) - CROP_AMOUNT

    def clean(self, text: str):
        """Returns a token list, not a string; encode() accepts either."""
        keep = set(self.symbols[CROP_AMOUNT:])
        return [t for t in self._tokenise(text) if t in keep]

    def encode(self, tokens) -> np.ndarray:
        if isinstance(tokens, str):
            tokens = self.clean(tokens)
        return np.array([self.stoi[t] for t in tokens], dtype=np.int64)

    def decode(self, indices, strip_pad: bool = True) -> str:
        out = []
        for i in indices:
            tok = self.itos.get(int(i), "?")
            if tok == PAD_TOKEN:
                if strip_pad:
                    continue
                tok = "_"
            out.append(tok)
        return " ".join(out)


'''
    s = s.replace(anchor, word_vocab + anchor, 1)
    ast.parse(s)
    open(CE, "w").write(s)
    changed.append("cipher_engine: WordVocab")

t = open(MD).read()
if "--token_level" not in t:
    a = '    p.add_argument("--alphabet", default=DEFAULT_ALPHABET)'
    if a not in t:
        raise SystemExit("\nANCHOR NOT FOUND: --alphabet\n"
                         "  grep -n 'alphabet' make_data.py\n")
    t = t.replace(a, a + '''
    p.add_argument("--token_level", default="char", choices=["char", "word"],
                   help="word-level tokens lift the 27-symbol cap")
    p.add_argument("--vocab_words", type=int, default=200,
                   help="vocabulary size for --token_level word")''', 1)

    b = "    vocab = Vocab(tuple(args.alphabet))"
    if b not in t:
        raise SystemExit("\nANCHOR NOT FOUND: vocab construction\n"
                         "  grep -n 'Vocab(' make_data.py\n")
    t = t.replace(b, '''    if args.token_level == "word":
        vocab = WordVocab.from_corpus(
            load_corpus(args.corpus, args.min_chars)[:args.max_chars or None],
            args.vocab_words)
        print(f"[cipher] word-level vocabulary, {vocab.n_usable} types, "
              f"most frequent: {' '.join(vocab.symbols[1:9])}")
    else:
        vocab = Vocab(tuple(args.alphabet))''', 1)

    if "WordVocab" not in t.split("def main")[0]:
        t = t.replace("from cipher_engine import", "from cipher_engine import WordVocab,", 1)

    ast.parse(t)
    open(MD, "w").write(t)
    changed.append("make_data: --token_level and --vocab_words")

print("applied:\n  " + "\n  ".join(changed) if changed
      else "nothing to do (already patched)")
