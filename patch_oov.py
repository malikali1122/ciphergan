"""
patch_oov.py

Run from the repository root, after patch_wordlevel.py. Safe to run twice.

WHY
---
patch_wordlevel.py dropped out-of-vocabulary tokens, on the reasoning that the
character pipeline drops punctuation. That reasoning was wrong, and the
consequence is severe.

Dropping OOV tokens destroys word order. Surviving tokens were not adjacent in
the source text, so the bigram statistics the whole task depends on become
noise. The effect is visible directly:

    50 types, dropped:    the said an of s that the said in that
    2000 types, dropped:  the fulton county grand jury said friday an
                          investigation of

At small vocabularies the sequence is no longer English. That is why the
classical baseline, which solves 27-symbol substitution in thirteen seconds,
collapses to 0.16 key recovery on a 200-word vocabulary: there is no bigram
signal left to search for.

Gomez et al. (2018) do not do this. Their Brown-W200 keeps the 200 most
frequent word types and replaces every other token with a single <unk>, which
preserves position and therefore preserves the sequence statistics. They report
98.7% on shift at that vocabulary.

WHAT THIS ADDS
--------------
    --oov {drop,unk}    default drop, so existing datasets are unaffected

Under --oov unk, <unk> occupies index 1 and is a cipher symbol like any other:
it is enciphered and must be recovered. Two consequences to report rather than
hide.

  * <unk> is very frequent at small vocabularies, so the majority-class rate
    rises sharply and a high character accuracy is worth less than it looks.
    The evaluation already reports the majority-class rate; quote it beside
    every word-level number.
  * Because <unk> is by far the most frequent symbol, the frequency-matching
    that any attack begins with will map it correctly almost immediately. Key
    recovery accuracy, which weights all symbols equally, is the more honest
    metric here.
"""

import ast

CE = "cipher_engine.py"
MD = "make_data.py"
changed = []

s = open(CE).read()

if "UNK_TOKEN" not in s:
    if "PAD_TOKEN = " not in s:
        raise SystemExit("\nANCHOR NOT FOUND: PAD_TOKEN\n"
                         "  grep -n 'PAD_TOKEN' cipher_engine.py\n")
    s = s.replace("PAD_TOKEN = ", 'UNK_TOKEN = "<unk>"\nPAD_TOKEN = ', 1)

    old_init = '''    def __init__(self, words):
        self.symbols = [PAD_TOKEN] + list(words)'''
    if old_init not in s:
        raise SystemExit("\nANCHOR NOT FOUND: WordVocab.__init__\n"
                         "  grep -n 'class WordVocab' -A 8 cipher_engine.py\n")
    s = s.replace(old_init, '''    def __init__(self, words, oov="drop"):
        self.oov = oov
        extra = [UNK_TOKEN] if oov == "unk" else []
        self.symbols = [PAD_TOKEN] + extra + list(words)''', 1)

    old_from = '''    @staticmethod
    def from_corpus(text: str, n_words: int):
        """The n_words most frequent types, ties broken alphabetically."""
        from collections import Counter
        counts = Counter(WordVocab._tokenise(text))
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return WordVocab([w for w, _ in ranked[:n_words]])'''
    if old_from not in s:
        raise SystemExit("\nANCHOR NOT FOUND: WordVocab.from_corpus\n"
                         "  grep -n 'def from_corpus' -A 8 cipher_engine.py\n")
    s = s.replace(old_from, '''    @staticmethod
    def from_corpus(text: str, n_words: int, oov: str = "drop"):
        """The n_words most frequent types, ties broken alphabetically.

        Under oov="unk" the vocabulary is n_words types plus <unk>, so the
        cipher alphabet is n_words + 1.
        """
        from collections import Counter
        counts = Counter(WordVocab._tokenise(text))
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return WordVocab([w for w, _ in ranked[:n_words]], oov=oov)''', 1)

    old_clean = '''    def clean(self, text: str):
        """Returns a token list, not a string; encode() accepts either."""
        keep = set(self.symbols[CROP_AMOUNT:])
        return [t for t in self._tokenise(text) if t in keep]'''
    if old_clean not in s:
        raise SystemExit("\nANCHOR NOT FOUND: WordVocab.clean\n"
                         "  grep -n 'def clean' -A 5 cipher_engine.py\n")
    s = s.replace(old_clean, '''    def clean(self, text: str):
        """Returns a token list, not a string; encode() accepts either.

        Under oov="unk" every token is kept, so position is preserved and the
        bigram statistics of the source text survive. Under oov="drop" the
        surviving tokens are no longer adjacent in the original, which is why
        that mode should not be used for anything reported.
        """
        keep = set(self.symbols[CROP_AMOUNT:])
        toks = self._tokenise(text)
        if self.oov == "unk":
            return [t if t in keep else UNK_TOKEN for t in toks]
        return [t for t in toks if t in keep]''', 1)

    ast.parse(s)
    open(CE, "w").write(s)
    changed.append("cipher_engine: UNK_TOKEN and WordVocab(oov=...)")

t = open(MD).read()
if "--oov" not in t:
    a = ('    p.add_argument("--vocab_words", type=int, default=200,\n'
         '                   help="vocabulary size for --token_level word")')
    if a not in t:
        raise SystemExit("\nANCHOR NOT FOUND: --vocab_words\n"
                         "  grep -n 'vocab_words' make_data.py\n")
    t = t.replace(a, a + '''
    p.add_argument("--oov", default="drop", choices=["drop", "unk"],
                   help="drop out-of-vocabulary tokens, or map them to a "
                        "single <unk>. Use unk: dropping destroys word order "
                        "and with it the bigram statistics the task needs.")''', 1)

    b = '''        vocab = WordVocab.from_corpus(
            load_corpus(args.corpus, args.min_chars)[:args.max_chars or None],
            args.vocab_words)'''
    if b not in t:
        raise SystemExit("\nANCHOR NOT FOUND: WordVocab.from_corpus call\n"
                         "  grep -n 'WordVocab.from_corpus' -A 3 make_data.py\n")
    t = t.replace(b, '''        vocab = WordVocab.from_corpus(
            load_corpus(args.corpus, args.min_chars)[:args.max_chars or None],
            args.vocab_words, oov=args.oov)
        if args.oov == "drop":
            print("[cipher] WARNING: --oov drop removes tokens and therefore "
                  "breaks word order. Bigram statistics will not reflect the "
                  "source text. Use --oov unk for anything reported.")''', 1)

    ast.parse(t)
    open(MD, "w").write(t)
    changed.append("make_data: --oov with a warning on drop")

print("applied:\n  " + "\n  ".join(changed) if changed
      else "nothing to do (already patched)")
