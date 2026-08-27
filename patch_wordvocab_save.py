"""
patch_wordvocab_save.py

Run from the repository root. Safe to run twice.

WordVocab was written without the save() method that Vocab has, so make_data.py
crashes on its last line after the dataset has already been written. Any
word-level dataset built before this patch has a valid dataset.npz and no
vocab.txt; re-running make_data.py once this is applied produces both.
"""

import ast

CE = "cipher_engine.py"
s = open(CE).read()

if "def save(self, path: str, separate_domains: bool = False) -> None:" not in s:
    raise SystemExit("\nANCHOR NOT FOUND: Vocab.save\n"
                     "  grep -n 'def save' cipher_engine.py\n")

if s.count("def save(self, path: str, separate_domains: bool = False) -> None:") > 1:
    raise SystemExit("nothing to do (already patched)")

anchor = '''    def decode(self, indices, strip_pad: bool = True) -> str:
        out = []
        for i in indices:
            tok = self.itos.get(int(i), "?")
            if tok == PAD_TOKEN:
                if strip_pad:
                    continue
                tok = "_"
            out.append(tok)
        return " ".join(out)'''

if anchor not in s:
    raise SystemExit("\nANCHOR NOT FOUND: WordVocab.decode\n"
                     "  grep -n 'class WordVocab' -A 60 cipher_engine.py\n")

s = s.replace(anchor, anchor + '''

    def save(self, path: str, separate_domains: bool = False) -> None:
        """Same one-token-per-line format as Vocab.save.

        Word types can contain no whitespace by construction, since the
        tokeniser splits on anything non-alphabetic, so one line per token is
        unambiguous.
        """
        tokens = list(self.symbols)
        if separate_domains:
            tokens += [f"{t}#c" for t in self.symbols[CROP_AMOUNT:]]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\\n".join(tokens) + "\\n")''', 1)

ast.parse(s)
open(CE, "w").write(s)
print("applied: WordVocab.save")
