"""
patch_positional.py

Run from the repository root, after the other patches. Safe to run twice.
Off by default: without --pos_dim every existing run is bit-identical.

THE PROBLEM
-----------
Cipher.encrypt selects the key row by ABSOLUTE POSITION in the sample:

    pos = np.arange(length) % self.period
    cipher = self.key_table[pos, plain]

Every sample starts at phase 0, so for a period-3 Vigenere the correct
decryption maps one input token to three different output tokens depending on
where it sits.

SeqGenerator is embedding -> Conv1d -> residual blocks -> Conv1d, with no
positional signal anywhere. Under --pointwise_G every kernel is width 1, so the
network computes a single function of a single token and applies it identically
at all positions. No parameter setting can express a position-dependent map.
The best available approximation is to agree with one key row and be wrong on
the rest, which is what the observed 0.14-0.25 accuracies are: roughly 1/period
of positions correct.

So the Vigenere results measure a representational limit of the architecture,
not a limit of adversarial cipher cracking. This patch removes the limit so the
question can actually be asked.

THE FIX
-------
A learned embedding of absolute position, concatenated to the token embedding
before the first convolution.

Absolute position, not position % period, and not position % some assumed
maximum. Indexing by position % max_period is wrong unless max_period is an
exact multiple of the true period: with max_period 8 and period 3, position 8
lands in slot 0 while 8 % 3 = 2, so the slot no longer determines the phase.
Absolute position always determines it, and the network is free to learn
whatever modular structure the data has.

WHAT IT DOES NOT DO
-------------------
The discriminators are left alone. Under a Vigenere the ciphertext domain does
have position-dependent statistics that a translation-equivariant
discriminator cannot fully model, so there may be a second, weaker version of
this problem on the D side. It is a smaller effect - D only has to detect a
distributional mismatch, not inverta position-dependent map - and changing one
thing at a time is worth more here than changing two.

USAGE
-----
    --pos_dim 32          size of the positional embedding; 0 disables it
    --max_len 512         table size; must exceed the sample length

Checkpoints are not compatible across this flag: it changes the input width of
the first convolution. Give patched runs new --name values.
"""

import ast

NS = "models/networks_seq.py"
CM = "models/cipher_cycle_gan_model.py"

changed = []

# ------------------------------------------------------------ the generator ---
s = open(NS).read()
if "pos_dim" not in s:
    old_sig = """    def __init__(self, vocab_size, embed_dim=64, ngf=128, n_blocks=4,
                 dilations=(1, 2, 4, 8), norm_layer=nn.InstanceNorm1d,
                 pointwise=False, embedding=None):"""
    if old_sig not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: SeqGenerator.__init__\n"
            "Run this and send the output:\n"
            "  grep -n 'class SeqGenerator' -A 12 models/networks_seq.py\n")
    s = s.replace(old_sig, """    def __init__(self, vocab_size, embed_dim=64, ngf=128, n_blocks=4,
                 dilations=(1, 2, 4, 8), norm_layer=nn.InstanceNorm1d,
                 pointwise=False, embedding=None, pos_dim=0, max_len=512):""", 1)

    old_inp = """        self.inp = nn.Sequential(
            nn.Conv1d(embed_dim, ngf, 1 if pointwise else 5,
                      padding=0 if pointwise else 2),"""
    if old_inp not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: SeqGenerator input conv\n"
            "Run this and send the output:\n"
            "  grep -n 'self.inp = ' -A 6 models/networks_seq.py\n")
    s = s.replace(old_inp, """        # Absolute position, so the network can recover position % period for
        # any period. Concatenated rather than added: the token embedding is
        # shared with the discriminators under --share_embedding, and adding
        # would push positional information into a table that is also being
        # trained adversarially.
        self.pos_dim = int(pos_dim)
        self.pos_embedding = (nn.Embedding(int(max_len), self.pos_dim)
                              if self.pos_dim > 0 else None)
        self.inp = nn.Sequential(
            nn.Conv1d(embed_dim + self.pos_dim, ngf, 1 if pointwise else 5,
                      padding=0 if pointwise else 2),""", 1)

    old_fwd = """        h = self.embedding(x, tau=tau)          # [B, L, E]
        h = h.transpose(1, 2)                   # [B, E, L]"""
    if old_fwd not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: SeqGenerator.forward\n"
            "Run this and send the output:\n"
            "  grep -n 'def forward' -A 6 models/networks_seq.py\n")
    s = s.replace(old_fwd, """        h = self.embedding(x, tau=tau)          # [B, L, E]
        if self.pos_embedding is not None:
            idx = torch.arange(h.shape[1], device=h.device)
            p = self.pos_embedding(idx)         # [L, P]
            h = torch.cat([h, p.unsqueeze(0).expand(h.shape[0], -1, -1)],
                          dim=-1)               # [B, L, E+P]
        h = h.transpose(1, 2)                   # [B, E+P, L]""", 1)

    if "import torch" not in s.split("class ")[0]:
        raise SystemExit("networks_seq.py does not import torch at module level")

    ast.parse(s)
    open(NS, "w").write(s)
    changed.append("SeqGenerator takes an absolute-position embedding")

# ----------------------------------------------------------------- the model ---
t = open(CM).read()
if "--pos_dim" not in t:
    anchor = 'parser.add_argument("--n_blocks_G", type=int, default=4)'
    if anchor not in t:
        raise SystemExit(
            "\nANCHOR NOT FOUND: --n_blocks_G\n"
            "Run this and send the output:\n"
            "  grep -n 'n_blocks_G' models/cipher_cycle_gan_model.py\n")
    t = t.replace(anchor, anchor + '''
        parser.add_argument("--pos_dim", type=int, default=0,
                            help="width of the generator's positional "
                                 "embedding; 0 disables it. Required for any "
                                 "cipher whose key depends on position.")
        parser.add_argument("--max_len", type=int, default=512,
                            help="positional table size; must exceed the "
                                 "sample length")''', 1)

    old = """            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False)),"""
    if t.count(old) != 2:
        raise SystemExit(
            f"\nANCHOR NOT FOUND: SeqGenerator construction ({t.count(old)} of 2)\n"
            "Run this and send the output:\n"
            "  grep -n 'SeqGenerator(' -A 3 models/cipher_cycle_gan_model.py\n")
    t = t.replace(old, """            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False),
                         pos_dim=getattr(opt, "pos_dim", 0),
                         max_len=getattr(opt, "max_len", 512)),""")

    t = t.replace('''        norm_layer = get_norm_layer_1d(opt.norm)
''', '''        norm_layer = get_norm_layer_1d(opt.norm)

        if getattr(opt, "pos_dim", 0) > 0:
            print(f"[cipher] positional encoding on, width {opt.pos_dim}")
        elif int(getattr(opt, "cipher_period", 1) or 1) > 1:
            print("[cipher] WARNING: periodic cipher with --pos_dim 0. The "
                  "generator has no positional signal and cannot represent a "
                  "position-dependent key.")
''', 1)
    ast.parse(t)
    open(CM, "w").write(t)
    changed.append("--pos_dim and --max_len, passed to both generators")

# ------------------------------------------------- publish the cipher period ---
DS = "data/cipher_dataset.py"
u = open(DS).read()
if "cipher_period" not in u:
    a = """        if getattr(opt, "vocab_size", -1) in (None, -1):
            opt.vocab_size = self.vocab_size"""
    if a not in u:
        raise SystemExit(
            "\nANCHOR NOT FOUND: opt.vocab_size publish\n"
            "Run this and send the output:\n"
            "  grep -n 'opt.vocab_size' -B 3 -A 3 data/cipher_dataset.py\n")
    u = u.replace(a, """        # The model warns when a periodic cipher is trained without
        # positional encoding, which it can only do if it knows the period.
        opt.cipher_period = int(c.get("period", 1))

""" + a, 1)
    ast.parse(u)
    open(DS, "w").write(u)
    changed.append("dataset publishes opt.cipher_period")

print("applied:\n  " + "\n  ".join(changed) if changed
      else "nothing to do (already patched)")
