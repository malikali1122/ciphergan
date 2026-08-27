"""
patch_shared_embedding.py

Allows one embedding table to be shared across both generators and both
discriminators, which is what --share_embedding needs in order to do anything.

Run from the repository root. Safe to run twice.

WHY
---
CipherGAN (Gomez et al. 2018) uses a single embedding matrix W_Emb, referenced
by every network and trained adversarially. Its non-stationarity is the
mechanism their Proposition 1 relies on to prevent uninformative
discrimination: because the table keeps moving, the discriminator cannot lock
onto the fixed lattice of exact embedding vectors.

As written, SeqGenerator and SeqDiscriminator each construct their own table,
so four independent embeddings exist and nothing can be shared. This patch adds
an optional `embedding=` argument; when one is supplied the network uses it
instead of building its own.
"""

import ast

P = "models/networks_seq.py"
s = open(P).read()
changed = []

if "embedding=None" not in s:
    # ---- generator ---------------------------------------------------------
    gen_old = """                 pointwise=False):
        super().__init__()
        self.embedding = SoftEmbedding(vocab_size, embed_dim)"""
    gen_new = """                 pointwise=False, embedding=None):
        super().__init__()
        # CipherGAN uses ONE table W_Emb shared by both generators and both
        # discriminators, trained adversarially. Accepting it as an argument
        # rather than constructing it here is what makes that sharing possible.
        self.embedding = embedding or SoftEmbedding(vocab_size, embed_dim)
        self.owns_embedding = embedding is None"""
    assert gen_old in s, (
        "SeqGenerator.__init__ does not match the expected form. "
        "Paste the output of:  sed -n '100,112p' models/networks_seq.py")
    s = s.replace(gen_old, gen_new, 1)
    changed.append("SeqGenerator accepts a shared embedding")

    # ---- discriminator -----------------------------------------------------
    dis_old = """                 norm_layer=nn.InstanceNorm1d):
        super().__init__()
        self.embedding = SoftEmbedding(vocab_size, embed_dim)"""
    dis_new = """                 norm_layer=nn.InstanceNorm1d, embedding=None):
        super().__init__()
        self.embedding = embedding or SoftEmbedding(vocab_size, embed_dim)
        self.owns_embedding = embedding is None"""
    assert dis_old in s, (
        "SeqDiscriminator.__init__ does not match the expected form. "
        "Paste the output of:  grep -n 'class SeqDiscriminator' -A 8 "
        "models/networks_seq.py")
    s = s.replace(dis_old, dis_new, 1)
    changed.append("SeqDiscriminator accepts a shared embedding")

    ast.parse(s)
    open(P, "w").write(s)

print("applied: " + "; ".join(changed) if changed
      else "nothing to do (already patched)")
