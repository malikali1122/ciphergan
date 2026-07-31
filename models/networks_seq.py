"""Sequence networks for the cipher task.

Drop this file at `models/networks_seq.py` in your fork.

The repo's `networks.py` builds 2D convolutional generators and PatchGAN
discriminators for images. These are the 1D equivalents over token sequences,
plus the soft-embedding bridge that makes the adversarial loop differentiable
on discrete data.

They deliberately mirror the repo's conventions so the rest of the codebase is
untouched:
  * generators map one tensor to another and compose, so `G_B(G_A(x))` works
  * discriminators return a map of per-position scores, exactly like PatchGAN,
    so `networks.GANLoss` (vanilla | lsgan | wgangp) works unmodified
  * `networks.init_net` still applies
"""

import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    """Embedding that accepts hard indices OR a distribution over the vocabulary.

    argmax over generator logits is not differentiable, so the generator emits
    logits and we take the expected embedding under softmax(logits / tau):

        e = softmax(logits / tau) @ W

    Real tokens go through the SAME matrix W, so real embeddings are lattice
    points and generated ones sit inside their convex hull. As tau falls the
    two become indistinguishable. This is CipherGAN's contribution and the
    reason a plain CycleGAN fails on discrete data: without it the
    discriminator wins trivially by noticing that generated vectors are off the
    embedding manifold, which is "uninformative discrimination" - the
    discriminator is right for a reason that teaches the generator nothing.
    """

    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.tau = 1.0  # set by the model; annealed during training
        self.hard = False
        self.weight = nn.Embedding(vocab_size, embed_dim,
                                   padding_idx=padding_idx)

    def forward(self, x, tau=None, hard=None):
        """x: [B, L] long (hard tokens) or [B, L, V] float (logits) -> [B, L, E]"""
        tau = self.tau if tau is None else tau
        hard = self.hard if hard is None else hard
        if x.dtype in (torch.long, torch.int64, torch.int32):
            return self.weight(x)
        if hard:
            probs = nn.functional.gumbel_softmax(x, tau=tau, hard=True, dim=-1)
        else:
            probs = torch.softmax(x / tau, dim=-1)
        if getattr(self, "straight_through", False):
            # Exact one-hot forward, soft backward: both real and generated
            # inputs land on exact embedding lattice points, leaving the
            # discriminator no sharpness cue. Unlike Gumbel, no sampling noise.
            hard = torch.zeros_like(probs).scatter_(
                -1, probs.argmax(-1, keepdim=True), 1.0)
            probs = hard + probs - probs.detach()
        return probs @ self.weight.weight


class ResBlock1d(nn.Module):
    """Residual dilated conv block. Dilation widens the receptive field fast,
    which matters because n-gram evidence for a substitution key is spread over
    tens of characters."""

    def __init__(self, ch, dilation, norm_layer, k=3):
        super().__init__()
        # k=1 keeps DEPTH while removing CONTEXT, as CipherGAN does.
        pad = 0 if k == 1 else dilation
        dilation = 1 if k == 1 else dilation
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            norm_layer(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            norm_layer(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class SeqGenerator(nn.Module):
    """Tokens (or logits) -> logits over the vocabulary. Length preserved.

    Length preservation is a real assumption worth stating in the write-up: a
    substitution cipher is position-wise bijective, so the output sequence has
    exactly the length of the input. That is a strictly easier setting than the
    seq2seq generation used in text style transfer, and it removes decoding,
    beam search and length modelling from the problem entirely.
    """

    def __init__(self, vocab_size, embed_dim=64, ngf=128, n_blocks=4,
                 dilations=(1, 2, 4, 8), norm_layer=nn.InstanceNorm1d,
                 pointwise=False):
        super().__init__()
        self.embedding = SoftEmbedding(vocab_size, embed_dim)
        self.inp = nn.Sequential(
            nn.Conv1d(embed_dim, ngf, 1 if pointwise else 5,
                      padding=0 if pointwise else 2),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[
            ResBlock1d(ngf, dilations[i % len(dilations)], norm_layer,
                       k=1 if pointwise else 3)
            for i in range(n_blocks)
        ])
        self.out = nn.Conv1d(ngf, vocab_size, 1)

    def forward(self, x, tau=None):
        h = self.embedding(x, tau=tau)          # [B, L, E]
        h = h.transpose(1, 2)                   # [B, E, L]
        h = self.out(self.body(self.inp(h)))    # [B, V, L]
        return h.transpose(1, 2)                # [B, L, V] logits


class SeqDiscriminator(nn.Module):
    """PatchGAN over a sequence: per-position realness scores.

    Local scores rather than one global score is the right choice here. The
    evidence that a decryption is wrong is local - an impossible bigram, a 'q'
    with no 'u' - so scoring every window gives dense gradient and makes the
    discriminator a literal learned n-gram frequency analyser. That is exactly
    the "vulnerability scanner" framing for the second-marker write-up.

    RECEPTIVE FIELD = THE ORDER OF n-GRAM STATISTICS IT CAN SEE
    -----------------------------------------------------------
    Each output score depends on a fixed window of the input. Measured by
    gradient support (not derived on paper - see the note in the project
    README for the one-liner that reproduces these):

        --n_layers_D 1   ->  10 characters
        --n_layers_D 2   ->  22 characters
        --n_layers_D 3   ->  46 characters   (default)
        --n_layers_D 4   ->  94 characters

    This turns a hyperparameter into a research question. Your Phase 1 analysis
    showed a monoalphabetic cipher preserves unigram and bigram statistics
    exactly; a discriminator restricted to 10 characters can still exploit
    those, so it should crack substitution. A Vigenere cipher with key length k
    only reveals itself to a discriminator whose window spans multiple key
    periods, so accuracy should fall off sharply once k approaches the
    receptive field. That is a falsifiable prediction you can state in the
    methods chapter and test in the results chapter, which is worth
    considerably more than reporting whichever setting happened to work.
    """

    def __init__(self, vocab_size, embed_dim=64, ndf=128, n_layers=3, kw=4,
                 norm_layer=nn.InstanceNorm1d):
        super().__init__()
        self.embedding = SoftEmbedding(vocab_size, embed_dim)
        layers = [nn.Conv1d(embed_dim, ndf, kw, stride=2, padding=kw // 2),
                  nn.LeakyReLU(0.2, inplace=True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [
                nn.Conv1d(ndf * prev, ndf * mult, kw, stride=2, padding=kw // 2),
                norm_layer(ndf * mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [nn.Conv1d(ndf * mult, 1, kw, stride=1, padding=kw // 2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x, tau=None):
        h = self.embedding(x, tau=tau).transpose(1, 2)  # [B, E, L]
        return self.model(h)                            # [B, 1, L']


class _ChannelLayerNorm(nn.Module):
    """Normalise over channels at each position. GroupNorm(1,C) would pool
    across the sequence and hand a 'pointwise' generator hidden context."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                 # [B, C, L]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def get_norm_layer_1d(norm_type="instance"):
    if norm_type == "instance":
        return lambda c: nn.InstanceNorm1d(c, affine=False, track_running_stats=False)
    if norm_type == "batch":
        return lambda c: nn.BatchNorm1d(c, affine=True, track_running_stats=True)
    if norm_type == "layer":
        return lambda c: _ChannelLayerNorm(c)
    if norm_type == "none":
        return lambda c: nn.Identity()
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
