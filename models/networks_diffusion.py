"""Conditional absorbing-state diffusion generator for the cipher cycle.

Drop at `models/networks_diffusion.py`.

CONTRACT (verified against networks_seq.SeqGenerator and
cipher_cycle_gan_model.forward/backward_G):

  * forward(x, tau=None) accepts [B, L] long OR [B, L, V] float logits
  * returns [B, L, V] logits, length preserved
  * exposes .embedding (a SoftEmbedding) so --share_embedding still works
  * composes: G_B(G_A(x)) must run, since rec_A = netG_B(fake_B) feeds
    logits straight back in
  * differentiable end to end, because backward_G puts both a GAN loss and
    a cycle loss on the output

The generative story is different from SeqGenerator's. SeqGenerator is a
single feed-forward pass: cipher in, plaintext logits out. Here the output
is produced by a REVERSE DIFFUSION CHAIN. The target sequence starts fully
absorbed (every position at [MASK]) and is denoised over T steps, each step
conditioned on the input sequence. That is the second paradigm: the mapping
is a sampling procedure rather than a function.

Relaxation, and it must be stated in the write-up: true absorbing-state
diffusion unmasks a discrete SUBSET of positions per step, which is a
sampling operation and not differentiable. Here the mask is a per-position
CONTINUOUS weight that decays on a fixed schedule, and the partially
denoised sequence is a convex combination of the mask embedding and the
current soft token embedding. This is the same convex-hull relaxation
SoftEmbedding already uses for the discriminator, applied to the reverse
chain. At tau -> 0 with the schedule reaching 0 it recovers hard unmasking.
"""

import torch
import torch.nn as nn

# Import must work BOTH as `models.networks_diffusion` (how train.py loads
# it, via models/__init__.py) and as a flat `networks_diffusion` (how the
# contract test runs it, from inside models/). The first form is the one
# that matters in production; the fallback keeps the test runnable.
try:
    from models.networks_seq import SoftEmbedding, ResBlock1d
except ImportError:                     # running flat, from inside models/
    from networks_seq import SoftEmbedding, ResBlock1d


class TimeConditioning(nn.Module):
    """Continuous t in (0, 1], projected to the feature width.

    Matches diffusion_lm.py's convention (`self.time = nn.Linear(1, d_model)`)
    so the two models describe the diffusion time the same way and the write-up
    can refer to one schedule rather than two.
    """

    def __init__(self, dim):
        super().__init__()
        self.time = nn.Linear(1, dim)

    def forward(self, t, device):
        tt = torch.tensor([[float(t)]], device=device)
        return self.time(tt)                      # [1, dim]


class CondDiffusionGenerator(nn.Module):
    """Reverse-diffusion generator. Drop-in for SeqGenerator.

    The denoiser sees three things concatenated per position: the current
    (partially masked) target embedding, the conditioning embedding, and a
    positional embedding, plus a timestep vector added as a bias.
    """

    def __init__(self, vocab_size, embed_dim=64, ngf=128, n_blocks=4,
                 dilations=(1, 2, 4, 8), norm_layer=nn.InstanceNorm1d,
                 pointwise=False, embedding=None, pos_dim=0, max_len=512,
                 n_steps=4, mask_idx=None, loopholing=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = embedding or SoftEmbedding(vocab_size, embed_dim)
        self.owns_embedding = embedding is None
        self.n_steps = int(n_steps)
        # The absorbing state. diffusion_lm.py puts MASK at index
        # vocab_size, one past the alphabet. That is not available here:
        # the embedding table is SHARED with both discriminators under
        # --share_embedding, so widening it would change a table that is
        # also trained adversarially, and the cycle loss and metrics both
        # index by vocab_size. The mask is therefore a free vector in
        # embedding space, outside the shared table.
        self.mask_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)

        self.pos_dim = int(pos_dim)
        self.pos_embedding = (nn.Embedding(int(max_len), self.pos_dim)
                              if self.pos_dim > 0 else None)

        in_ch = embed_dim * 2 + self.pos_dim  # target ; condition ; position
        self.inp = nn.Sequential(
            nn.Conv1d(in_ch, ngf, 1 if pointwise else 5,
                      padding=0 if pointwise else 2),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        )
        self.t_embed = TimeConditioning(ngf)
        self.body = nn.Sequential(*[
            ResBlock1d(ngf, dilations[i % len(dilations)], norm_layer,
                       k=1 if pointwise else 3)
            for i in range(n_blocks)
        ])
        self.out = nn.Conv1d(ngf, vocab_size, 1)
        # Loopholing: a deterministic latent pathway between reverse
        # steps (Jo et al. 2026). Off by default so the earlier runs
        # remain reproducible from this same file.
        self.loopholing = bool(loopholing)
        self.loop_norm = nn.LayerNorm(ngf) if self.loopholing else None

    # ------------------------------------------------------------ denoiser --
    def _denoise(self, tgt_emb, cond_emb, t, h_prev=None):
        """One denoising pass. Returns (logits [B,L,V], hidden [B,ngf,L]).

        The hidden state is returned so the caller can feed it into the next
        step - the loopholing mechanism of Jo et al. (ICLR 2026). Their
        diagnosis: once a step's output is collapsed to a categorical sample
        (or, here, to a softmax-weighted embedding), the distributional
        information the denoiser computed is discarded, and the next step
        starts starved. Carrying h forward on a deterministic pathway keeps
        it. Their Algorithm 1 fuses at the input as e_t = v_t + LayerNorm(h_t);
        the equivalent point in this architecture is after the input conv.
        """
        h = torch.cat([tgt_emb, cond_emb], dim=-1)
        if self.pos_embedding is not None:
            idx = torch.arange(h.shape[1], device=h.device)
            p = self.pos_embedding(idx).unsqueeze(0).expand(h.shape[0], -1, -1)
            h = torch.cat([h, p], dim=-1)
        h = self.inp(h.transpose(1, 2))                    # [B, ngf, L]
        h = h + self.t_embed(t, h.device).t().unsqueeze(0)
        if self.loopholing and h_prev is not None:
            # LayerNorm over channels, so transpose in and back out.
            h = h + self.loop_norm(h_prev.transpose(1, 2)).transpose(1, 2)
        h = self.body(h)
        return self.out(h).transpose(1, 2), h              # [B,L,V], [B,ngf,L]

    # ------------------------------------------------------------- forward --
    def forward(self, x, tau=None):
        tau_v = self.embedding.tau if tau is None else tau
        cond = self.embedding(x, tau=tau)                  # [B, L, E]
        B, L, E = cond.shape

        # Fully absorbed start: t = 1 means every position is masked, which
        # is diffusion_lm.corrupt() at t = 1. The chain walks t down to 0.
        tgt = self.mask_embed.view(1, 1, E).expand(B, L, E)
        logits, h = None, None
        for step in range(self.n_steps):
            t_now = 1.0 - float(step) / self.n_steps
            logits, h = self._denoise(tgt, cond, t_now, h_prev=h)
            # Continuous unmasking. True absorbing diffusion would sample a
            # discrete subset to reveal; the mask weight is relaxed to a
            # scalar so the chain stays differentiable for the cycle and GAN
            # losses. With loopholing on, the information lost to this
            # relaxation is carried separately by h.
            m = 1.0 - float(step + 1) / self.n_steps
            pred_emb = self.embedding(logits, tau=tau_v)
            tgt = m * self.mask_embed.view(1, 1, E) + (1.0 - m) * pred_emb
        return logits


def build_generator(kind, **kw):
    """Factory so --netG_kind switches paradigm without touching the model."""
    if kind == "diffusion":
        return CondDiffusionGenerator(**kw)
    try:
        from models.networks_seq import SeqGenerator
    except ImportError:
        from networks_seq import SeqGenerator
    kw.pop("n_steps", None)
    kw.pop("loopholing", None)
    kw.pop("mask_idx", None)
    return SeqGenerator(**kw)
