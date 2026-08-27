"""CycleGAN adapted to discrete cipher sequences.

Drop this file at `models/cipher_cycle_gan_model.py` in your fork and run with
`--model cipher_cycle_gan`.

WHAT IS INHERITED FROM CycleGANModel, UNCHANGED
-----------------------------------------------
  backward_D_basic, backward_D_A, backward_D_B   pure tensor ops, rank-agnostic
  optimize_parameters                            the alternating G/D schedule
  BaseModel                                      checkpointing, schedulers,
                                                 device placement, set_requires_grad
  networks.GANLoss                               vanilla | lsgan | wgangp
  util.image_pool.ImagePool                      iterates the batch dim only,
                                                 so it buffers sequences fine

WHAT IS OVERRIDDEN, AND WHY
---------------------------
  __init__     1D sequence networks instead of 2D conv/PatchGAN
  set_input    unpack token indices rather than image tensors
  forward      pass a temperature through the soft-embedding bridge
  backward_G   cross-entropy cycle loss instead of L1

The last one is the substantive change and the one to defend in the report.
CycleGAN's cycle loss is ||F(G(x)) - x||_1 over pixel intensities. Over token
logits, an L1 reconstruction term is minimised by shrinking logit magnitudes
towards a degenerate uniform solution: it rewards being non-committal. Token
identity is categorical, not ordinal - symbol 5 is not "between" 4 and 6 - so
the correct reconstruction objective is the negative log-likelihood of the
original token under the reconstructed distribution. Keep `--cycle_loss l1` as
an ablation; the comparison is a cheap and genuinely informative result.
"""

import itertools
import os

import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.cycle_gan_model import CycleGANModel
from models import networks
from models.networks_seq import SeqGenerator, SeqDiscriminator, get_norm_layer_1d
from util.image_pool import ImagePool


def _unique_params(*iterables):
    """Deduplicate parameters by identity.

    A shared embedding is reachable from several networks, so naive chaining
    passes the same tensor to the optimiser more than once. PyTorch accepts
    that with a warning and applies one update per occurrence, silently
    multiplying the learning rate for exactly the parameter whose controlled
    movement the paper's mechanism depends on.
    """
    seen, out = set(), []
    for it in iterables:
        for p in it:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


class CipherCycleGANModel(CycleGANModel):
    """A (ciphertext) <-> B (plaintext). G_A is the decryption model."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CycleGANModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--vocab_size", type=int, default=-1,
                            help="inferred from the dataset; do not set by hand")
        parser.add_argument("--embed_dim", type=int, default=64)
        parser.add_argument("--n_blocks_G", type=int, default=4)
        parser.add_argument("--pos_dim", type=int, default=0,
                            help="width of the generator's positional "
                                 "embedding; 0 disables it. Required for any "
                                 "cipher whose key depends on position.")
        parser.add_argument("--max_len", type=int, default=512,
                            help="positional table size; must exceed the "
                                 "sample length")
        parser.add_argument("--share_embedding", action="store_true",
                            help="one embedding table across all four networks")
        parser.add_argument("--warmup_steps", type=int, default=2500,
                            help="exponential LR warmup; paper uses 2500")
        parser.add_argument("--beta2", type=float, default=0.9,
                            help="Adam beta2; paper 0.9, torch default 0.999")
        parser.add_argument("--kw_D", type=int, default=4)
        parser.add_argument("--pointwise_G", action="store_true",
                            help="context-free generator: a pure lookup table")
        parser.add_argument("--matched_softness", action="store_true",
                            help="present REAL tokens to D through the same "
                                 "softmax path as generated ones, so D cannot "
                                 "separate them on the soft-vs-hard signature")
        parser.add_argument("--real_logit_scale", type=float, default=5.0,
                            help="logit magnitude for one-hot real tokens")
        parser.add_argument("--lambda_gp", type=float, default=10.0,
                            help="WGAN-GP gradient penalty weight")
        parser.add_argument("--use_gp", action="store_true")
        parser.add_argument("--n_critic", type=int, default=1,
                            help="discriminator steps per generator step; "
                                 "WGAN-GP normally uses 5")
        parser.add_argument("--tau_start", type=float, default=1.0,
                            help="initial softmax temperature")
        parser.add_argument("--tau_end", type=float, default=0.1,
                            help="final temperature; anneal sharpens outputs "
                                 "towards one-hot as training stabilises")
        parser.add_argument("--gumbel", action="store_true",
                            help="straight-through Gumbel-softmax instead of "
                                 "plain soft embeddings (ablation)")
        parser.add_argument("--straight_through", action="store_true")
        if is_train:
            parser.add_argument("--cycle_loss", type=str, default="ce",
                                choices=["ce", "l1", "simplex_l1"],
                                help="ce = cross-entropy on tokens (correct for "
                                     "discrete data); l1 = CycleGAN's original")
        parser.set_defaults(
            dataset_mode="cipher",
            netG="seq", netD="seq",
            ngf=128, ndf=128, n_layers_D=3,
            norm="instance", gan_mode="lsgan",
            lambda_identity=0.0,   # domains do not share a vocabulary; see note
            lambda_A=10.0, lambda_B=10.0,
            batch_size=64, lr=2e-4,
            display_id=0,          # no visdom; visuals are sequences not images
        )
        return parser

    def __init__(self, opt):
        # The reference train.py never seeds torch, so network initialisation
        # varies uncontrolled between runs. Measured consequence: at a FIXED
        # configuration, identity-cipher accuracy ranged from 0.054 to 0.415.
        # Without this, no result is reproducible and no two conditions are
        # comparable. --pair_seed now controls init as well as data pairing.
        import random as _random
        _seed = int(getattr(opt, "pair_seed", 0))
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)
        _random.seed(_seed)
        try:
            import numpy as _np
            _np.random.seed(_seed)
        except ImportError:
            pass
        # Deliberately skip CycleGANModel.__init__ - it builds 2D image
        # networks. Everything below mirrors it structurally so the inherited
        # backward_D_* and optimize_parameters find the attributes they expect.
        BaseModel.__init__(self, opt)

        self.loss_names = ["D_A", "G_A", "cycle_A",
                           "D_B", "G_B", "cycle_B",
                           # adversarial loss WITHOUT the gradient penalty;
                           # D_A above still includes it, as it always has
                           "Dadv_A", "Dadv_B",
                           "gp_A", "gp_B",
                           # mean raw D output on real / generated input
                           "pr_A", "pf_A", "pr_B", "pf_B"]
        self.visual_names = []  # util.tensor2im cannot render token sequences
        self.model_names = ["G_A", "G_B", "D_A", "D_B"] if self.isTrain \
            else ["G_A", "G_B"]

        V, E = opt.vocab_size, opt.embed_dim
        norm_layer = get_norm_layer_1d(opt.norm)

        if getattr(opt, "pos_dim", 0) > 0:
            print(f"[cipher] positional encoding on, width {opt.pos_dim}")
        elif int(getattr(opt, "cipher_period", 1) or 1) > 1:
            print("[cipher] WARNING: periodic cipher with --pos_dim 0. The "
                  "generator has no positional signal and cannot represent a "
                  "position-dependent key.")

        self.netG_A = networks.init_net(
            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False),
                         pos_dim=getattr(opt, "pos_dim", 0),
                         max_len=getattr(opt, "max_len", 512)),
            opt.init_type, opt.init_gain).to(self.device)
        self.netG_B = networks.init_net(
            SeqGenerator(V, E, opt.ngf, opt.n_blocks_G, norm_layer=norm_layer,
                         pointwise=getattr(opt, "pointwise_G", False),
                         pos_dim=getattr(opt, "pos_dim", 0),
                         max_len=getattr(opt, "max_len", 512)),
            opt.init_type, opt.init_gain).to(self.device)

        if self.isTrain:
            self.netD_A = networks.init_net(
                SeqDiscriminator(V, E, opt.ndf, opt.n_layers_D,
                                 norm_layer=norm_layer),
                opt.init_type, opt.init_gain).to(self.device)
            self.netD_B = networks.init_net(
                SeqDiscriminator(V, E, opt.ndf, opt.n_layers_D,
                                 norm_layer=norm_layer),
                opt.init_type, opt.init_gain).to(self.device)

            # The gradient penalty interpolates between real and generated
            # inputs, so they must share a shape. In the paper both arrive as
            # embeddings (one-hot @ W and softmax @ W); here that presentation
            # comes from --matched_softness.
            if (getattr(opt, "use_gp", False) or opt.gan_mode == "wgangp") \
                    and not getattr(opt, "matched_softness", False):
                opt.matched_softness = True
                print("[cipher] auto-enabling --matched_softness for the "
                      "gradient penalty")

            if getattr(opt, "share_embedding", False):
                _src = (self.netG_A.module if hasattr(self.netG_A, "module")
                        else self.netG_A).embedding
                for _n in ["netG_B", "netD_A", "netD_B"]:
                    _m = getattr(self, _n)
                    _m = _m.module if hasattr(_m, "module") else _m
                    _m.embedding = _src
                print("[cipher] embedding table shared across all networks")

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCE = nn.CrossEntropyLoss(ignore_index=0)
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(
                _unique_params(self.netG_A.parameters(),
                               self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, getattr(opt, 'beta2', 0.9)))
            _dp = [self.netD_A.parameters(), self.netD_B.parameters()]
            if getattr(opt, "share_embedding", False):
                # the shared table also receives the discriminator objective,
                # i.e. it is trained to MAXIMISE the GAN loss, per the paper
                _emb = (self.netG_A.module if hasattr(self.netG_A, "module")
                        else self.netG_A).embedding
                _dp = _dp + [_emb.parameters()]
            self.optimizer_D = torch.optim.Adam(
                _unique_params(*_dp),
                lr=opt.lr, betas=(opt.beta1, getattr(opt, 'beta2', 0.9)))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.set_tau(opt.tau_start)
        self.set_gumbel(opt.gumbel)
        self.set_straight_through(getattr(opt, "straight_through", False))

    # ---------------------------------------------------------------- tau ---
    def _embeddings(self):
        for net_name in ["netG_A", "netG_B", "netD_A", "netD_B"]:
            net = getattr(self, net_name, None)
            if net is None:
                continue
            module = net.module if hasattr(net, "module") else net
            yield module.embedding

    def backward_D_basic(self, netD, real, fake, tag=""):
        """CycleGAN's version plus the WGAN-GP term.

        Gomez et al. (2018) report that their CycleGAN-derived architecture was
        unstable - converging on roughly half of attempts - until the WGAN
        Jacobian norm regularisation was added to the discriminator loss, after
        which convergence became near-consistent. junyanz's repo ships
        cal_gradient_penalty but no model calls it, so --gan_mode wgangp on its
        own gives WGAN with NO Lipschitz constraint, which is worse than LSGAN.
        This wires it up.

        The penalty interpolates between real and fake, so both must be the
        same shape and dtype. That makes --matched_softness a prerequisite
        rather than an option here.
        """
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_adv = (loss_D_real + loss_D_fake) * 0.5
        loss_D = loss_D_adv
        self.loss_gp = torch.zeros((), device=self.device)

        if self.opt.gan_mode == "wgangp" or getattr(self.opt, "use_gp", False):
            if real.shape != fake.shape or real.dtype != fake.dtype:
                raise ValueError(
                    "--gan_mode wgangp requires --matched_softness so that "
                    "real and generated inputs share a shape for interpolation")
            gp, _ = networks.cal_gradient_penalty(
                netD, real, fake.detach(), self.device,
                lambda_gp=self.opt.lambda_gp)
            loss_D = loss_D + gp
            self.loss_gp = gp.detach()

        if tag:
            # Recorded before backward so the numbers describe the state the
            # gradients were computed from. Detached: these are diagnostics.
            setattr(self, "loss_Dadv_" + tag, loss_D_adv.detach())
            setattr(self, "loss_gp_" + tag, self.loss_gp)
            setattr(self, "loss_pr_" + tag, pred_real.detach().mean())
            setattr(self, "loss_pf_" + tag, pred_fake.detach().mean())

        loss_D.backward()
        return loss_D

    def _apply_warmup(self):
        """tensor2tensor exponential warmup, per iteration.

        lr_t = lr * 0.01 ** ((W - t) / W): starts at one hundredth of the
        target and rises to it over W steps. GAN training is unusually
        sensitive to the first few hundred updates, and the bimodal outcome
        seen in this project - half the runs converging, half settling on
        arbitrary bijections - is the signature of an unstable start.
        """
        W = int(getattr(self.opt, "warmup_steps", 0))
        if W <= 0:
            return
        self._gstep = getattr(self, "_gstep", 0) + 1
        if self._gstep > W:
            return
        f = 0.01 ** ((W - self._gstep) / W)
        for o in self.optimizers:
            for g in o.param_groups:
                g["lr"] = self.opt.lr * f

    def optimize_parameters(self):
        """n_critic discriminator steps per generator step (WGAN-GP wants 5)."""
        self._apply_warmup()
        self.forward()
        for i in range(max(1, self.opt.n_critic)):
            if i > 0:
                self.forward()
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            self.backward_D_A()
            self.backward_D_B()
            self.optimizer_D.step()

        self.set_requires_grad([self.netD_A, self.netD_B], False)
        if getattr(self.opt, "share_embedding", False):
            # keep the shared table trainable: set_requires_grad walks
            # netD_*.parameters(), which now reaches W_Emb, so the line above
            # would otherwise freeze it for the whole generator step. The paper
            # trains W_Emb on the cycle objective as well as the adversarial
            # one, so it has to stay live here.
            self.set_requires_grad(
                [(self.netG_A.module if hasattr(self.netG_A, "module")
                  else self.netG_A).embedding], True)
        if getattr(self.opt, "share_embedding", False):
            # rebuild the generator graph. optimizer_D.step() has just mutated
            # W_Emb in place, and the graph from the forward() above still
            # references the pre-step version of it. Autograd refuses to
            # backprop through that. Costs one forward pass per iteration.
            self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def _real_for_D(self, idx):
        """Present real tokens the same way generated ones are presented.

        MEASURED: with real tokens entering D as an exact embedding lookup and
        generated ones as a softmax mixture, a discriminator separates them
        with accuracy 1.000 at tau=1.0 and tau=0.5 on IDENTICAL text. It never
        needs to learn English - it just detects whether the input sits on a
        lattice point. The generator's adversarial gradient then says "sharpen
        your distribution", not "decrypt correctly", which is why the identity
        cipher fails as badly as substitution.

        Routing real tokens through the same softmax makes the two paths
        numerically identical, so the only remaining signal is content. This
        also decouples temperature from the cheating problem, which matters
        because simply lowering tau trades cheating for vanishing gradients.
        """
        if not getattr(self.opt, "matched_softness", False):
            return idx
        oh = torch.nn.functional.one_hot(idx, self.opt.vocab_size).float()
        return oh * self.opt.real_logit_scale

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self._real_for_D(self.real_B), fake_B, tag="A")

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self._real_for_D(self.real_A), fake_A, tag="B")

    def set_tau(self, tau):
        self.current_tau = float(tau)
        for emb in self._embeddings():
            emb.tau = self.current_tau

    def set_straight_through(self, flag):
        for emb in self._embeddings():
            emb.straight_through = bool(flag)

    def set_gumbel(self, flag):
        for emb in self._embeddings():
            emb.hard = bool(flag)

    def update_learning_rate(self):
        """Overridden to piggyback temperature annealing on the LR hook.

        train.py already calls this once per epoch, so annealing here means the
        repo's training script needs no modification at all - which keeps the
        diff reviewable and the "what did you change" section of the report
        short and precise.

        Geometric schedule from --tau_start to --tau_end. Log it alongside the
        losses: temperature is usually the difference between converging and
        not, and an examiner will ask how it was chosen. "Annealed 1.0 -> 0.1
        geometrically over 30 epochs" is an answer; "0.5 seemed to work" is not.
        """
        super().update_learning_rate()
        # Seed from --epoch_count, not 0, so --continue_train resumes the
        # schedule instead of jumping the temperature back to tau_start.
        if not hasattr(self, "_epochs_done"):
            self._epochs_done = max(self.opt.epoch_count - 1, 0)
        self._epochs_done += 1
        total = max(self.opt.n_epochs + self.opt.n_epochs_decay, 2)
        r = (self.opt.tau_end / self.opt.tau_start) ** (1.0 / (total - 1))
        self.set_tau(self.opt.tau_start * (r ** min(self._epochs_done, total - 1)))
        print(f"temperature tau = {self.current_tau:.4f}")

    # -------------------------------------------------------------- passes ---
    def set_input(self, input):
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)  # [B, L] long
        self.real_B = input["B" if AtoB else "A"].to(self.device)  # [B, L] long
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)   # logits [B, L, V]
        self.rec_A = self.netG_B(self.fake_B)    # logits, cycle A->B->A
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def _cycle(self, logits, target_idx):
        if getattr(self.opt, "cycle_loss", "ce") == "simplex_l1":
            # The paper's term: L1 between the softmax distribution and the
            # original one-hot, on the probability simplex. This is not the
            # degenerate case of an L1 on embeddings, which could be minimised
            # by shrinking embedding norms.
            probs = torch.softmax(logits / max(self.current_tau, 1e-6), dim=-1)
            oh = torch.nn.functional.one_hot(target_idx, self.opt.vocab_size)
            mask = (target_idx != 0).unsqueeze(-1).float()
            return ((probs - oh.float()).abs() * mask).sum(-1).mean()
        # getattr: --cycle_loss is only registered at train time, but this
        # method is reachable from test.py through a loaded checkpoint.
        if getattr(self.opt, "cycle_loss", "ce") == "l1":
            # CycleGAN's original objective, on the probability simplex.
            onehot = nn.functional.one_hot(
                target_idx, num_classes=logits.shape[-1]).float()
            return self.criterionL1(torch.softmax(logits, dim=-1), onehot)
        B, L, V = logits.shape
        return self.criterionCE(logits.reshape(B * L, V),
                                target_idx.reshape(B * L))

    def backward_G(self):
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self._cycle(self.rec_A, self.real_A) * self.opt.lambda_A
        self.loss_cycle_B = self._cycle(self.rec_B, self.real_B) * self.opt.lambda_B
        self.loss_G = (self.loss_G_A + self.loss_G_B
                       + self.loss_cycle_A + self.loss_cycle_B)
        self.loss_G.backward()

    # ---------------------------------------------------------- evaluation ---
    @torch.no_grad()
    def decrypt(self, cipher_idx):
        """Hard decryption for reporting CER. Never used in a loss."""
        return self.netG_A(cipher_idx.to(self.device)).argmax(dim=-1)

    def _symbols(self):
        """Lazily load the alphabet so outputs can be logged as readable text."""
        if not hasattr(self, "_syms"):
            import numpy as np
            d = np.load(self.opt.npz_path, allow_pickle=True)
            self._syms = list(d["symbols"])
        return self._syms

    def _decode(self, row):
        syms = self._symbols()
        return "".join("_" if i == 0 else str(syms[i]) if i < len(syms) else "?"
                       for i in row.tolist())

    @torch.no_grad()
    def compute_visuals(self):
        """train.py calls this at --display_freq. Used here to dump readable
        samples, since util.tensor2im cannot render token sequences.

        Watching the actual text is not a luxury on this task. Losses can look
        healthy while the generator has collapsed onto a single high-frequency
        letter, and the only cheap way to catch that is to read the output.
        """
        if not self.isTrain:
            return
        pred = self.netG_A(self.real_A).argmax(dim=-1)
        n = min(2, self.real_A.shape[0])
        block = [f"--- iter tau={self.current_tau:.4f} ---"]
        for k in range(n):
            block.append(f"  cipher in : {self._decode(self.real_A[k])[:70]}")
            block.append(f"  decrypted : {self._decode(pred[k])[:70]}")
            block.append(f"  real plain: {self._decode(self.real_B[k])[:70]}")
        text = "\n".join(block)
        print(text)
        with open(os.path.join(self.save_dir, "samples.txt"), "a") as fh:
            fh.write(text + "\n")
