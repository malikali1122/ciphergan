"""Plain adversarial baseline: one generator, one discriminator, no cycle.

Selected with `--model cipher_gan`.

WHY THIS FILE EXISTS
--------------------
Two reasons, both about interpreting the main results.

1. It is a working GAN on the toy task, independent of whether
   cycle-consistency ever converges.

2. It is the control that gives the cycle-consistency result meaning. Without
   it, "our CycleGAN reached 94% character accuracy" is an isolated number. With
   it, the comparison shows *what the cycle term buys*, and the expected failure is
   specific and predictable rather than vague: a generator trained only to fool
   a language discriminator has no incentive to preserve information. It can map
   every ciphertext to the same fluent English sentence and win. That is mode
   collapse, and on this task it should be visible directly in the decoded
   samples, not just inferred from a metric.

   Concretely, watch for high discriminator-fooling with near-chance character
   accuracy, and for the entropy of the output distribution collapsing. Both are
   logged below.

Everything structural is inherited from CipherCycleGANModel; only the B->A
direction and the cycle terms are removed.
"""

import os

import torch

from models.cipher_cycle_gan_model import CipherCycleGANModel


class CipherGANModel(CipherCycleGANModel):
    """A (ciphertext) -> B (plaintext), adversarial loss only."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CipherCycleGANModel.modify_commandline_options(parser, is_train)
        # lambda_A / lambda_B remain registered by the parent but are unused;
        # they are left in place so a single option set drives both models and
        # the two runs stay comparable on every other hyperparameter.
        return parser

    def __init__(self, opt):
        CipherCycleGANModel.__init__(self, opt)
        self.loss_names = ["D_A", "G_A", "entropy"]
        self.model_names = ["G_A", "D_A"] if self.isTrain else ["G_A"]

        # Free the unused halves so the reported parameter count
        # reflects what actually trains.
        if hasattr(self, "netG_B"):
            del self.netG_B
        if self.isTrain and hasattr(self, "netD_B"):
            del self.netD_B

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(
                self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # logits [B, L, V]

    def backward_G(self):
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # Diagnostic only - not added to the objective. Mean per-position
        # entropy of the output distribution over the batch. If the generator
        # collapses onto one token this falls towards zero while loss_G_A stays
        # healthy, which is the signature of mode collapse.
        with torch.no_grad():
            p = torch.softmax(self.fake_B, dim=-1)
            self.loss_entropy = -(p * (p + 1e-12).log()).sum(-1).mean()

        self.loss_G = self.loss_G_A
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD_A], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.optimizer_D.step()

    def _embeddings(self):
        for net_name in ["netG_A", "netD_A"]:
            net = getattr(self, net_name, None)
            if net is None:
                continue
            module = net.module if hasattr(net, "module") else net
            yield module.embedding

    @torch.no_grad()
    def compute_visuals(self):
        if not self.isTrain:
            return
        pred = self.netG_A(self.real_A).argmax(dim=-1)
        n = min(2, self.real_A.shape[0])
        block = [f"--- iter tau={self.current_tau:.4f} "
                 f"H={float(self.loss_entropy):.3f} ---"]
        for k in range(n):
            block.append(f"  cipher in : {self._decode(self.real_A[k])[:70]}")
            block.append(f"  decrypted : {self._decode(pred[k])[:70]}")
        text = "\n".join(block)
        print(text)
        with open(os.path.join(self.save_dir, "samples.txt"), "a") as fh:
            fh.write(text + "\n")
