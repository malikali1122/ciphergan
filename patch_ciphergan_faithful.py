"""
patch_ciphergan_faithful.py

Brings the model in line with the mechanism CipherGAN (Gomez et al. 2018,
arXiv:1801.04883) actually describes, rather than with its visible
hyperparameters alone. Run from the repository root; safe to run twice.

WHAT THIS CHANGES AND WHY
=========================

1. DEDUPLICATE THE SHARED EMBEDDING IN THE OPTIMISERS  [bug fix]
   With --share_embedding, one table is referenced by all four networks, so
   itertools.chain(netG_A.parameters(), netG_B.parameters()) yields it twice
   and the discriminator chain yields it three times. PyTorch accepts this
   with a warning and steps the parameter once per occurrence, giving the
   embedding two to three times the intended learning rate. Verified
   experimentally. Parameters are now deduplicated by identity.

2. LEARNING-RATE WARMUP  [missing]
   Section 4.2: "Our learning rate is exponentially warmed up to 2e-4 over
   2500 steps, and held constant thereafter."
   Implemented as the tensor2tensor exponential warmup the authors would have
   had by default: lr_t = lr * 0.01 ** ((W - t) / W), which starts at one
   hundredth of the target and rises exponentially. Applied per iteration for
   the first --warmup_steps updates only; the repository's epoch-level
   scheduler takes over afterwards.

3. beta2 = 0.9  [was 0.999]
   Section 4.2 specifies Adam with beta1 = 0, beta2 = 0.9. PyTorch defaults to
   0.999. This is the WGAN-GP setting and matters for gradient-penalty
   training.

4. SIMPLEX L1 CYCLE LOSS  [--cycle_loss simplex_l1]
   The paper's cycle term is an L1 distance between the generator's softmax
   output and the original one-hot vector - a distance on the probability
   simplex. Earlier work in this project used cross-entropy on the argument
   that an L1 penalty on embeddings is degenerate, since it can be minimised
   by shrinking embedding norms. That argument does not apply here: the L1 is
   taken on probabilities, not embeddings, and has no such minimiser. Both are
   now available so the substitution can be measured rather than assumed.

5. --kw_D DEFAULT RESTORED TO 4  [correction]
   The paper states only that two-dimensional convolutions were replaced with
   one-dimensional ones and that "the filter sizes in our generators" were
   reduced to 1. It says nothing about the discriminator, which therefore
   inherits CycleGAN's width of 4. A previous value of 15 in this project came
   from a secondary source, not the paper.

WHAT REMAINS DIFFERENT
======================
The paper draws 2 * batch_size fresh plaintext samples per step, half of which
are enciphered, giving effectively unlimited data. This project uses a fixed
disjoint split with deduplication, which is the stricter protocol but supplies
far less diversity. That difference is left in place deliberately and should be
stated in the write-up rather than silently removed.

Also, the paper trains the embedding to minimise the cycle loss and maximise
the GAN loss. Here the shared table sits in both optimisers, so it additionally
receives the generator's fool-the-discriminator gradient. The essential
property - that the embedding is non-stationary, which is what Proposition 1
relies on to prevent uninformative discrimination - holds either way.
"""

import ast
import re

MODEL = "models/cipher_cycle_gan_model.py"
NETS = "models/networks_seq.py"
changed = []


# --------------------------------------------------------------------------- #
def patch_model():
    global changed
    s = open(MODEL).read()

    # ---- 1. new flags ----------------------------------------------------- #
    if "--warmup_steps" not in s:
        anchor = 'parser.add_argument("--share_embedding"'
        i = s.index(anchor)
        j = s.index("\n", s.index(")", s.index("help=", i)) if "help=" in
                    s[i:i + 400] else s.index(")", i))
        s = s[:j + 1] + (
            '        parser.add_argument("--warmup_steps", type=int, default=2500,\n'
            '                            help="exponential LR warmup; paper uses 2500")\n'
            '        parser.add_argument("--beta2", type=float, default=0.9,\n'
            '                            help="Adam beta2; paper uses 0.9, torch defaults 0.999")\n'
        ) + s[j + 1:]
        changed.append("flags --warmup_steps, --beta2")

    # ---- 2. beta2 --------------------------------------------------------- #
    if "betas=(opt.beta1, 0.999)" in s:
        s = s.replace("betas=(opt.beta1, 0.999)",
                      "betas=(opt.beta1, getattr(opt, 'beta2', 0.9))")
        changed.append("beta2 wired into both optimisers")

    # ---- 3. deduplicate optimiser parameters ------------------------------ #
    if "_unique_params" not in s:
        helper = '''
def _unique_params(*iterables):
    """Deduplicate by identity.

    A shared embedding table is reachable from several networks, so naive
    chaining passes the same tensor to the optimiser more than once. PyTorch
    accepts that and applies one update per occurrence, silently multiplying
    the learning rate for exactly the parameter the paper's mechanism depends
    on being updated in a controlled way.
    """
    seen, out = set(), []
    for it in iterables:
        for p in it:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


'''
        m = re.search(r"^class CipherCycleGANModel", s, re.M)
        s = s[:m.start()] + helper.lstrip("\n") + s[m.start():]

        s = s.replace(
            """            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(),
                                self.netG_B.parameters()),""",
            """            self.optimizer_G = torch.optim.Adam(
                _unique_params(self.netG_A.parameters(),
                               self.netG_B.parameters()),""")
        s = s.replace(
            """            _dp = itertools.chain(self.netD_A.parameters(),
                                  self.netD_B.parameters())""",
            """            _dp = [self.netD_A.parameters(), self.netD_B.parameters()]""")
        s = s.replace(
            """                _dp = itertools.chain(_dp, _emb.parameters())""",
            """                _dp = _dp + [_emb.parameters()]""")
        s = s.replace(
            """            self.optimizer_D = torch.optim.Adam(
                _dp, lr=opt.lr,""",
            """            self.optimizer_D = torch.optim.Adam(
                _unique_params(*_dp), lr=opt.lr,""")
        changed.append("optimiser parameter deduplication")

    # ---- 4. exponential LR warmup ----------------------------------------- #
    if "_apply_warmup" not in s:
        warm = '''    def _apply_warmup(self):
        """tensor2tensor-style exponential warmup, applied per iteration.

        lr_t = lr * 0.01 ** ((W - t) / W), rising from one hundredth of the
        target to the target over W steps. GAN training is unusually sensitive
        to the first few hundred updates, and the bimodal success pattern
        observed in this project - half the runs converging, half settling on
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

'''
        m = re.search(r"^    def optimize_parameters\(self\):", s, re.M)
        assert m, "optimize_parameters not found"
        s = s[:m.start()] + warm + s[m.start():]
        # call it at the top of the step
        s = re.sub(r"(    def optimize_parameters\(self\):\n(?:        \"\"\".*?\"\"\"\n)?)",
                   r"\1        self._apply_warmup()\n", s, count=1, flags=re.S)
        changed.append("exponential LR warmup")

    # ---- 5. simplex L1 cycle loss ----------------------------------------- #
    if "simplex_l1" not in s:
        s = s.replace('choices=["ce", "l1"]', 'choices=["ce", "l1", "simplex_l1"]')
        m = re.search(r"    def _cycle\(self, logits, target_idx\):\n", s)
        assert m, "_cycle not found"
        ins = '''        if getattr(self.opt, "cycle_loss", "ce") == "simplex_l1":
            # The paper's term: L1 between the softmax distribution and the
            # original one-hot vector, on the probability simplex.
            probs = torch.softmax(logits / max(self.current_tau, 1e-6), dim=-1)
            oh = torch.nn.functional.one_hot(target_idx, self.opt.vocab_size)
            mask = (target_idx != 0).unsqueeze(-1).float()
            return ((probs - oh.float()).abs() * mask).sum(-1).mean()
'''
        s = s[:m.end()] + ins + s[m.end():]
        changed.append("--cycle_loss simplex_l1")

    # ---- 6. kw_D default -------------------------------------------------- #
    s2 = re.sub(r'(add_argument\("--kw_D", type=int, default=)15', r"\g<1>4", s)
    if s2 != s:
        s = s2
        changed.append("--kw_D default 15 -> 4")

    ast.parse(s)
    open(MODEL, "w").write(s)


def main():
    patch_model()
    if changed:
        print("applied:")
        for c in changed:
            print("  -", c)
    else:
        print("nothing to do (already patched)")


if __name__ == "__main__":
    main()
