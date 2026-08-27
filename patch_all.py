"""
patch_all.py  -  one patch, everything.

    python patch_all.py

Run from the repository root, AFTER patch_shared_embedding.py.
Safe to run twice. Prints a diagnostic command if any anchor is missing.

Brings training in line with the mechanism CipherGAN (Gomez et al. 2018,
arXiv:1801.04883) describes, rather than with its visible hyperparameters only.

  1. --share_embedding   one embedding table W_Emb shared by both generators
                         and both discriminators, and placed in the
                         discriminator optimiser so that it is trained to
                         maximise the GAN loss as the paper specifies. Its
                         non-stationarity is what Proposition 1 relies on to
                         prevent uninformative discrimination.
  2. deduplication       a shared table is reachable from four networks, so
                         naive chaining hands it to the optimiser two or three
                         times and PyTorch applies one update per occurrence.
                         Verified experimentally. Parameters are now unique.
  3. --warmup_steps      "exponentially warmed up to 2e-4 over 2500 steps"
                         (Section 4.2), absent until now.
  4. --beta2             paper uses Adam beta2 = 0.9; PyTorch defaults to 0.999.
  5. --cycle_loss simplex_l1
                         the paper's cycle term: L1 between the softmax output
                         and the one-hot input, on the probability simplex.
  6. auto --matched_softness when a gradient penalty is active, since the
                         penalty interpolates between real and generated and
                         therefore needs them to share a shape.
"""

import ast
import re

P = "models/cipher_cycle_gan_model.py"


def die(msg, cmd):
    raise SystemExit(f"\nANCHOR NOT FOUND: {msg}\nRun this and send the output:\n  {cmd}\n")


s = open(P).read()
done = []

# --------------------------------------------------------------- 1. flags ---
if "--share_embedding" not in s.split("def __init__")[0]:
    a = 'parser.add_argument("--n_blocks_G", type=int, default=4)'
    if a not in s:
        die("--n_blocks_G", "grep -n 'n_blocks_G' models/cipher_cycle_gan_model.py")
    s = s.replace(a, a + '''
        parser.add_argument("--share_embedding", action="store_true",
                            help="one embedding table across all four networks")
        parser.add_argument("--warmup_steps", type=int, default=2500,
                            help="exponential LR warmup; paper uses 2500")
        parser.add_argument("--beta2", type=float, default=0.9,
                            help="Adam beta2; paper 0.9, torch default 0.999")''', 1)
    done.append("flags --share_embedding, --warmup_steps, --beta2")

# ------------------------------------------------- 2. dedup helper function ---
if "_unique_params" not in s:
    m = re.search(r"^class CipherCycleGANModel", s, re.M)
    if not m:
        die("class CipherCycleGANModel", "grep -n 'class Cipher' models/cipher_cycle_gan_model.py")
    s = s[:m.start()] + '''def _unique_params(*iterables):
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


''' + s[m.start():]
    done.append("_unique_params helper")

# ------------------------------------------------- 3. share + auto softness ---
if "embedding table shared across" not in s:
    a = "            self.fake_A_pool = ImagePool(opt.pool_size)"
    if a not in s:
        die("fake_A_pool", "grep -n 'fake_A_pool' models/cipher_cycle_gan_model.py")
    s = s.replace(a, '''            # The gradient penalty interpolates between real and generated
            # inputs, so they must share a shape. In the paper both arrive as
            # embeddings (one-hot @ W and softmax @ W); here that presentation
            # comes from --matched_softness.
            if (getattr(opt, "use_gp", False) or opt.gan_mode == "wgangp") \\
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

''' + a, 1)
    done.append("shared embedding + auto matched_softness")

# ------------------------------------------------------------ 4. optimisers ---
if "_unique_params(self.netG_A" not in s:
    old_g = """            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(),
                                self.netG_B.parameters()),"""
    if old_g not in s:
        die("optimizer_G", "sed -n '150,170p' models/cipher_cycle_gan_model.py")
    s = s.replace(old_g, """            self.optimizer_G = torch.optim.Adam(
                _unique_params(self.netG_A.parameters(),
                               self.netG_B.parameters()),""", 1)
    old_d = """            _dp = itertools.chain(self.netD_A.parameters(),
                                  self.netD_B.parameters())"""
    if old_d not in s:
        die("_dp chain", "sed -n '155,170p' models/cipher_cycle_gan_model.py")
    s = s.replace(old_d, """            _dp = [self.netD_A.parameters(), self.netD_B.parameters()]
            if getattr(opt, "share_embedding", False):
                # the shared table also receives the discriminator objective,
                # i.e. it is trained to MAXIMISE the GAN loss, per the paper
                _emb = (self.netG_A.module if hasattr(self.netG_A, "module")
                        else self.netG_A).embedding
                _dp = _dp + [_emb.parameters()]""", 1)
    s = s.replace("""            self.optimizer_D = torch.optim.Adam(
                _dp, lr=opt.lr,""", """            self.optimizer_D = torch.optim.Adam(
                _unique_params(*_dp), lr=opt.lr,""", 1)
    done.append("optimiser deduplication + embedding in D")

if "betas=(opt.beta1, 0.999)" in s:
    s = s.replace("betas=(opt.beta1, 0.999)",
                  "betas=(opt.beta1, getattr(opt, 'beta2', 0.9))")
    done.append("beta2")

# --------------------------------------------------------------- 5. warmup ---
if "_apply_warmup" not in s:
    m = re.search(r"^    def optimize_parameters\(self\):", s, re.M)
    if not m:
        die("optimize_parameters", "grep -n 'def optimize_parameters' models/cipher_cycle_gan_model.py")
    warm = '''    def _apply_warmup(self):
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

'''
    s = s[:m.start()] + warm + s[m.start():]
    m2 = re.search(r"(    def optimize_parameters\(self\):\n(?:        \"\"\".*?\"\"\"\n)?)",
                   s, re.S)
    s = s[:m2.end(1)] + "        self._apply_warmup()\n" + s[m2.end(1):]
    done.append("exponential LR warmup")

# ---------------------------------------------------------- 6. simplex cycle ---
if "simplex_l1" not in s:
    s = s.replace('choices=["ce", "l1"]', 'choices=["ce", "l1", "simplex_l1"]')
    m = re.search(r"    def _cycle\(self, logits, target_idx\):\n", s)
    if not m:
        die("_cycle", "grep -n 'def _cycle' models/cipher_cycle_gan_model.py")
    s = s[:m.end()] + '''        if getattr(self.opt, "cycle_loss", "ce") == "simplex_l1":
            # The paper's term: L1 between the softmax distribution and the
            # original one-hot, on the probability simplex. This is not the
            # degenerate case of an L1 on embeddings, which could be minimised
            # by shrinking embedding norms.
            probs = torch.softmax(logits / max(self.current_tau, 1e-6), dim=-1)
            oh = torch.nn.functional.one_hot(target_idx, self.opt.vocab_size)
            mask = (target_idx != 0).unsqueeze(-1).float()
            return ((probs - oh.float()).abs() * mask).sum(-1).mean()
''' + s[m.end():]
    done.append("--cycle_loss simplex_l1")

ast.parse(s)
open(P, "w").write(s)
print("applied:\n  " + "\n  ".join(done) if done else "nothing to do (already patched)")
