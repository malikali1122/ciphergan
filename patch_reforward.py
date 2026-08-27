"""
patch_reforward.py

Run from the repository root, AFTER patch_all_real.py. Safe to run twice.

THE BUG
-------
--share_embedding places W_Emb in optimizer_D, because the paper trains the
table to maximise the GAN loss as well as minimise the cycle loss. But
optimize_parameters builds the generator graph first and steps the
discriminator second:

    forward()             # netG_A/netG_B embed real_A/real_B; the graph now
                          # holds a reference to W_Emb at its current version
    backward_D_A/B()      # independent: these use fake.detach()
    optimizer_D.step()    # W_Emb is in this optimiser, so Adam mutates it
                          # in place and the version counter advances
    backward_G()          # tries to backprop the graph from forward(), which
                          # needs the pre-step W_Emb  ->  RuntimeError

This never fired before the shared embedding existed. Discriminator weights are
also reachable from the generator loss, but backward_G calls netD_A(fake_B)
itself, so those reads happen after the step. The embedding is consumed inside
forward(), before it.

THE FIX
-------
Re-run forward() after the discriminator steps and before backward_G, so the
generator graph is built against the current table. One extra forward pass per
iteration.

Guarded on --share_embedding so that every run already in the results chapter
stays bit-identical.

WHAT THIS IS NOT
----------------
Removing W_Emb from optimizer_D would also silence the error, and would also
delete the mechanism the experiment exists to test. Do not do that.
"""

import ast

P = "models/cipher_cycle_gan_model.py"
s = open(P).read()

if "rebuild the generator graph" in s:
    raise SystemExit("nothing to do (already patched)")

a = "        self.optimizer_G.zero_grad()"
if s.count(a) != 1:
    raise SystemExit(
        f"\nANCHOR NOT FOUND: optimizer_G.zero_grad ({s.count(a)} matches)\n"
        "Run this and send the output:\n"
        "  grep -n 'def optimize_parameters' -A 30 "
        "models/cipher_cycle_gan_model.py\n")

new = '''        if getattr(self.opt, "share_embedding", False):
            # rebuild the generator graph. optimizer_D.step() has just mutated
            # W_Emb in place, and the graph from the forward() above still
            # references the pre-step version of it. Autograd refuses to
            # backprop through that. Costs one forward pass per iteration.
            self.forward()
''' + a

s = s.replace(a, new, 1)
ast.parse(s)
open(P, "w").write(s)
print("applied: forward() rebuilt before the generator step under --share_embedding")
