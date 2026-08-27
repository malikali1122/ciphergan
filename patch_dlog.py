"""
patch_dlog.py

Run from the repository root. Safe to run twice.

WHY
---
The logged ``D_A`` is not the adversarial loss. ``backward_D_basic`` computes

    loss_D = (loss_D_real + loss_D_fake) * 0.5 + gp

and returns that, so the penalty is folded into the number. At lambda_gp = 10
the penalty was roughly two thirds of the reported value, which means the
column you have been reading against LSGAN's 0.25 equilibrium was never
comparable to it.

Separately, ``self.loss_gp`` is written by ``backward_D_A`` and then overwritten
by ``backward_D_B``, so the single ``gp`` column has only ever shown D_B.

WHAT IT ADDS
------------
Eight columns, replacing the one ambiguous ``gp``:

  Dadv_A, Dadv_B   adversarial loss alone, without the penalty. THIS is the
                   number to read against 0.25. D -> 0 means the discriminator
                   has won outright and the generator's signal is dead.
  gp_A,   gp_B     the penalty for each discriminator, no longer overwritten.
  pr_A,   pr_B     mean raw D output on real input.
  pf_A,   pf_B     mean raw D output on generated input.

The last four are the diagnostic that matters. Under LSGAN a working
discriminator drives pr toward 1 and pf toward 0. If pr and pf sit on top of
each other, D cannot tell the domains apart and the adversarial term is
supplying no information. If pf exceeds pr, D is scoring generated sequences
as more real than real ones, and something is wrong with how the two are
presented rather than with the generator.

``self.loss_gp`` is still set, so anything reading it keeps working.
"""

import ast

P = "models/cipher_cycle_gan_model.py"
Q = "plot_curves.py"

s = open(P).read()
changed = []

# ------------------------------------------------------------- loss_names ---
old_names = ('        self.loss_names = ["D_A", "G_A", "cycle_A", '
             '"D_B", "G_B", "cycle_B", "gp"]')
new_names = '''        self.loss_names = ["D_A", "G_A", "cycle_A",
                           "D_B", "G_B", "cycle_B",
                           # adversarial loss WITHOUT the gradient penalty;
                           # D_A above still includes it, as it always has
                           "Dadv_A", "Dadv_B",
                           "gp_A", "gp_B",
                           # mean raw D output on real / generated input
                           "pr_A", "pf_A", "pr_B", "pf_B"]'''

if '"Dadv_A"' not in s:
    if old_names not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: loss_names\nRun this and send the output:\n"
            "  grep -n 'loss_names' models/cipher_cycle_gan_model.py\n")
    s = s.replace(old_names, new_names, 1)
    changed.append("loss_names: eight diagnostic columns")

# -------------------------------------------------------- backward_D_basic ---
if "def backward_D_basic(self, netD, real, fake, tag" not in s:
    old_sig = "    def backward_D_basic(self, netD, real, fake):"
    if old_sig not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: backward_D_basic signature\n"
            "Run this and send the output:\n"
            "  grep -n 'def backward_D_basic' models/cipher_cycle_gan_model.py\n")
    s = s.replace(old_sig,
                  '    def backward_D_basic(self, netD, real, fake, tag=""):',
                  1)

    old_body = """        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        self.loss_gp = torch.zeros((), device=self.device)
"""
    new_body = """        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_adv = (loss_D_real + loss_D_fake) * 0.5
        loss_D = loss_D_adv
        self.loss_gp = torch.zeros((), device=self.device)
"""
    if old_body not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: backward_D_basic body\n"
            "Run this and send the output:\n"
            "  sed -n '/def backward_D_basic/,/return loss_D/p' "
            "models/cipher_cycle_gan_model.py\n")
    s = s.replace(old_body, new_body, 1)

    old_tail = """        loss_D.backward()
        return loss_D
"""
    new_tail = """        if tag:
            # Recorded before backward so the numbers describe the state the
            # gradients were computed from. Detached: these are diagnostics.
            setattr(self, "loss_Dadv_" + tag, loss_D_adv.detach())
            setattr(self, "loss_gp_" + tag, self.loss_gp)
            setattr(self, "loss_pr_" + tag, pred_real.detach().mean())
            setattr(self, "loss_pf_" + tag, pred_fake.detach().mean())

        loss_D.backward()
        return loss_D
"""
    if old_tail not in s:
        raise SystemExit(
            "\nANCHOR NOT FOUND: backward_D_basic tail\n"
            "Run this and send the output:\n"
            "  sed -n '/def backward_D_basic/,/return loss_D/p' "
            "models/cipher_cycle_gan_model.py\n")
    s = s.replace(old_tail, new_tail, 1)
    changed.append("backward_D_basic records per-discriminator diagnostics")

# ------------------------------------------------------- pass the tag along ---
for letter, call in (("A", """        self.loss_D_A = self.backward_D_basic(
            self.netD_A, self._real_for_D(self.real_B), fake_B)"""),
                     ("B", """        self.loss_D_B = self.backward_D_basic(
            self.netD_B, self._real_for_D(self.real_A), fake_A)""")):
    if f'tag="{letter}"' in s:
        continue
    if call not in s:
        raise SystemExit(
            f"\nANCHOR NOT FOUND: backward_D_{letter} call\n"
            "Run this and send the output:\n"
            f"  grep -n 'def backward_D_{letter}' -A 4 "
            "models/cipher_cycle_gan_model.py\n")
    s = s.replace(call, call[:-1] + f', tag="{letter}")', 1)
    changed.append(f"backward_D_{letter} tagged")

ast.parse(s)
open(P, "w").write(s)

# ----------------------------------------------------- plot_curves panels ---
try:
    t = open(Q).read()
    old_groups = '    (["entropy", "gp"], "diagnostics"),'
    new_groups = ('    (["Dadv_A", "Dadv_B"], "adversarial loss, penalty excluded"),\n'
                  '    (["pr_A", "pf_A"], "D_A output: real vs generated"),\n'
                  '    (["pr_B", "pf_B"], "D_B output: real vs generated"),\n'
                  '    (["entropy", "gp_A", "gp_B"], "diagnostics"),')
    if old_groups in t and "Dadv_A" not in t:
        t = t.replace(old_groups, new_groups, 1)
        ast.parse(t)
        open(Q, "w").write(t)
        changed.append("plot_curves.py panels")
except FileNotFoundError:
    pass

print("applied:\n  " + "\n  ".join(changed) if changed
      else "nothing to do (already patched)")
