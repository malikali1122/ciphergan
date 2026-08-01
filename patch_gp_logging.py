import ast
p = 'models/cipher_cycle_gan_model.py'
s = open(p).read()
if '"gp"' in s and 'self.loss_gp = torch.zeros' in s:
    print("already patched"); raise SystemExit
# report the penalty as its own loss so D_A stays comparable to 0.25
old = 'self.loss_names = ['
i = s.index(old); j = s.index(']', i)
names = s[i:j+1]
if '"gp"' not in names:
    s = s[:j] + ', "gp"' + s[j:]
# make sure it always exists, even when the penalty is off
anchor = '        loss_D = (loss_D_real + loss_D_fake) * 0.5\n'
assert anchor in s, "backward_D_basic anchor missing"
s = s.replace(anchor, anchor + '        self.loss_gp = torch.zeros((), device=self.device)\n', 1)
s = s.replace('            self.loss_gp = gp.detach()\n', '')
s = s.replace('            loss_D = loss_D + gp\n',
              '            loss_D = loss_D + gp\n            self.loss_gp = gp.detach()\n')
ast.parse(s); open(p, 'w').write(s)
print("patched: gp now logged separately; D_A is the LSGAN term alone plus gp")
