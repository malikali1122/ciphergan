import re, ast
n, m = 'models/networks_seq.py', 'models/cipher_cycle_gan_model.py'
s = open(n).read()
if 'straight_through' not in s:
    old = '''        else:
            probs = torch.softmax(x / tau, dim=-1)
        return probs @ self.weight.weight'''
    assert old in s, "SoftEmbedding.forward anchor missing"
    s = s.replace(old, '''        else:
            probs = torch.softmax(x / tau, dim=-1)
        if getattr(self, "straight_through", False):
            # Exact one-hot forward, soft backward: both real and generated
            # inputs land on exact embedding lattice points, leaving the
            # discriminator no sharpness cue. Unlike Gumbel, no sampling noise.
            hard = torch.zeros_like(probs).scatter_(
                -1, probs.argmax(-1, keepdim=True), 1.0)
            probs = hard + probs - probs.detach()
        return probs @ self.weight.weight''')
    ast.parse(s); open(n, 'w').write(s); print("networks patched")
else:
    print("networks already patched")

s = open(m).read()
if '--straight_through' not in s:
    g = re.search(r'(        parser\.add_argument\("--gumbel".*?\)\n)', s, re.S)
    assert g, "--gumbel anchor missing"
    s = s[:g.end(1)] + '        parser.add_argument("--straight_through", action="store_true")\n' + s[g.end(1):]
    assert '    def set_gumbel(self, flag):' in s, "set_gumbel anchor missing"
    s = s.replace('    def set_gumbel(self, flag):',
                  '    def set_straight_through(self, flag):\n'
                  '        for emb in self._embeddings():\n'
                  '            emb.straight_through = bool(flag)\n\n'
                  '    def set_gumbel(self, flag):')
    c = re.search(r'(\n\s*self\.set_gumbel\([^\n]*\)\n)', s)
    assert c, "set_gumbel call site missing"
    s = s[:c.end(1)] + '        self.set_straight_through(getattr(opt, "straight_through", False))\n' + s[c.end(1):]
    ast.parse(s); open(m, 'w').write(s); print("model patched")
else:
    print("model already patched")
