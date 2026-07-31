import re, ast
n, m = 'models/networks_seq.py', 'models/cipher_cycle_gan_model.py'
s = open(n).read(); changed = []

if 'k=3' not in s:                       # ResBlock gets a kernel size
    old = '''    def __init__(self, ch, dilation, norm_layer):
        super().__init__()
        pad = dilation'''
    assert old in s, "ResBlock1d anchor missing"
    s = s.replace(old, '''    def __init__(self, ch, dilation, norm_layer, k=3):
        super().__init__()
        # k=1 keeps DEPTH while removing CONTEXT, as CipherGAN does.
        pad = 0 if k == 1 else dilation
        dilation = 1 if k == 1 else dilation''')
    s = s.replace('nn.Conv1d(ch, ch, 3, padding=pad, dilation=dilation)',
                  'nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation)')
    changed.append('ResBlock1d kernel')

if 'k=1 if pointwise else 3' not in s:    # pointwise keeps the blocks
    old = '''        self.body = nn.Sequential(*([] if pointwise else [
            ResBlock1d(ngf, dilations[i % len(dilations)], norm_layer)
            for i in range(n_blocks)
        ]))'''
    assert old in s, "generator body anchor missing"
    s = s.replace(old, '''        self.body = nn.Sequential(*[
            ResBlock1d(ngf, dilations[i % len(dilations)], norm_layer,
                       k=1 if pointwise else 3)
            for i in range(n_blocks)
        ])''')
    changed.append('pointwise depth')

if 'kw=4,' not in s:                      # discriminator kernel width
    s = s.replace('n_layers=3,', 'n_layers=3, kw=4,', 1)
    s = s.replace('nn.Conv1d(embed_dim, ndf, 4, stride=2, padding=1)',
                  'nn.Conv1d(embed_dim, ndf, kw, stride=2, padding=kw // 2)')
    s = s.replace('nn.Conv1d(ndf * prev, ndf * mult, 4, stride=2, padding=1)',
                  'nn.Conv1d(ndf * prev, ndf * mult, kw, stride=2, padding=kw // 2)')
    s = s.replace('nn.Conv1d(ndf * mult, 1, 4, stride=1, padding=1)',
                  'nn.Conv1d(ndf * mult, 1, kw, stride=1, padding=kw // 2)')
    changed.append('discriminator kernel width')

if '_ChannelLayerNorm' not in s:          # position-safe layer norm
    s = s.replace('def get_norm_layer_1d(', '''class _ChannelLayerNorm(nn.Module):
    """Normalise over channels at each position. GroupNorm(1,C) would pool
    across the sequence and hand a 'pointwise' generator hidden context."""
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):                 # [B, C, L]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def get_norm_layer_1d(''', 1)
    s = s.replace('    if norm_type == "none":',
                  '    if norm_type == "layer":\n        return lambda c: _ChannelLayerNorm(c)\n    if norm_type == "none":', 1)
    changed.append('layer norm')

ast.parse(s); open(n, 'w').write(s)

s = open(m).read()
if '--kw_D' not in s:
    a = 'parser.add_argument("--n_blocks_G", type=int, default=4)'
    assert a in s, "n_blocks_G anchor missing"
    s = s.replace(a, a + '\n        parser.add_argument("--kw_D", type=int, default=4)', 1)
    s = s.replace('SeqDiscriminator(V, E, opt.ndf, opt.n_layers_D, norm_layer=norm_layer)',
                  'SeqDiscriminator(V, E, opt.ndf, opt.n_layers_D,\n'
                  '                             kw=getattr(opt, "kw_D", 4), norm_layer=norm_layer)')
    changed.append('--kw_D')

if '--use_gp' not in s:
    mt = re.search(r'(        parser\.add_argument\("--lambda_gp".*?\)\n)', s, re.S)
    assert mt, "lambda_gp anchor missing"
    s = s[:mt.end(1)] + '        parser.add_argument("--use_gp", action="store_true")\n' + s[mt.end(1):]
    s = s.replace('if self.opt.gan_mode == "wgangp":',
                  'if self.opt.gan_mode == "wgangp" or getattr(self.opt, "use_gp", False):')
    changed.append('--use_gp')

ast.parse(s); open(m, 'w').write(s)
print("applied:", ", ".join(changed) if changed else "nothing (already patched)")
