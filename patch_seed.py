import ast
p = 'models/cipher_cycle_gan_model.py'
s = open(p).read()
if 'torch.manual_seed' in s:
    print("already patched"); raise SystemExit

old = '    def __init__(self, opt):\n'
i = s.index('class CipherCycleGANModel')
j = s.index(old, i)
assert j > 0, "__init__ anchor missing"

ins = '''        # The reference train.py never seeds torch, so network initialisation
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
'''
s = s[:j + len(old)] + ins + s[j + len(old):]
ast.parse(s); open(p, 'w').write(s)
print("patched: --pair_seed now controls network init as well as data pairing")
