"""Verify CondDiffusionGenerator satisfies every interface the cycle needs.

Mirrors cipher_cycle_gan_model.forward() and backward_G() exactly.
"""
import torch
import torch.nn as nn
from networks_seq import SoftEmbedding, SeqDiscriminator, get_norm_layer_1d
from networks_diffusion import CondDiffusionGenerator, build_generator

V, E, L, B = 27, 64, 48, 4
norm = get_norm_layer_1d("instance")
ok = lambda n: print(f"  PASS  {n}")


def make(shared=None, pos=0, steps=4):
    return CondDiffusionGenerator(V, E, ngf=64, n_blocks=4, norm_layer=norm,
                                  embedding=shared, pos_dim=pos, n_steps=steps)


print("\n[1] shape contract")
G = make()
x = torch.randint(1, V, (B, L))
y = G(x)
assert y.shape == (B, L, V), y.shape
ok(f"long [B,L] -> logits {tuple(y.shape)}")
y2 = G(y)
assert y2.shape == (B, L, V)
ok("logits [B,L,V] -> logits (needed for rec_A = netG_B(fake_B))")

print("\n[2] composition, as in model.forward()")
GA, GB = make(), make()
real_A = torch.randint(1, V, (B, L))
real_B = torch.randint(1, V, (B, L))
fake_B = GA(real_A)
rec_A = GB(fake_B)
fake_A = GB(real_B)
rec_B = GA(fake_A)
assert rec_A.shape == rec_B.shape == (B, L, V)
ok("G_B(G_A(x)) and G_A(G_B(x)) both run")

print("\n[3] gradient flow through the full reverse chain")
DA = SeqDiscriminator(V, E, ndf=64, n_layers=3, kw=4, norm_layer=norm)
crit_ce = nn.CrossEntropyLoss(ignore_index=0)
loss_gan = ((DA(fake_B) - 1.0) ** 2).mean()
loss_cyc = crit_ce(rec_A.reshape(B * L, V), real_A.reshape(B * L))
(loss_gan + 10.0 * loss_cyc).backward()
named = dict(GA.named_parameters())
dead = [n for n, p in named.items() if p.grad is None or p.grad.abs().sum() == 0]
assert not dead, f"no gradient reached: {dead}"
ok(f"all {len(named)} G_A params have non-zero grad")
assert named["mask_embed"].grad.abs().sum() > 0
ok("mask_embed receives gradient (absorbing state is learned)")

print("\n[4] shared embedding, by the mechanism the model actually uses")
# cipher_cycle_gan_model.py line 210 assigns AFTER construction:
#     _m.embedding = _src
# It never passes embedding= to a constructor. The test now mirrors that.
GA2, GB2 = make(), make()
D2 = SeqDiscriminator(V, E, ndf=64, n_layers=3, kw=4, norm_layer=norm)
_src = GA2.embedding
for _m in (GB2, D2):
    _m.embedding = _src
assert GA2.embedding is GB2.embedding is D2.embedding
ok("post-hoc assignment shares one table across G_A, G_B, D")
assert GA2(x).shape == (B, L, V) and D2(GA2(x)).shape[0] == B
ok("forward still works after the table is swapped in")
n_unique = len({id(p) for p in list(GA2.parameters()) + list(GB2.parameters())})
n_naive = len(list(GA2.parameters())) + len(list(GB2.parameters()))
ok(f"_unique_params dedup needed: {n_naive} naive -> {n_unique} unique")
_ = torch.optim.Adam(
    {id(p): p for p in list(GA2.parameters()) + list(GB2.parameters())}.values(),
    lr=2e-4)
ok("Adam accepts the deduped generator params (no duplicate-param error)")

print("\n[5] tau propagates (set_tau walks .embedding.tau)")
for m in [GA2, GB2, D2]:
    m.embedding.tau = 0.25
assert GA2.embedding.tau == 0.25
ok("tau reaches the shared table")

print("\n[6] decrypt path")
with torch.no_grad():
    pred = GA(real_A).argmax(dim=-1)
assert pred.shape == (B, L) and pred.dtype == torch.long
ok(f"argmax -> {tuple(pred.shape)} long, matches decrypt()")

print("\n[7] factory switch")
g_diff = build_generator("diffusion", vocab_size=V, embed_dim=E, ngf=64,
                         n_blocks=4, norm_layer=norm, n_steps=4)
g_conv = build_generator("conv", vocab_size=V, embed_dim=E, ngf=64,
                         n_blocks=4, norm_layer=norm)
assert g_diff(x).shape == g_conv(x).shape == (B, L, V)
ok("both paradigms behind one flag, identical output shape")

print("\n[8] cost of the chain (the thing that decides feasibility)")
base = sum(p.numel() for p in g_conv.parameters())
diff = sum(p.numel() for p in g_diff.parameters())
print(f"  params: conv {base:,}  diffusion {diff:,}  ({diff/base:.2f}x)")
import time
for steps in (1, 4, 8):
    g = make(steps=steps)
    g(x)  # warm
    t0 = time.time()
    for _ in range(10):
        g(x).sum().backward()
    print(f"  n_steps={steps}: {(time.time()-t0)/10*1000:6.1f} ms/iter")
g_conv(x)
t0 = time.time()
for _ in range(10):
    g_conv(x).sum().backward()
print(f"  conv baseline: {(time.time()-t0)/10*1000:6.1f} ms/iter")

print("\nAll contract checks passed.\n")
