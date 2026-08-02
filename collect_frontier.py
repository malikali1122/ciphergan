import json, os, numpy as np
print(f"{'n':>3} {'best10':>7} {'mean':>7} {'base':>7} {'norm':>7} {'succ':>6} {'zeros':>5}")
for d, n in [("a7",7),("a10",10),("a14",14),("a20",20),("a27",27)]:
    a=[]
    for s in range(10):
        p=f"checkpoints/{d}_s{s}_gp1/eval_metrics.json"
        if os.path.exists(p): a.append(json.load(open(p))["character_accuracy"])
    if not a: print(f"{n:>3} no runs"); continue
    a=np.array(a)
    x=np.load(f"data/{d}/dataset.npz",allow_pickle=True)["eval_plain"].ravel()
    x=x[x!=0]; b=np.bincount(x).max()/len(x)
    print(f"{n:>3} {a.max():>7.3f} {a.mean():>7.3f} {b:>7.3f} "
          f"{(a.max()-b)/(1-b):>7.3f} {sum(a>b)}/{len(a):<4} {int((a==0).sum()):>5}")
