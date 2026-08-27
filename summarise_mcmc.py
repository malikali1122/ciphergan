import json, glob, os
print(f"{'condition':20s} {'cipher':12s} {'p':>2s} {'V':>3s} {'best':>7s} {'mean':>7s} {'solved':>7s} {'maj':>7s} {'sec':>6s}")
for p in sorted(glob.glob("results/mcmc/*.json")):
    d = json.load(open(p))
    print(f"{os.path.basename(p)[:-5]:20s} {d['cipher']:12s} {d['period']:2d} "
          f"{d['vocab_size']:3d} {d['key_recovery_best']:7.4f} "
          f"{d['key_recovery_mean']:7.4f} {str(d['n_solved'])+'/10':>7s} "
          f"{d['majority_class_accuracy']:7.4f} {d['seconds']:6.1f}")
