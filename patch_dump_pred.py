"""
patch_dump_pred.py

Run from the repository root. Safe to run twice.

Makes eval_cipher.py write two extra files per invocation, alongside the
existing eval_metrics.json:

  eval_metrics_<epoch>.json   the same metrics, but not clobbered by the next
                              --epoch in a sweep
  pred_<epoch>.npz            the model's decrypted output and the ciphertext
                              it was given

The npz is what select_checkpoint.py consumes. Dumping it lets checkpoint
selection run as pure numpy, with no torch and no model instantiation, which
means the selection logic can be tested independently of the cluster.

No new command-line flags: adding options risks colliding with the existing
TestOptions plumbing, and these files are small.
"""

import ast

P = "eval_cipher.py"
s = open(P).read()

if "pred_dump" in s:
    raise SystemExit("nothing to do (already patched)")

anchor = """    out_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(out_dir, exist_ok=True)"""
if anchor not in s:
    raise SystemExit(
        "\nANCHOR NOT FOUND: out_dir\nRun this and send the output:\n"
        "  grep -n 'out_dir' eval_cipher.py\n")

s = s.replace(anchor, anchor + '''

    # Raw outputs, for unsupervised checkpoint selection. cin is the ciphertext
    # the model was given, pred is what it produced. truth is deliberately NOT
    # saved: the selector must not be able to reach it even by accident.
    pred_dump = os.path.join(out_dir, f"pred_{opt.epoch}.npz")
    np.savez_compressed(pred_dump, pred=pred.astype(np.int16),
                        cin=cin.astype(np.int16))''', 1)

old_write = """    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as fh:
        json.dump({"""
new_write = """    metrics = {"""
s = s.replace(old_write, new_write, 1)

old_tail = """            "chance_accuracy": float(1 / (ds.vocab_size - 1)),
        }, fh, indent=2)
    print(f"\\nwrote {out_dir}/eval_metrics.json")"""
new_tail = """            "chance_accuracy": float(1 / (ds.vocab_size - 1)),
            "epoch": str(opt.epoch),
    }
    # Both: the flat name for anything already reading it, and a per-epoch name
    # so a sweep does not overwrite its own earlier results.
    for fname in ("eval_metrics.json", f"eval_metrics_{opt.epoch}.json"):
        with open(os.path.join(out_dir, fname), "w") as fh:
            json.dump(metrics, fh, indent=2)
    print(f"\\nwrote {out_dir}/eval_metrics_{opt.epoch}.json and {pred_dump}")"""

if old_tail not in s:
    raise SystemExit(
        "\nANCHOR NOT FOUND: json tail\nRun this and send the output:\n"
        "  sed -n '108,127p' eval_cipher.py\n")
s = s.replace(old_tail, new_tail, 1)

# the dict body was indented for json.dump(...); re-indent it for `metrics = {`
s = s.replace('''    metrics = {
            "cipher": cipher.name,''', '''    metrics = {
            "cipher": cipher.name,''', 1)

ast.parse(s)
open(P, "w").write(s)
print("applied: eval_cipher.py dumps pred_<epoch>.npz and eval_metrics_<epoch>.json")
