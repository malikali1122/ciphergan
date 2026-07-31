"""Evaluate a trained cipher model on the held-out paired split.

    python eval_cipher.py --dataroot . --npz_path data/sub/dataset.npz \
        --name sub_cyclegan --model cipher_cycle_gan

Reports three numbers, which are the three numbers the results chapter needs:

  character accuracy  fraction of characters decrypted correctly. The headline
                      metric. Chance is ~1/27 = 3.7% for a 27-symbol alphabet,
                      though a generator that has only learned unigram
                      frequencies will beat chance without having solved
                      anything, so read it next to the other two.

  key recovery        the model's implied substitution table, read off its own
                      outputs by majority vote, scored against the true key. No
                      labels are used to produce it. This is the metric that
                      distinguishes "learned the cipher" from "learned to emit
                      plausible English", and it is the one worth leading with.

  coverage            the fraction of the alphabet the key-recovery score was
                      computed over. Symbols absent from the eval corpus are
                      excluded rather than counted as errors, so a low coverage
                      means the accuracy figure is on thin evidence.

The eval split is aligned. Nothing here ever touches a loss - this script only
runs after training, on a checkpoint.
"""

import json
import os

import numpy as np
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

from cipher_engine import (build_cipher, character_accuracy,
                           infer_mapping_from_outputs, mapping_accuracy,
                           Cipher, CROP_AMOUNT)


def rebuild_cipher(meta):
    """Reconstruct the ground-truth Cipher object from the dataset metadata."""
    c = meta["cipher"]
    return Cipher(name=c["name"],
                  key_table=np.array(c["key_table"], dtype=np.int64),
                  vocab_size=int(c["vocab_size"]),
                  period=int(c["period"]),
                  key_repr=c.get("key", ""),
                  separate_domains=bool(c["separate_domains"]))


def main():
    opt = TestOptions().parse()
    # test.py sets this itself; TestOptions does not, and BaseModel reads it.
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.batch_size = 64
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    ds = dataset.dataset  # the CipherDataset itself
    assert ds.paired, "eval must run on the aligned split"

    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    cipher = rebuild_cipher(ds.meta)
    preds, truths, inputs = [], [], []

    for data in dataset:
        model.set_input(data)
        with torch.no_grad():
            out = model.decrypt(model.real_A)
        preds.append(out.cpu().numpy())
        truths.append(model.real_B.cpu().numpy())
        inputs.append(model.real_A.cpu().numpy())

    pred = np.concatenate(preds)
    truth = np.concatenate(truths)
    cin = np.concatenate(inputs)

    acc = character_accuracy(pred, truth)
    table = infer_mapping_from_outputs(cin, pred, ds.vocab_size,
                                       period=cipher.period)
    key_acc, coverage = mapping_accuracy(table, cipher)

    syms = ds.symbols
    dec = lambda r: "".join("_" if i == 0 else str(syms[i]) if i < len(syms)
                            else "?" for i in r)

    print("\n" + "=" * 62)
    print(f"cipher              {cipher.name} (key={cipher.key_repr})")
    print(f"samples             {len(pred)} x {pred.shape[1]}")
    print(f"character accuracy  {acc:.4f}   (chance ~{1 / (ds.vocab_size - 1):.4f})")
    print(f"character error     {1 - acc:.4f}")
    print(f"key recovery        {key_acc:.4f}   (coverage {coverage:.3f})")
    print("=" * 62)
    print(f"cipher in : {dec(cin[0])[:70]}")
    print(f"decrypted : {dec(pred[0])[:70]}")
    print(f"truth     : {dec(truth[0])[:70]}")

    out_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as fh:
        json.dump({
            "cipher": cipher.name,
            "key": cipher.key_repr,
            "n_samples": int(len(pred)),
            "character_accuracy": float(acc),
            "character_error_rate": float(1 - acc),
            "key_recovery_accuracy": float(key_acc),
            "key_recovery_coverage": float(coverage),
            "chance_accuracy": float(1 / (ds.vocab_size - 1)),
        }, fh, indent=2)
    print(f"\nwrote {out_dir}/eval_metrics.json")


if __name__ == "__main__":
    main()
