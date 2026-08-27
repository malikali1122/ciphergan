"""Cipher dataset for pytorch-CycleGAN-and-pix2pix.

Drop this file at `data/cipher_dataset.py` in your fork.

The repo discovers datasets by filename convention: `--dataset_mode cipher`
imports `data.cipher_dataset` and looks for a `BaseDataset` subclass whose
lowercased name is `cipherdataset`. Nothing else needs registering.

Domain A = ciphertext, domain B = plaintext, matching the repo's
"G_A: A -> B" convention, so G_A is the decryption model.
"""

import json
import os

import numpy as np
import torch

from data.base_dataset import BaseDataset


class CipherDataset(BaseDataset):
    """Serves unpaired (ciphertext, plaintext) samples from the cipher engine.

    Index `i` selects a ciphertext sample deterministically and a plaintext
    sample at random. No item ever contains a plaintext and its own
    ciphertext, which is what makes the task unsupervised.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--npz_path", type=str, required=True,
                            help="dataset.npz produced by make_data.py")
        parser.add_argument("--pair_seed", type=int, default=0,
                            help="seed for drawing the unpaired plaintext partner")
        # Image-preprocessing options are meaningless for sequences.
        parser.set_defaults(preprocess="none", no_flip=True, serial_batches=False)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        if not os.path.exists(opt.npz_path):
            raise FileNotFoundError(
                f"{opt.npz_path} not found. Generate it first with:\n"
                f"  python make_data.py --corpus corpus.txt --cipher substitution "
                f"--out_dir data/sub")
        d = np.load(opt.npz_path, allow_pickle=True)

        split = "train" if opt.isTrain else "eval"
        if split == "train":
            self.A = np.ascontiguousarray(d["train_cipher"], dtype=np.int64)
            self.B = np.ascontiguousarray(d["train_plain"], dtype=np.int64)
            self.paired = False
        else:
            # The eval split IS aligned. It is used only to score CER after
            # training; never let a loss touch it.
            self.A = np.ascontiguousarray(d["eval_cipher"], dtype=np.int64)
            self.B = np.ascontiguousarray(d["eval_plain"], dtype=np.int64)
            self.paired = True

        self.meta = json.loads(str(d["meta"]))
        self.symbols = list(d["symbols"])
        self.key_table = d["key_table"]
        self.rng = np.random.default_rng(opt.pair_seed)

        if not self.paired:
            self._assert_unpaired()

        # Vocabulary spans both domains when separate_domains was used. Take it
        # from the metadata rather than max()+1 over the data: a rare symbol
        # absent from one split would otherwise give the two splits different
        # vocab sizes and silently break checkpoint loading at eval time.
        c = self.meta["cipher"]
        self.vocab_size = int(c["vocab_size"]) + int(c["domain_offset"])

        # train.py builds the dataset before the model, so publishing the size
        # here means the user never has to pass --vocab_size by hand.
        # The model warns when a periodic cipher is trained without
        # positional encoding, which it can only do if it knows the period.
        opt.cipher_period = int(c.get("period", 1))

        if getattr(opt, "vocab_size", -1) in (None, -1):
            opt.vocab_size = self.vocab_size
        elif opt.vocab_size != self.vocab_size:
            raise ValueError(
                f"--vocab_size {opt.vocab_size} contradicts the dataset "
                f"({self.vocab_size}). Omit the flag and let it be inferred.")

        self.sample_length = int(self.A.shape[1])
        print(f"[cipher] {split}: A(cipher)={self.A.shape} B(plain)={self.B.shape} "
              f"vocab={self.vocab_size} cipher={c['name']} paired={self.paired}")

    def _assert_unpaired(self):
        """Fail loudly if a plaintext and its own ciphertext are both present.

        This is the single most damaging silent bug available in this project:
        it produces a supervised model that looks like a breakthrough.
        """
        overlap = {r.tobytes() for r in self.A} & {r.tobytes() for r in self.B}
        if overlap:
            raise ValueError(
                f"{len(overlap)} rows are identical across domains A and B. "
                "Either the cipher is the identity, or the split leaked and "
                "training would be covertly supervised.")

    def __len__(self):
        return len(self.A)

    def __getitem__(self, index):
        i = index % len(self.A)
        j = i if self.paired else int(self.rng.integers(len(self.B)))
        return {
            "A": torch.from_numpy(self.A[i].copy()),   # [L] long, ciphertext
            "B": torch.from_numpy(self.B[j].copy()),   # [L] long, plaintext
            "A_paths": f"cipher_{i}",
            "B_paths": f"plain_{j}",
        }
