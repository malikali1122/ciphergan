"""
fetch_corpus.py - run ONCE on the HPC login node (compute nodes have no network)

    python fetch_corpus.py --corpus brown --out mydata.txt

Converts an NLTK corpus into a plain text file so that nltk is never needed
again. After this, nothing in the training pipeline touches the network.

WHY CONVERT RATHER THAN CALL NLTK AT TRAIN TIME
-----------------------------------------------
Calling nltk.download() from inside a batch job fails on any cluster whose
compute nodes lack outbound network, and it fails late - after the job has
queued and started. Converting once on the login node removes nltk from the
runtime dependency set entirely and makes the exact training corpus a
concrete, citable artefact that can be archived with the results.

ON BROWN'S TOKENISATION
-----------------------
brown.words() returns tokens, not raw prose, so joining with spaces detaches
punctuation ("said Friday ." and "`` no evidence ''"). This looks like it would
distort the character statistics the attack depends on - but Vocab.clean()
strips everything outside a-z and space and collapses runs of whitespace, which
removes the artefacts on its own. Measured, cleaned:

    brown, joined tokens     5,751,046 chars   IoC 0.0753   space 17.48%
    brown, detokenised       5,750,797 chars   IoC 0.0753   space 17.47%

Identical to four decimal places. Detokenising is wasted effort; don't bother.

CHOOSING A CORPUS
-----------------
    brown       5.75M clean chars   IoC 0.0753   mixed genre (news, fiction,
                                                 academic) - the default
    gutenberg  11.03M clean chars   IoC 0.0797   genuine raw prose, ~2x larger,
                                                 but all 19th-century literature

Both give realistic English character statistics. Brown is more genre-varied;
Gutenberg gives roughly twice the training samples, at the cost of a
narrower register.

SAMPLE COUNTS
-------------
Samples are non-overlapping, then deduplicated, then 10% is held out for
evaluation and the remainder is split in half so no text appears in both
domains. Brown therefore yields roughly:

    --sample_length  64   ->  ~40,000 samples per domain
    --sample_length 128   ->  ~20,000 samples per domain
    --sample_length 256   ->  ~10,000 samples per domain

Longer samples give the discriminator more context but fewer examples. Do not
try to recover the count with overlapping chunks: two overlapping chunks can
land on opposite sides of the plaintext/ciphertext split, which quietly
reintroduces the pairing the whole setup exists to prevent.
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="brown", choices=["brown", "gutenberg"])
    p.add_argument("--out", default="mydata.txt")
    p.add_argument("--nltk_data", default=None,
                   help="download target; defaults to ~/nltk_data")
    args = p.parse_args()

    try:
        import nltk
    except ImportError:
        sys.exit("nltk is not installed. pip install nltk")

    if args.nltk_data:
        os.makedirs(args.nltk_data, exist_ok=True)
        nltk.data.path.insert(0, args.nltk_data)

    print(f"downloading {args.corpus} ...")
    ok = nltk.download(args.corpus, download_dir=args.nltk_data, quiet=False)
    if not ok:
        sys.exit(
            "\nDownload failed - the login node may be behind a proxy.\n"
            "Fallback: fetch the zip on another machine and copy it over:\n"
            "  https://raw.githubusercontent.com/nltk/nltk_data/"
            "gh-pages/packages/corpora/brown.zip\n"
            "  scp brown.zip USER@host:~/nltk_data/corpora/\n"
            "  ssh USER@host 'cd ~/nltk_data/corpora && unzip brown.zip'")

    if args.corpus == "brown":
        from nltk.corpus import brown
        text = " ".join(brown.words())
    else:
        from nltk.corpus import gutenberg
        text = gutenberg.raw()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)

    size = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}  ({size:.1f} MB, {len(text):,} raw chars)")
    print("nltk is no longer needed. Next:")
    print(f"  CORPUS={args.out} LEN=128 ./run_training.sh data")


if __name__ == "__main__":
    main()
