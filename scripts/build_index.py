"""Build the job index.

  python scripts/build_index.py                          # 5,000 jobs from Hugging Face, sentence-transformers
  python scripts/build_index.py --source csv --path data/jobs.csv
  python scripts/build_index.py --backend tfidf --out index_tfidf   # keyword baseline
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobmatch.data import DEFAULT_HF_DATASET, load_csv, load_hf  # noqa: E402
from jobmatch.embed import DEFAULT_MODEL, Embedder  # noqa: E402
from jobmatch.search import JobSearch  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["hf", "csv"], default="hf")
    p.add_argument("--dataset", default=DEFAULT_HF_DATASET, help="Hugging Face dataset name")
    p.add_argument("--path", help="CSV path when --source csv")
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--backend", choices=["st", "tfidf"], default="st")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="index")
    args = p.parse_args()

    t0 = time.time()
    if args.source == "csv":
        if not args.path:
            p.error("--path is required with --source csv")
        jobs = load_csv(args.path, limit=args.limit)
    else:
        jobs = load_hf(args.dataset, limit=args.limit)
    print(f"Loaded {len(jobs)} jobs in {time.time() - t0:.1f}s")

    t0 = time.time()
    index = JobSearch.build(jobs, Embedder(args.backend, args.model))
    index.save(args.out)
    print(f"Embedded with {index.embedder.model_name} -> {index.vectors.shape} in {time.time() - t0:.1f}s")
    print(f"Saved index to {args.out}/")


if __name__ == "__main__":
    main()
