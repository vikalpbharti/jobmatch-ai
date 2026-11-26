"""Evaluate retrieval quality: embeddings vs a TF-IDF keyword baseline.

Each test query describes the WORK of a role without naming the role (e.g. "build ETL pipelines"
for Data Engineer). A result counts as relevant when its category is that role, using the
dataset's own category labels as ground truth. Metrics:
  precision@k  share of the top k results that are relevant
  MRR          1 / rank of the first relevant result, averaged over queries

  python eval/evaluate.py --index index
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobmatch.embed import Embedder  # noqa: E402
from jobmatch.search import JobSearch  # noqa: E402

QUERIES_FILE = Path(__file__).with_name("queries.jsonl")


def load_queries(categories: set[str]) -> list[dict]:
    queries = [json.loads(line) for line in QUERIES_FILE.read_text().splitlines() if line.strip()]
    known = {c.lower() for c in categories}
    kept = [q for q in queries if any(q["category"].lower() in c for c in known)]
    if not kept:
        raise SystemExit("None of the eval categories exist in this index. Edit eval/queries.jsonl to match your data.")
    return kept


def is_relevant(job: dict, category: str) -> bool:
    # "Senior Data Engineer" counts as relevant for a "Data Engineer" query.
    return category.lower() in job["category"].lower()


def evaluate(index: JobSearch, queries: list[dict], k: int) -> dict:
    precisions, rrs = [], []
    per_query = []
    for q in queries:
        results = index.search(q["query"], k=max(k, 50))
        rel = [is_relevant(r, q["category"]) for r in results]
        p_at_k = sum(rel[:k]) / k
        first = next((i for i, x in enumerate(rel) if x), None)
        rr = 1.0 / (first + 1) if first is not None else 0.0
        precisions.append(p_at_k)
        rrs.append(rr)
        per_query.append((q["category"], q["query"], p_at_k))
    return {"precision": float(np.mean(precisions)), "mrr": float(np.mean(rrs)), "per_query": per_query}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", default="index")
    p.add_argument("-k", type=int, default=10)
    args = p.parse_args()

    main_index = JobSearch.load(args.index)
    if not any(j["category"] for j in main_index.jobs):
        raise SystemExit("This index has no category labels, so it can't be scored automatically.")
    queries = load_queries({j["category"] for j in main_index.jobs})

    systems = {}
    if main_index.embedder.backend != "tfidf":
        print("Building TF-IDF baseline on the same jobs...")
        systems["TF-IDF (keyword baseline)"] = JobSearch.build(main_index.jobs, Embedder("tfidf"), show_progress=False)
    systems[f"{main_index.embedder.model_name}"] = main_index

    print(f"\n{len(queries)} queries over {len(main_index.jobs)} jobs\n")
    print(f"{'System':<50} {'P@' + str(args.k):>8} {'MRR':>8}")
    results = {}
    for name, idx in systems.items():
        results[name] = evaluate(idx, queries, args.k)
        print(f"{name:<50} {results[name]['precision']:>8.3f} {results[name]['mrr']:>8.3f}")

    print("\nPer query (precision@k):")
    names = list(results)
    for i, (cat, query, _) in enumerate(results[names[0]]["per_query"]):
        scores = "  ".join(f"{results[n]['per_query'][i][2]:.2f}" for n in names)
        print(f"  [{cat}] {query[:60]:<60} {scores}")


if __name__ == "__main__":
    main()
