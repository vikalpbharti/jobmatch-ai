"""Search the index from the terminal.

  python scripts/search.py "build ETL pipelines on the cloud"
  python scripts/search.py "machine learning for fraud" --location India -k 5
  python scripts/search.py --resume my_resume.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jobmatch.search import JobSearch  # noqa: E402


def show(r: dict, rank: int) -> None:
    score = f"score {r['final_score']:.3f} (semantic {r['score']:.3f}, skills {r['skill_overlap']:.0%})" \
        if "final_score" in r else f"score {r['score']:.3f}"
    print(f"{rank:>2}. {r['title']} | {r['company']} | {r['location']}  [{score}]")
    if r.get("matched_skills") is not None:
        print(f"    has: {', '.join(r['matched_skills']) or '-'}   missing: {', '.join(r['missing_skills']) or '-'}")
    elif r["skills"]:
        print(f"    skills: {', '.join(r['skills'][:10])}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?")
    p.add_argument("--resume", help="Path to a plain-text resume to match against jobs")
    p.add_argument("--location")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--index", default="index")
    args = p.parse_args()
    if not args.query and not args.resume:
        p.error("give a query or --resume")

    index = JobSearch.load(args.index)
    if args.resume:
        out = index.match_resume(Path(args.resume).read_text(encoding="utf-8"), k=args.k, location=args.location)
        print(f"Skills found in resume: {', '.join(out['resume_skills']) or '-'}\n")
        results = out["matches"]
    else:
        results = index.search(args.query, k=args.k, location=args.location)
    for rank, r in enumerate(results, 1):
        show(r, rank)


if __name__ == "__main__":
    main()
