"""Load job postings from Hugging Face or a CSV into one common shape.

Every job becomes a dict with: id, title, category, company, location, skills (list), description.
"""
from __future__ import annotations

import ast

import pandas as pd

# Column names we accept for each field, in order of preference.
COLUMNS = {
    "title": ["job_title", "title"],
    "category": ["job_title_short", "category"],
    "company": ["company_name", "company"],
    "location": ["job_location", "location"],
    "skills": ["job_skills", "skills", "skills_desc"],
    "description": ["description", "job_description"],
}

DEFAULT_HF_DATASET = "lukebarousse/data_jobs"


def _parse_skills(value) -> list[str]:
    """Skills may arrive as a list, a stringified list "['python', 'sql']" or "python, sql"."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(s).strip().lower() for s in value if str(s).strip()]
    if hasattr(value, "tolist"):  # numpy array from parquet
        return _parse_skills(value.tolist())
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(s).strip().lower() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            pass
    return [s.strip().lower() for s in text.split(",") if s.strip()]


def _first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((n for n in names if n in df.columns), None)


def normalise(df: pd.DataFrame, limit: int | None = None, seed: int = 42) -> list[dict]:
    """Map a raw dataframe to job dicts. Samples evenly across categories when a category column exists."""
    cols = {field: _first_present(df, names) for field, names in COLUMNS.items()}
    if cols["title"] is None:
        raise ValueError(f"No title column found. Expected one of {COLUMNS['title']}, got {list(df.columns)}")

    df = df.dropna(subset=[cols["title"]]).sample(frac=1.0, random_state=seed)
    if limit and len(df) > limit:
        if cols["category"]:
            per_cat = max(1, limit // df[cols["category"]].nunique())
            df = df.groupby(cols["category"], group_keys=False).head(per_cat)
        df = df.head(limit)

    def get(row, field):
        col = cols[field]
        if col is None:
            return ""
        val = row[col]
        return "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val).strip()

    jobs = []
    for i, (_, row) in enumerate(df.iterrows()):
        jobs.append({
            "id": i,
            "title": get(row, "title"),
            "category": get(row, "category"),
            "company": get(row, "company"),
            "location": get(row, "location"),
            "skills": _parse_skills(row[cols["skills"]]) if cols["skills"] else [],
            "description": get(row, "description"),
        })
    return jobs


def load_csv(path: str, limit: int | None = None) -> list[dict]:
    return normalise(pd.read_csv(path), limit=limit)


def load_hf(name: str = DEFAULT_HF_DATASET, limit: int | None = 5000) -> list[dict]:
    from datasets import load_dataset  # imported lazily: only needed for this source

    df = load_dataset(name, split="train").to_pandas()
    return normalise(df, limit=limit)


def job_text(job: dict, max_desc_chars: int = 1500) -> str:
    """The text we embed for a job: title first, then the fields that say what the work is."""
    parts = [job["title"]]
    if job.get("company"):
        parts.append(f"Company: {job['company']}")
    if job.get("location"):
        parts.append(f"Location: {job['location']}")
    if job.get("skills"):
        parts.append("Skills: " + ", ".join(job["skills"]))
    if job.get("description"):
        parts.append(job["description"][:max_desc_chars])
    return ". ".join(parts)
