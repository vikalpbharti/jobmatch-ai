"""Vector index over job postings, with semantic search and resume-to-job matching.

Vectors live in a NumPy matrix (fast enough for tens of thousands of jobs). The roadmap
swaps this for PostgreSQL + pgvector without changing the JobSearch interface.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .data import job_text
from .embed import Embedder


class JobSearch:
    def __init__(self, jobs: list[dict], vectors: np.ndarray, embedder: Embedder):
        if len(jobs) != len(vectors):
            raise ValueError("jobs and vectors must have the same length")
        self.jobs = jobs
        self.vectors = vectors
        self.embedder = embedder
        # Every skill seen in the index; used to pull skills out of a resume.
        self.skill_vocab = sorted({s for j in jobs for s in j["skills"]}, key=len, reverse=True)

    # --- build / save / load ----------------------------------------------
    @classmethod
    def build(cls, jobs: list[dict], embedder: Embedder, show_progress: bool = True) -> "JobSearch":
        texts = [job_text(j) for j in jobs]
        embedder.fit(texts)
        return cls(jobs, embedder.encode(texts, show_progress=show_progress), embedder)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "jobs.jsonl", "w", encoding="utf-8") as f:
            for job in self.jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        meta = {"backend": self.embedder.backend, "model": self.embedder.model_name, "count": len(self.jobs)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        self.embedder.save(d)

    @classmethod
    def load(cls, directory: str | Path) -> "JobSearch":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        with open(d / "jobs.jsonl", encoding="utf-8") as f:
            jobs = [json.loads(line) for line in f]
        embedder = Embedder.load(d, meta["backend"], meta["model"])
        return cls(jobs, np.load(d / "vectors.npy"), embedder)

    # --- search -------------------------------------------------------------
    def _filter_mask(self, location: str | None) -> np.ndarray:
        mask = np.ones(len(self.jobs), dtype=bool)
        if location:
            loc = location.lower()
            mask &= np.array([loc in j["location"].lower() for j in self.jobs])
        return mask

    def search(self, query: str, k: int = 10, location: str | None = None) -> list[dict]:
        """Rank jobs by cosine similarity between the query and each posting."""
        qvec = self.embedder.encode([query])[0]
        scores = self.vectors @ qvec
        scores = np.where(self._filter_mask(location), scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [self._result(i, float(scores[i])) for i in top if np.isfinite(scores[i])]

    def extract_skills(self, text: str) -> list[str]:
        """Skills from the index vocabulary that appear in the text as whole words."""
        lowered = text.lower()
        found = []
        for skill in self.skill_vocab:
            if re.search(r"(?<![\w+#])" + re.escape(skill) + r"(?![\w+#])", lowered):
                found.append(skill)
        return sorted(found)

    def match_resume(self, resume_text: str, k: int = 10, location: str | None = None,
                     skill_weight: float = 0.3, candidates: int = 100) -> dict:
        """Two-stage match: retrieve by embedding similarity, then re-rank by skill overlap.

        final = (1 - skill_weight) * semantic_similarity + skill_weight * share_of_job_skills_the_candidate_has
        """
        resume_skills = set(self.extract_skills(resume_text))
        retrieved = self.search(resume_text, k=candidates, location=location)
        for r in retrieved:
            job_skills = set(r["skills"])
            overlap = len(job_skills & resume_skills) / len(job_skills) if job_skills else 0.0
            r["skill_overlap"] = round(overlap, 3)
            r["matched_skills"] = sorted(job_skills & resume_skills)
            r["missing_skills"] = sorted(job_skills - resume_skills)
            r["final_score"] = round((1 - skill_weight) * r["score"] + skill_weight * overlap, 4)
        retrieved.sort(key=lambda r: r["final_score"], reverse=True)
        return {"resume_skills": sorted(resume_skills), "matches": retrieved[:k]}

    def _result(self, i: int, score: float) -> dict:
        return {**self.jobs[i], "score": round(score, 4)}
