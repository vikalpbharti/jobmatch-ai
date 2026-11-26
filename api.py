"""FastAPI service over the job index.

  uvicorn api:app --reload        then open http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from jobmatch.search import JobSearch

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["index"] = JobSearch.load(INDEX_DIR)  # load vectors + model once at startup
    yield
    state.clear()


app = FastAPI(title="JobMatch AI", version="0.1.0", lifespan=lifespan)


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    k: int = Field(10, ge=1, le=50)
    location: str | None = None


@app.get("/health")
def health():
    index = state["index"]
    return {"status": "ok", "jobs": len(index.jobs), "model": index.embedder.model_name}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50), location: str | None = None):
    return {"query": q, "results": state["index"].search(q, k=k, location=location)}


@app.post("/match")
def match(req: MatchRequest):
    out = state["index"].match_resume(req.resume_text, k=req.k, location=req.location)
    if not out["matches"]:
        raise HTTPException(404, "No jobs matched the filters")
    return out
