"""FastAPI service and web UI over the job index.

  uvicorn api:app --reload        then open http://127.0.0.1:8000  (API docs at /docs)
"""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from jobmatch.search import JobSearch

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["index"] = JobSearch.load(INDEX_DIR)  # load vectors once at startup
    state["index"].search("warm up the embedding model", k=1)  # load the model now, not on the first visitor's query
    yield
    state.clear()


app = FastAPI(title="JobMatch AI", version="0.2.0", lifespan=lifespan)


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=20)
    k: int = Field(10, ge=1, le=50)
    location: str | None = None


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    index = state["index"]
    return {"status": "ok", "jobs": len(index.jobs), "model": index.embedder.model_name}


@app.get("/search")
def search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50), location: str | None = None):
    return {"query": q, "results": state["index"].search(q, k=k, location=location or None)}


@app.post("/match")
def match(req: MatchRequest):
    return _match(req.resume_text, req.k, req.location)


@app.post("/match-file")
async def match_file(file: UploadFile = File(...), k: int = Form(10), location: str | None = Form(None)):
    """Upload a resume as PDF or plain text and get matching jobs."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 5 MB")
    text = _extract_text(file.filename or "", data)
    if len(text.strip()) < 20:
        raise HTTPException(422, "Couldn't read enough text from that file. Is it a scanned image?")
    return _match(text, max(1, min(k, 50)), location)


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf") or data[:5] == b"%PDF-":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            raise HTTPException(422, "Couldn't read that PDF")
    return data.decode("utf-8", errors="ignore")


def _match(text: str, k: int, location: str | None) -> dict:
    out = state["index"].match_resume(text, k=k, location=location or None)
    if not out["matches"]:
        raise HTTPException(404, "No jobs matched the filters")
    return out
