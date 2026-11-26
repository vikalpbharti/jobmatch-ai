# JobMatch AI

Semantic job search and resume-to-job matching. Job postings are embedded with
sentence-transformers, so a search for *"build ETL pipelines on the cloud"* finds Data Engineer
roles even when the posting never uses those words. A resume is matched in two stages:
embedding retrieval, then re-ranking by how many of each job's required skills the candidate has.

> Status: phase 1 (embeddings + semantic search + evaluation) is working. Next up: an LLM agent
> that calls this search as a tool and returns structured recommendations. See the roadmap.

## How it works

```
job postings ──► job_text() ──► sentence-transformers (all-MiniLM-L6-v2) ──► normalised vectors
                                                                               │
query / resume ──► same encoder ──► cosine similarity (dot product) ◄──────────┘
                                          │
                       top-100 candidates ▼
                  re-rank: 0.7 × semantic score + 0.3 × share of job skills the resume has
                                          │
                                          ▼
                       results + matched / missing skills per job
```

- **`jobmatch/data.py`** loads postings from Hugging Face (`lukebarousse/data_jobs`) or any CSV, and samples evenly across job categories
- **`jobmatch/embed.py`** wraps the embedding model, with a TF-IDF keyword backend kept as the baseline
- **`jobmatch/search.py`** holds the vector index: semantic search, location filter, resume skill extraction and re-ranking
- **`api.py`** is the FastAPI service (`/search`, `/match`, `/match-file` for PDF uploads, `/health`)
- **`static/index.html`** is the web UI: search by description, or upload a resume to see matching jobs with the skills you have and the ones you're missing
- **`eval/evaluate.py`** measures precision@10 and MRR, embeddings vs the keyword baseline

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_index.py --limit 5000              # downloads data + model on first run
python scripts/search.py "build ETL pipelines on the cloud"
python scripts/search.py "machine learning" --location India -k 5
python scripts/search.py --resume my_resume.txt          # plain-text resume

python eval/evaluate.py                                  # embeddings vs keyword baseline
uvicorn api:app --reload                                 # web UI at http://127.0.0.1:8000, API docs at /docs
```

On Windows you can also double-click `start.bat`: it starts the server and opens the web UI.

Use your own data with `--source csv --path data/jobs.csv`. The CSV needs a `title` column;
`company`, `location`, `skills`, `description` and `category` are used when present.

## Evaluation

Each test query in `eval/queries.jsonl` describes the *work* of a role without naming it
(e.g. *"stream processing with Kafka and Spark"* → Data Engineer). Relevance comes from the
dataset's own category labels. Results on 5,000 postings:

| System | Precision@10 | MRR |
|---|---|---|
| TF-IDF keyword baseline | 0.545 | **0.795** |
| all-MiniLM-L6-v2 embeddings | **0.582** | 0.776 |

What this shows:
- Embeddings return more relevant jobs in the top 10 overall (+3.7 points precision@10) and win 7 of
  11 queries, most clearly on technical work descriptions (*"train neural networks with PyTorch and build
  MLOps pipelines"*: 0.90 vs 0.40; *"deploy and serve deep learning models"*: 0.90 vs 0.60).
- Keyword search wins 3 queries (one ties), mostly business-language ones where words like *dashboards*, *stakeholders* and
  *requirements* map directly onto job titles (*"turn business data into dashboards"*: 1.00 vs 0.30), and
  puts its first hit slightly higher (MRR).
- This dataset has titles and skill tags but no job descriptions, which limits what embeddings can
  pick up. Next steps: hybrid search (combine both scores) and indexing full descriptions.

## Roadmap

- [x] Embeddings, semantic search, location filter
- [x] Resume matching with skill-overlap re-ranking
- [x] Retrieval evaluation against a keyword baseline
- [ ] Move vectors to PostgreSQL + pgvector
- [ ] LLM agent with tools (`search_jobs`, `get_job_details`, `match_resume`) returning structured JSON
- [ ] RAG: ground the agent's match explanations in the retrieved postings
- [ ] Docker image + public deploy
