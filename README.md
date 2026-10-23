# JobRadar

Job matching platform: ingest real jobs from Greenhouse / Lever / Ashby, then match them to a
resume, measuring every improvement against a labeled eval set.

**Status:** The backend ingestion/API and frozen evaluation workflow are implemented. A private single-user local web MVP now supports live job browsing, profile setup, and job tracking. Embedding ranking is used when stored vectors exist; hybrid ranking and public multi-user authentication are not implemented.

## Quick start

```bash
docker compose up -d db                    # Postgres + pgvector on :5434

cd backend
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

alembic upgrade head                       # create tables
pytest                                     # all tests (DB tests skip if Postgres is down)
python -m app.ingest                       # fetch every company in ../companies.yaml
uvicorn app.main:app --reload              # http://localhost:8000/docs
```

Set `USER_AGENT` (see `.env.example`) to something with a real contact before you ingest.

## Private local application

The private single-user interface browses active jobs from PostgreSQL, supports search/location/remote filters, profile editing and PDF resume extraction, saved/applied tracking, and embedding-similarity ordering when stored profile and job vectors are available. Without an embedding, the UI clearly falls back to newest-first ordering. Resume uploads are parsed in memory and are not stored. If Gemini profile extraction is configured, resume text is sent to Gemini for extraction; review the provider terms before uploading sensitive information. Saving a profile automatically regenerates its embedding when skills or target titles change. If Gemini is unconfigured or quota-limited, the profile is still saved and the UI reports that its embedding is pending.

For local development, start PostgreSQL, apply migrations, and start the API using the Quick start above. In a second terminal run:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies API requests to `http://127.0.0.1:8000`. To run the local Docker staging stack instead, copy the root `.env.example` to `backend/.env` if needed, then run `docker compose up --build -d`; open `http://localhost:8080`. The compose ports bind to localhost only. The API container applies Alembic migrations when it starts. A scheduler container ingests configured companies daily at 06:00 Asia/Kolkata; it reads the root `companies.yaml` file read-only and restarts automatically with Docker. View logs with `docker compose logs -f scheduler`; pause scheduled ingestion with `docker compose stop scheduler`. Newly ingested or materially changed jobs have stale/missing vectors cleared by ingestion; refresh job embeddings separately with `cd backend && python -m app.embeddings --entity jobs --batch-size 1` when Gemini quota permits. Profile saves refresh the profile vector automatically. This compose setup uses local development database credentials and is not a public production deployment. Do not expose it to the internet without authentication, managed secrets, production database credentials, and a security review.

## Layout

```
backend/app/fetchers/   base.py (NormalizedJob + Fetcher), http.py (retries), greenhouse.py
backend/app/normalize.py   HTML -> text, seniority guess, dedupe hash
backend/app/ingest.py      upsert jobs, mark missing jobs inactive
backend/alembic/           migrations
eval/                      labeled pairs + run_eval.py (Phase 3)
frontend/                  React app (Phase 5)
companies.yaml             which boards to ingest
```

## Phase 3: frozen eval set

The labeling/evaluation flow uses a versioned snapshot of profiles and active jobs. Only the
snapshot export command reads PostgreSQL; labeling and evaluation run entirely from files.
Candidate jobs are sampled from skill-keyword matches, title matches, and a seeded random
pool. Location, remote preference, years of experience, and seniority do not affect candidate
selection or role-relevance metrics. The label screen hides those logistics fields, along with
scores, ATS sources, pool-selection reasons, and extracted job skills.

For a more representative eval set, export multiple profiles and a larger pool per profile.
If `--profile` is omitted, all saved profiles are included. Version 3 snapshots retain the
role-based source category assigned to each candidate for dataset auditing; the labeling
display still hides those categories. Use a new filename so an existing snapshot and its
labels stay intact.

From the repository root (after installing `backend/requirements-dev.txt`):

```bash
python eval/label_jobs.py export --pool-size 100 --seed 42 --snapshot eval/snapshot-v3.json
python eval/label_jobs.py label --snapshot eval/snapshot-v3.json --labels eval/labels-v3.json
python eval/run_eval.py --snapshot eval/snapshot-v3.json --labels eval/labels-v3.json
```

Sampling uses balanced per-source quotas across skill-keyword matches, title matches, and
random jobs, then fills any remaining slots deterministically.
Export refuses to overwrite an existing snapshot unless `--force` is passed. The snapshot
includes frozen profile data, the candidate pools, and only the job records referenced by
those pools; it does not duplicate the entire live jobs table. Labels are stored separately.
Frozen snapshots contain profile attributes and source job descriptions; keep snapshot and
label JSON files local/private (they are ignored by Git), and do not publish them without
permission to share both the profile data and source text.
Before labeling, the tool prints this guideline: judge the fit of the job title,
responsibilities, and skills against the profile's skills and titles. Use `2` for a strong
role fit, `1` for a plausible but imperfect fit, and `0` for a poor role/skills fit. Ignore
location, remote arrangement, seniority, and years of experience; those are separate
eligibility filters, not relevance labels. Do not infer missing details or use hidden scores,
sources, extracted skills, or candidate-selection reasons.

Evaluation requires every candidate to be labeled and reports macro-averaged per-profile
precision@5 and nDCG@10 for the keyword baseline (skill matches in job title/description and
profile-title matches in job title). It also reports the label distribution, source counts,
and location/remote and experience-filter eligibility counts separately. Eligibility filters
do not change the ranking or relevance metrics. Structured minimum-experience requirements
are not currently present in job records, so experience eligibility is reported as unknown.
Neither the labeler nor evaluator queries the live database.

### Job eligibility filters

The backend job-list endpoint accepts optional `remote` and `location` filters
(`GET /api/jobs?remote=true&location=Seattle`). These filters narrow the job candidates
returned by the API before any client-side ranking; they do not modify evaluation labels or
relevance scores. A job with no recorded location remains in results when a location filter
is used, because its location eligibility is unknown rather than known-incompatible.

There is intentionally no years-of-experience filter yet. Profiles have an experience value,
but jobs do not have structured minimum-experience requirements. Job seniority is not used as
a substitute, and missing experience requirements are treated as unknown in the eval summary.

The default `eval/snapshot.json` v1 snapshot and its labels were created with
location-influenced sampling, so treat them as historical only. The role-only 100-job
`eval/snapshot-v3.json` snapshot and its existing labels are the current baseline set.

## Embedding ranking experiment

The embedding provider is Google's Gemini `gemini-embedding-001`. Its recommended 1536
output dimensions match both `jobs.embedding` and `profiles.embedding` (`VECTOR(1536)`) in
the initial schema. Requests explicitly ask for 1536 dimensions, and provider output is
checked for exact length and finite numeric values before it can be stored. If Gemini returns
its documented 3072-dimensional default anyway, the embedding client takes the supported
1536-value Matryoshka prefix and normalizes it (required for `gemini-embedding-001`) before
storage. Profiles use
Gemini's `RETRIEVAL_QUERY` task type and jobs use `RETRIEVAL_DOCUMENT`. Set `GEMINI_API_KEY`
in your environment or in `backend/.env` before making embedding requests. API availability,
limits, and pricing depend on Google's current account and model terms.

Apply the migration that records which model produced each stored vector, then backfill
existing active jobs and profiles. Run these commands from `backend/`:

```bash
alembic upgrade head
python -m app.embeddings --entity all --batch-size 32
```

The backfill stores role-only text: job title and description for jobs, and profile titles
and skills for profiles. Jobs with no vector or a different embedding model are selected;
profiles with no role skills/titles are skipped. Completed batches are committed, so rerunning
resumes safely. Use `--entity jobs` or `--entity profiles` to run one type, `--limit N` to
cap records per type, and `--force` to regenerate vectors for the selected model. Set
`GEMINI_EMBEDDING_MODEL` to override the model only if it supports the requested 1536
dimensions. Rows marked with a previous model (including OpenAI's) are detected as stale and
re-embedded with the configured Gemini model.
When job title/description changes, ingestion clears its vector and model marker. Saving a
profile clears its vector and marker. A model change is detected by the stored model marker
and causes a refresh on the next backfill.

To measure embeddings against the frozen role-relevance labels, generate an independent
artifact from the exact snapshot and evaluate it. These commands run from the repository
root and do not read or modify the live database or frozen snapshot:

```bash
python eval/embed_snapshot.py --snapshot eval/snapshot-v3.json --output eval/embeddings-v3.json
python eval/run_eval.py --snapshot eval/snapshot-v3.json --labels eval/labels-v3.json --embeddings eval/embeddings-v3.json
```

The artifact records the snapshot ID, model, dimension, and candidate vectors; the evaluator
refuses artifacts from a different snapshot or with incomplete/wrong-sized vectors. It
reports the cosine-similarity embedding ranker alongside the keyword baseline on the same
labeled candidates. This is an offline ranking experiment; it does not yet change `/api/jobs`
ordering or implement a production recommendations endpoint. Keep generated vectors
local/private, as they are derived from profile and job text.

## Gemini LLM reranking experiment

The optional Gemini experiment reranks the top keyword-matched jobs for each profile using
Google's `generateContent` API with JSON-constrained output. It sees only the profile's
titles/skills and each shortlisted job's title/description. It does not receive human labels,
extracted job skills, or location/remote/experience fields. The reranker must return each
shortlisted job ID exactly once; invalid or incomplete output is rejected.

Set `GEMINI_API_KEY` in `backend/.env` or the process environment. The default model is
`gemini-2.5-flash`; optionally set `GEMINI_MODEL` to another model available to your AI Studio
account. API limits and free-tier availability are controlled by Google and may vary by model
and account. Review Google's data-use terms before sending profile and job text.

Generate the snapshot-bound artifact from the repository root, then evaluate it against the
same frozen labels:

```bash
python eval/rerank_gemini.py --snapshot eval/snapshot-v3.json --output eval/gemini-rerank-v3.json --top-k 20
python eval/run_eval.py --snapshot eval/snapshot-v3.json --labels eval/labels-v3.json --gemini-rerank eval/gemini-rerank-v3.json
```

The first command sends only the top 20 keyword-ranked jobs per profile (change with
`--top-k`). The evaluator puts Gemini's ordering first, then appends all non-shortlisted jobs
in their original keyword order, so precision@5 and nDCG@10 still cover the complete frozen
candidate pool. The output artifact contains the snapshot ID, model, shortlist sizes, and job
ID order, but no API key or generated job/profile text. This is an offline experiment; it
does not change live job ordering.

The same `GEMINI_API_KEY` is used for Gemini embeddings and reranking. Embedding generation
uses `gemini-embedding-001` by default and can be changed independently with
`GEMINI_EMBEDDING_MODEL`.

## Eval results

| Version | precision@5 | nDCG@10 |
|---|---:|---:|
| Role-relevance keyword baseline (v3; 100 candidates, one profile) | 1.00 | 0.712 |
| Embeddings (v3; run experiment) | — | — |
| + hybrid | — | — |
| Gemini LLM reranker (v3; run experiment) | — | — |
