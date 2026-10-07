# Component 3 — GraphRAG Knowledge Base & Regulatory Retrieval
Owner: Elvitigala C S (IT23280724)

Builds a versioned knowledge graph of Sri Lankan individual income tax rules (Inland Revenue
Act No. 24 of 2017 and its amendments) using lightweight, non-LLM extraction, and serves
evidence-traceable, confidence-scored, point-in-time retrieval to C1 and C4.

**Contract:** requests and responses follow
`contracts/schemas/retrieval_request.schema.json` and `retrieval_output.schema.json`.
See `contracts/mock_data/retrieval_output_examples/README.md` for examples and consumer rules.

## Status

| Part | Folder | Status |
|---|---|---|
| Contract + mock API | `retrieval_api/`, `contracts/` | ✅ v1.0.0 (mock mode) |
| Data acquisition | `data_collection/` | ⏳ Phase 1 |
| Entity extraction (spaCy + regex) | `entity_recognition/` | ⏳ Phase 2 |
| Knowledge graph + versioning (Neo4j) | `knowledge_graph/` | ⏳ Phase 3 (connection helper ready) |
| Vector store (ChromaDB) | `vector_store/` | ⏳ Phase 4 |
| GraphRAG retrieval | `retrieval_api/` | ⏳ Phase 4 |

## How to run standalone

Requires Python 3.11+ and Docker Desktop (for Neo4j).

```bash
cd backend/component3-graphrag
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,api,graph]"

copy .env.example .env            # macOS/Linux: cp .env.example .env  — then set NEO4J_PASSWORD

pytest                            # contract + API tests
ruff check .                      # lint
```

**Start Neo4j and check the connection**
```bash
docker compose up -d
python -m knowledge_graph.connection   # → "Connected to Neo4j Kernel 5.x ..."
```
Neo4j Browser: http://localhost:7474 (user `neo4j`, password from `.env`).

**Run the retrieval API**
```bash
uvicorn retrieval_api.app:app --reload --port 8003
```
Interactive docs: http://localhost:8003/docs

```bash
curl -X POST http://localhost:8003/retrieve -H "Content-Type: application/json" ^
  -d "{\"schema_version\":\"1.0.0\",\"request_id\":\"demo-1\",\"query_text\":\"personal relief\",\"as_of_date\":\"2021-06-01\",\"include_history\":true}"
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + current mode |
| POST | `/retrieve` | Retrieval (body = `RetrievalRequest`, response = `RetrievalOutput`) |

Invalid requests get `422` with the list of contract errors. Every response is validated
against the output schema before it is sent.

## Confidence levels (provisional — calibrated in Phase 5)

| Level | Score |
|---|---|
| high | ≥ 0.80 |
| medium | 0.50 – 0.79 |
| low | < 0.50 |

## Dependency groups

Install only what you need: `ingest` (scraping/PDF), `nlp` (spaCy), `graph` (Neo4j),
`vector` (ChromaDB + sentence-transformers), `api` (FastAPI), `dev` (tests/lint), or `all`.
