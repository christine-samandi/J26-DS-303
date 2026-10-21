# Component 3 — GraphRAG Knowledge Base & Regulatory Retrieval
Owner: Elvitigala C S (IT23280724)

Builds a versioned knowledge graph of Sri Lankan individual income tax rules (Inland Revenue
Act No. 24 of 2017 and its amendments) using lightweight, non-LLM extraction, and serves
evidence-traceable, confidence-scored, point-in-time retrieval to C1 and C4.

**Contract:** requests and responses follow
`contracts/schemas/retrieval_request.schema.json` and `retrieval_output.schema.json`.
See `contracts/mock_data/retrieval_output_examples/README.md` for examples and consumer rules.

**Validation (Pydantic):** the contract is defined as Pydantic models in the shared package
`contracts/j26_contracts/retrieval.py` (the JSON Schema files are generated from them). C3 uses
the same models as C1 and C4: FastAPI rejects bad requests with them (422 with the exact field
and reason) and documents every field at `/docs`. Every response is validated by the models,
then double-checked against the generated JSON Schema.

The models also enforce rules JSON Schema cannot express:
- `in_force_on_as_of_date` must match the rule's `valid_from`/`valid_to` and the queried date
- `valid_to` cannot be before `valid_from`
- results are ranked 1..n in order
- with `include_history: false`, only rules in force on the date are returned

## Status

| Part | Folder | Status |
|---|---|---|
| Contract + mock API | `retrieval_api/`, `contracts/j26_contracts/` | ✅ v1.0.0 (mock mode, Pydantic-validated) |
| Data acquisition + PDF parsing | `data_collection/` | ✅ Phase 1 (full Act series: 2017 Act + 6 amendments; parser tuned on the real PDFs: 379/379 sections, all 6 schedules) |
| Ground truth | `ground_truth/` | ✅ rule versions (personal relief V1–V4, tax bands B1–B4, all verified; the six other Fifth Schedule reliefs in `relief_versions.csv`, to be verified); entity answer key (10 core chunks, 246 entities) |
| Entity extraction (spaCy + regex) | `entity_recognition/` | ✅ Phase 2: 1,324 chunks, 10 entity types; blind P 0.97 / R 1.00, adjudicated P = R = 0.996 (10 core chunks) |
| Knowledge graph + versioning (Neo4j) | `knowledge_graph/` | ✅ Phase 3: rule versions rebuilt from the Acts (14/14 exact: personal relief, tax bands and, through the general relief engine, Fifth Schedule para 2(b)–(g)); graph builder |
| Vector store (ChromaDB) | `vector_store/` | ⏳ next (TF-IDF term coverage used for now) |
| Indexing cost vs LLM extraction | `evaluation/indexing_cost.py` | ✅ whole pipeline timed on all 7 Acts (~3 s, 0 tokens, deterministic); LLM baseline (Gemini) scored on the same gold chunks |
| GraphRAG retrieval | `retrieval_api/` | ✅ Phase 4: text + graph neighbourhood + date filter, contract output, confidence (`C3_MODE=graphrag`) |

## How to run standalone

Requires Python 3.11+ and Docker Desktop (for Neo4j).

```bash
cd backend/component3-graphrag
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -e ..\..\contracts        # shared contract models (macOS/Linux: ../../contracts)
pip install -e ".[dev,api,graph,ingest]"

copy .env.example .env            # macOS/Linux: cp .env.example .env  — then set NEO4J_PASSWORD

pytest                            # contract, model, API, data collection and parser tests
ruff check .                      # lint
```

**Phase 1: collect and parse the corpus**
```bash
python -m data_collection.download     # 7 official Acts -> raw_regulatory_docs/ + manifest.csv
python -m data_collection.parse        # -> data/processed/<document_id>.json (sections, schedules)
python -m data_collection.inspect_pdf raw_regulatory_docs/IRA_AMEND_02_2025_E.pdf   # layout check
```
If a site blocks the script, download the PDF in a browser and run
`python -m data_collection.download --register <DOC_ID> <path-to-pdf>`.

**Phase 2: extract tax entities (no LLM)**
```bash
pip install -e ".[nlp]"                # spaCy (no language model download needed)
python -m entity_recognition.extract   # data/processed -> data/extracted/<document_id>.json
python -m entity_recognition.extract --text "Rs. 1,800,000 from April 1, 2025"   # quick check
```

**Phase 2: measure extraction (precision / recall / F1)**
```bash
python -m entity_recognition.annotation  # -> ground_truth/entity_annotations.xlsx (30 chunks)
# annotate by hand in Excel (see the Guide sheet), type "done" on each finished chunk
python -m entity_recognition.evaluate    # scores -> data/evaluation/
```

**Start Neo4j and check the connection**
```bash
docker compose up -d
python -m knowledge_graph.connection   # → "Connected to Neo4j Kernel 5.x ..."
```
Neo4j Browser: http://localhost:7474 (user `neo4j`, password from `.env`).

**Phase 3: rule versions and the knowledge graph**
```bash
python -m knowledge_graph.versions     # rebuild rule histories from the Acts, score vs ground_truth/
pip install -e ".[graph]"
python -m knowledge_graph.build        # (re)build the whole graph in Neo4j (--dry-run: no database)
```
Try in Neo4j Browser:
```cypher
MATCH (v:RuleVersion)-[:VERSION_OF]->(:Rule {rule_key:'personal_relief'})
OPTIONAL MATCH (v)-[:DEFINED_IN]->(c:Chunk)
RETURN v, c
```

**The general relief engine** (`knowledge_graph/reliefs.py`). Every relief in Fifth Schedule
paragraph 2, not only personal relief, is rebuilt by one method: the 2017 wording is split into
its subparagraphs, every amendment is applied in order (subparagraph replaced, words
substituted, new subparagraphs added), and the value (fixed amount, cap or percentage) and
period are read from the latest wording. A new subparagraph with no date of its own starts on
its section's operation date (the Act's Table 'A'). Only names, conditions and question words
are written by hand (`RELIEFS`); values and dates come from the Acts.

| Relief | Para | Value | In force |
|---|---|---|---|
| Employment relief | 2(b) | up to Rs. 700,000 | 2018-04-01 .. 2019-12-31 (withdrawn) |
| Rent relief | 2(c) | 25% of rental income | 2018-04-01 .. now |
| Senior citizen interest relief | 2(d) | up to Rs. 1,500,000 | 2018-04-01 .. 2019-12-31 (withdrawn) |
| Foreign-currency service income | 2(e) | up to Rs. 15,000,000 | 2018-04-01 .. 2019-12-31 (withdrawn) |
| Expenditure relief (health, education, housing loan, pension, shares) | 2(f) | up to Rs. 1,200,000 | 2020-01-01 .. 2022-12-31 (withdrawn) |
| Solar panel relief | 2(g) | up to Rs. 600,000 | 2021-04-01 .. now |

A withdrawn relief has status `outdated`; asked about on a date it is not in force, the API
returns no result and the warning `NO_RULE_IN_FORCE_ON_DATE` (with `include_history`, the
version and its dates). Not yet covered: qualifying payments (para 1: donations) and
exemptions (Third Schedule).

**Adding a newly published Act** (no full rebuild; see `DEMO.md`)
```bash
python -m knowledge_graph.build --until 2025-01-01      # the graph as it was known before 2025
python -m knowledge_graph.update DOC-IRA-AMEND-02-2025  # new Act -> new versions + change report
```

**Law monitor, automatic updates and notifications** (keeping the knowledge base current)
```bash
python -m data_collection.monitor              # check the IRD Acts page now; add what is safe
python -m data_collection.monitor --every      # keep checking, once a month
python -m data_collection.monitor --list       # what was found: added / held (with reason)
python -m data_collection.monitor --retry URL  # run the checks again on a held document
```
1. The monitor finds an income tax PDF that is not in the corpus.
2. `knowledge_graph.auto_update` adds it **by itself if all seven safety checks pass**: it is a
   PDF; its cover page says "Inland Revenue (Amendment) Act, No. N of YYYY" (it is registered
   from that); it is not already in the corpus; it parses without warnings; no rule version
   disappears; every hand-verified version is still rebuilt exactly (only an open-ended version
   may get an end date); all values are possible (amounts > 0, rates 0-100%, bands in order).
   If any check fails, nothing changes (registry, files and graph are put back) and the
   document is HELD with the reason in a maintainer notification.
3. Added: the graph is updated, the API reloaded, and the change is recorded as a
   `law_change` notification (GET /notifications) in plain words with the Act's sentence.

Future improvement: deliver each change to the in-app inbox of the users whose saved
calculations it affects, once the app has user accounts.

**Phase 4: ask the graph (Neo4j running, graph built)**
```bash
python -m retrieval_api.graph_backend "What personal relief can I claim?" --date 2024-06-01
python -m retrieval_api.graph_backend "tax slabs" --date 2021-06-01 --history
python -m retrieval_api.graph_backend "Can I claim relief for solar panels?" --date 2025-06-01
python -m retrieval_api.graph_backend "What reliefs can I claim?" --date 2021-06-01
```
Set `C3_MODE=graphrag` in `.env` to make the API below answer from the graph instead of the
mock examples (restart the API after rebuilding the graph).

**Phase 5: evaluate retrieval (GraphRAG vs flat RAG)**
```bash
python -m evaluation.retrieval                  # questions in ground_truth/retrieval_questions.csv
python -m evaluation.retrieval --questions ground_truth/retrieval_questions_heldout.csv
```
Scores -> `data/evaluation/retrieval_scores.json` and `retrieval_per_question.csv`.

**Indexing cost: this pipeline vs LLM extraction (proposal 3.4)**
```bash
python -m evaluation.indexing_cost                     # Part A: time every indexing stage
pip install -e ".[llm]"                                # once, for Part B
python -m evaluation.indexing_cost --llm gemini        # + Part B (GEMINI_API_KEY in .env)
python -m evaluation.indexing_cost --llm gemini --price-in 0.30 --price-out 2.50
```
Part A times parse -> chunk -> extract -> versions -> graph rows on the whole corpus and checks
that extraction gives the same output twice. Part B sends each gold chunk (the same 10 chunks,
246 entities) to the LLM with the annotation guide, scores it with the same scorer
(`entity_recognition.evaluate`), runs it twice for consistency, and projects tokens, time and
cost to the whole corpus. Prices are not built in: take them from the provider's price page.
Limits: the LLM is measured on the gold chunks and scaled up; results depend on the model and
prompt (both recorded in `data/evaluation/indexing_cost.json`).

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
| POST | `/reload` | Re-read the graph after `knowledge_graph.update` |
| GET | `/notifications` | Law-change notifications, newest first (`?audience=users\|admin&since=...`) |

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
`vector` (ChromaDB + sentence-transformers), `llm` (Gemini, cost baseline only), `api` (FastAPI), `dev` (tests/lint), or `all`.
