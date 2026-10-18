# Component 1 — Agentic Orchestrator, Tax Calculation Agent, Explanation Agent
Owner: Fernando (IT23180420)

## How to run standalone
Test against contracts/mock_data/ for Comp2 and Comp3 outputs before real integration.

## Orchestrator (LangGraph)

`orchestrator/` runs one request through the pipeline:

```
extract -> normalise -> retrieve -> calculate -> explain -> score -> assure -> deliver
                                                                       |
                                    REGENERATE (once) -> rerun from the stage C4 names
```

- `stages.py`: each stage is a plain function. Only `graph.py` imports LangGraph (pinned), and a test enforces that.
- `clients.py`: one adapter interface each for C2, C3 and C4, with mock versions that read `orchestrator/mock_data/`.
- C3 uses the real `j26_contracts.retrieval` contract. C2 (`ocr_output`), C4 (`assurance_verdict`) and `pipeline_trace` use **provisional** C1-internal models in `models.py` until those contracts are agreed.
- `explain` builds template text from the engine's steps. If `Deps.explainer_llm` is set, the LLM rewords it, and `explanation_agent`'s post-check rejects any changed number or citation, falling back to the template (`used_fallback`, `hallucination_flag`). Every LLM call goes into the trace's `llm_calls`, rejected output included.
- `llm.py`: the Gemini client for the explainer (`GeminiTextLLM`), off by default. `explainer_llm_from_env()` turns it on with `C1_EXPLAINER_LLM=gemini` + `GEMINI_API_KEY` (optional `GEMINI_MODEL`, `C1_EXPLAINER_TIMEOUT_MS`, default 20000). Needs the `llm` extra; the only orchestrator module that imports an LLM SDK.
- C3's rules are cross-checked against the seed rules. Any disagreement is shown in the result, lowers confidence, and sends the result for review.

```python
from decimal import Decimal
from orchestrator.clients import MockAssuranceClient, MockOcrClient, MockRetrievalClient
from orchestrator.graph import run
from orchestrator.models import TaxRequest
from orchestrator.stages import Deps

deps = Deps(MockOcrClient(), MockRetrievalClient(), MockAssuranceClient(["REGENERATE"]))
result = run(TaxRequest(request_id="req-0001", year_of_assessment="2025/26",
                        documents=("payslip_clean",)), deps)
result.status, result.attempt, result.calc.total_tax   # ('delivered', 2, Decimal('96000.00'))
```

## Live C3 (GraphRAG) instead of the mock

`HttpRetrievalClient` calls C3's real service (`POST /retrieve`, the frozen retrieval contract,
validated both ways with `j26_contracts`). If C3 is down or answers outside the contract, the
request ends as `failed` with the reason, never with made-up rules.

1. Start C3 (in `backend/component3-graphrag`, Neo4j running, `.env` has `C3_MODE=graphrag`):
   `uvicorn retrieval_api.app:app --port 8003`
2. In the demo sidebar, turn on **Live C3 GraphRAG** and pick a year of assessment. Or set it in
   `.env`: `C1_RETRIEVAL=http` (and `C3_API_URL`, default `http://localhost:8003`).

The **Law used** table shows the rules C3 returned: value, validity dates and the Act + provision.
Only YA 2025/26 has seed rules, so other years show "not cross-checked".

```python
from orchestrator.clients import retrieval_client_from_env
deps = Deps(MockOcrClient(), retrieval_client_from_env({"C1_RETRIEVAL": "http"}), MockAssuranceClient())
```
