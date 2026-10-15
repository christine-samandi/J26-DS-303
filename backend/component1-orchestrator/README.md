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
