"""C3 retrieval API.

Run locally:
    uvicorn retrieval_api.app:app --reload --port 8003

Then open http://localhost:8003/docs to try it.

Requests and responses are the shared Pydantic contract models (contracts/j26_contracts),
so FastAPI checks every
request automatically and /docs shows every field. Each response is also checked against the
shared JSON Schema before it is sent.

Right now C3_MODE=mock serves the contract examples. The real GraphRAG backend replaces
`mock_backend.retrieve` in Phase 4; the endpoint and contract stay the same.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from c3_common.config import get_settings
from c3_common.contracts import validate_output
from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json
from retrieval_api import mock_backend

app = FastAPI(
    title="C3 — GraphRAG Regulatory Retrieval",
    version="0.1.0",
    description="Versioned, evidence-traceable retrieval of Sri Lankan income tax rules. "
    "Request/response follow contracts/schemas/retrieval_*.schema.json.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "component": "c3-graphrag", "mode": get_settings().mode}


@app.post("/retrieve", response_model=RetrievalOutput, response_model_exclude_unset=True)
def retrieve(request: RetrievalRequest) -> RetrievalOutput:
    # An invalid request never reaches this line: FastAPI + Pydantic reply 422 with the errors.
    mode = get_settings().mode
    if mode != "mock":
        raise HTTPException(status_code=501, detail=f"C3_MODE={mode} is not implemented yet")
    response = mock_backend.retrieve(request)

    validate_output(to_contract_json(response))  # double-check against the shared JSON Schema
    return response
