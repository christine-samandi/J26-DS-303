"""C3 retrieval API.

Run locally:
    uvicorn retrieval_api.app:app --reload --port 8003

Then open http://localhost:8003/docs to try it.

Right now C3_MODE=mock serves the contract examples. The real GraphRAG backend replaces
`mock_backend.retrieve` in Phase 4; the endpoint and contract stay the same.
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException

from c3_common.config import get_settings
from c3_common.contracts import contract_errors, validate_output
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


@app.post("/retrieve")
def retrieve(request: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
    errors = contract_errors(request, "retrieval_request.schema.json")
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    mode = get_settings().mode
    if mode != "mock":
        raise HTTPException(status_code=501, detail=f"C3_MODE={mode} is not implemented yet")
    response = mock_backend.retrieve(request)

    validate_output(response)  # C3 must never send a response that breaks the contract
    return response
