"""C3 retrieval API.

Run locally:
    uvicorn retrieval_api.app:app --reload --port 8003

Then open http://localhost:8003/docs to try it.

Requests and responses are the shared Pydantic contract models (contracts/j26_contracts),
so FastAPI checks every
request automatically and /docs shows every field. Each response is also checked against the
shared JSON Schema before it is sent.

C3_MODE (in .env) picks the backend; the endpoint and contract are the same for both:
- mock:     the contract examples (placeholder values), no database needed;
- graphrag: the real knowledge graph in Neo4j (retrieval_api.graph_backend).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from c3_common.config import get_settings
from c3_common.contracts import validate_output
from j26_contracts.retrieval import RetrievalOutput, RetrievalRequest, to_contract_json
from retrieval_api import graph_backend, mock_backend

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
    if mode == "mock":
        response = mock_backend.retrieve(request)
    elif mode == "graphrag":
        try:
            response = graph_backend.retrieve(request)
        except Exception as exc:  # the graph database is down or not built yet
            if type(exc).__module__.startswith("neo4j"):
                raise HTTPException(
                    status_code=503,
                    detail=f"Knowledge graph unavailable ({type(exc).__name__}). Start Neo4j "
                    "(docker compose up -d) and run python -m knowledge_graph.build.",
                ) from exc
            raise
    else:
        raise HTTPException(status_code=501, detail=f"C3_MODE={mode} is not supported")

    validate_output(to_contract_json(response))  # double-check against the shared JSON Schema
    return response


@app.post("/reload")
def reload_graph() -> dict[str, str | int]:
    """Forget the cached graph so the next request reads it again (after an update)."""
    graph_backend.graph.cache_clear()
    if get_settings().mode != "graphrag":
        return {"status": "ok", "mode": get_settings().mode}
    data = graph_backend.graph()
    return {"status": "reloaded", "rule_versions": len(data.versions), "chunks": len(data.chunks)}
