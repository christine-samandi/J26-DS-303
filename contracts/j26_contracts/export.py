"""Generate contracts/schemas/*.schema.json from the Pydantic models.

Run after ANY change to the models, and commit the regenerated files:
    python -m j26_contracts.export

`--check` only reports whether the files are up to date (used by tests).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel

from j26_contracts import retrieval

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"
REPO_URL = "https://github.com/christine-samandi/J26-DS-303"

# file name -> (model, description)
SCHEMAS: dict[str, tuple[type[BaseModel], str]] = {
    "retrieval_request.schema.json": (
        retrieval.RetrievalRequest,
        "Query sent to Component 3 (GraphRAG regulatory retrieval) by C1 or C4. Owner: C3.",
    ),
    "retrieval_output.schema.json": (
        retrieval.RetrievalOutput,
        "Response from Component 3, consumed by C1 (calculation, citations) and C4 (verification)."
        " Owner: C3.",
    ),
}

HEADER = "GENERATED from contracts/j26_contracts by `python -m j26_contracts.export`. Do not edit."


def build(model: type[BaseModel], description: str, file_name: str) -> dict:
    schema = model.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{REPO_URL}/contracts/schemas/{file_name}",
        "$comment": HEADER,
        **schema,
        "description": description,
    }


def render(file_name: str) -> str:
    model, description = SCHEMAS[file_name]
    return json.dumps(build(model, description, file_name), indent=2, ensure_ascii=False) + "\n"


def stale_files() -> list[str]:
    return [
        name
        for name in SCHEMAS
        if not (SCHEMAS_DIR / name).is_file()
        or (SCHEMAS_DIR / name).read_text(encoding="utf-8") != render(name)
    ]


def main(argv: list[str]) -> int:
    if "--check" in argv:
        stale = stale_files()
        for name in stale:
            print(f"OUT OF DATE: {name}  (run: python -m j26_contracts.export)")
        return 1 if stale else 0
    for name in SCHEMAS:
        (SCHEMAS_DIR / name).write_text(render(name), encoding="utf-8", newline="\n")
        print(f"wrote {SCHEMAS_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
