# Contracts

The data each component sends to another. Every component develops against these, so the
team can work in parallel against mock data.

## How contracts are defined

**The Pydantic models in `j26_contracts/` are the source of truth.**
The JSON Schema files in `schemas/` are generated from them, for documentation and for any
non-Python tool. Never edit `schemas/*.schema.json` by hand.

| Contract | Models | Generated schemas | Mock examples | Owner |
|---|---|---|---|---|
| Retrieval (C1/C4 → C3 → C1/C4) | `j26_contracts/retrieval.py` | `schemas/retrieval_request.schema.json`, `schemas/retrieval_output.schema.json` | `mock_data/retrieval_output_examples/` | C3 |

Other components can add their contracts the same way (e.g. `j26_contracts/ocr.py`,
`j26_contracts/assurance.py`) and register them in `j26_contracts/export.py`.

## Use the contracts in your component

Install once, inside your component's virtual environment:

```bash
pip install -e ../../contracts        # path from backend/<your-component>/
```

Then validate data you receive:

```python
from pydantic import ValidationError
from j26_contracts.retrieval import RetrievalOutput

try:
    result = RetrievalOutput.model_validate(response.json())
except ValidationError as e:
    ...  # e.errors() says exactly which field is wrong and why

result.results[0].values.amount.amount  # typed access, editor autocomplete
```

and build data you send:

```python
from datetime import date
from j26_contracts.retrieval import RetrievalRequest, to_contract_json

req = RetrievalRequest(
    schema_version="1.0.0",
    request_id="abc-1",
    query_text="personal relief",
    as_of_date=date.today(),
)
httpx.post(C3_URL + "/retrieve", json=to_contract_json(req))
```

## Changing a contract

1. Edit the models in `j26_contracts/`.
2. Regenerate the schemas: `python -m j26_contracts.export` (from this folder).
3. Update the mock examples if needed, and add an entry to `CONTRACT_CHANGELOG.md`.
4. Open a PR — changes to `contracts/` need approval from all four members.

`python -m j26_contracts.export --check` fails if the schema files are out of date;
the owner's CI runs it.
