# Contract Changelog
Log every change to any schema in contracts/schemas/ here, with date, author, and reason.
Any schema change must be reviewed by all components that depend on it.

## retrieval_request / retrieval_output — v1.0.0

- **Date:** 2026-10-08
- **Author:** Elvitigala C S (C3)
- **Change:** Added `schemas/retrieval_request.schema.json` and `schemas/retrieval_output.schema.json` (v1.0.0), plus mock examples in `mock_data/retrieval_output_examples/`.
- **Reason:** First version of the C3 → C1/C4 contract, so C1 (Tax Retrieval Agent, citations) and C4 (citation, evidence and effective-date verification) can build against mocks while C3 is implemented.
- **Key decisions:**
  - Point-in-time queries via `as_of_date`; each result carries `validity` (`valid_from`, `valid_to`, `status`, `in_force_on_as_of_date`, supersession links).
  - Numeric values returned in structured `values` (amount, cap, rate, slab bands) for C1's deterministic engine.
  - `confidence` returns a score, a high/medium/low level, and a four-part breakdown for C4.
  - `evidence` returns verbatim matched text, chunk IDs and the graph path.
- **Review needed from:** Fernando W I C S (C1), De Zoysa K R D P (C4), Senevirathne R S N N (C2, for awareness).

## retrieval_request / retrieval_output — v1.0.0 defined as Pydantic models

- **Date:** 2026-10-08
- **Author:** Elvitigala C S (C3)
- **Change:** The retrieval contract is now defined as Pydantic models in `j26_contracts/retrieval.py` (installable package `j26-contracts`). `schemas/retrieval_*.schema.json` are generated from the models (`python -m j26_contracts.export`) instead of being hand-written. Version stays 1.0.0: same fields, same rules; optional fields may now also be sent as `null`.
- **Reason:** Supervisor recommended Pydantic for validating JSON structure. One definition instead of two, and every component can import the same models to validate what it sends and receives.
- **Extra checks (Pydantic only):** `in_force_on_as_of_date` must match the rule's dates and the queried date; `valid_to` not before `valid_from`; results ranked 1..n; with `include_history: false` only rules in force are returned.
- **Review needed from:** Fernando W I C S (C1), De Zoysa K R D P (C4), Senevirathne R S N N (C2, for awareness).
