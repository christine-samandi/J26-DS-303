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
