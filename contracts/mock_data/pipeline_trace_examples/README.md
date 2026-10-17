# Pipeline trace — mock examples (C1)

> **PROPOSAL v0.1.0, not yet agreed.** Example traces for `contracts/schemas/pipeline_trace.schema.json`
> (Pydantic: `j26_contracts.trace.PipelineTrace`). Built from the real C1 engine with GS-001
> figures (income 3,000,000, APIT 50,000), so every number is consistent with its step.

> ⚠️ Rule ids, citations and section numbers are placeholders (`[MOCK]`, `TBC`). Shape only.

| File | Shows | C4 should answer |
|---|---|---|
| `trace_injected_error_wrong_band.json` | `req-0001` attempt 1. **Injected error:** band 2 taxed at 12% instead of the cited rule's 18%. The error carries through consistently (band tax 24,000, total 84,000, payable 34,000), so only checking against the rule catches it. | `REGENERATE`, failing `S3_BAND_2` (and the steps that depend on it) |
| `trace_correct.json` | `req-0001` attempt 2 after that REGENERATE: correct figures (total 96,000, payable 46,000), with `regeneration` saying what was flagged | `PASS` |
| `trace_low_confidence_ocr.json` | `req-0002` attempt 1: correct figures, but C2 read the payslip with low confidence, so C1 confidence is 0.55 | `PASS` on the arithmetic; C1 escalates it anyway (below 0.7) |

## Rules for consumers

- `steps[].result` must equal `steps[].operation` applied to `steps[].inputs`. Rounding is `round_half_up(x, 0.01)`.
- Every `citation_refs` id is in `rules_used`. Re-fetch a rule from C3 with `RetrievalRequest.rule_ids` to check a value against its source.
- Every step has at least one claim, and every claim names a real step.
- Read money with `Decimal` (`json.loads(s, parse_float=Decimal)` or the Pydantic model). It is exact to the cent.
- A trace never contains personal data.
