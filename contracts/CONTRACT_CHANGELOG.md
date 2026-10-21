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

## pipeline_trace — v0.1.0 PROPOSAL

- **Date:** 2026-10-15
- **Author:** Fernando W I C S (C1)
- **Status:** Proposal for review. Not agreed yet; becomes 1.0.0 once C4 approves.
- **Change:** Added `j26_contracts/trace.py` (`PipelineTrace`), the generated `schemas/pipeline_trace.schema.json`, and three examples in `mock_data/pipeline_trace_examples/`: an injected wrong-band error on attempt 1, the corrected attempt 2, and a low-OCR-confidence trace.
- **Reason:** C4 needs everything C1 computed for one attempt so it can check every figure and citation without calling C1 back. It also unblocks the `req-0001` REGENERATE → PASS mock flow.
- **Key decisions (for review):**
  - **One trace per attempt.** `attempt` is 1 or 2. Attempt 2 must carry `regeneration` (the failed step ids and where C1 restarted).
  - **Every step can be recomputed.** Each step has named `inputs`, an `operation` string and a `result`, so C4 can recompute it.
  - **Citations stay inside the evidence.** `citation_refs` are C3 `rule_id`s, and every one must appear in `rules_used`. C4 re-fetches rules by id (`RetrievalRequest.rule_ids`).
  - **Claims cover every step.** `claims[]` links each statement in the explanation to a step and its citations, and every step must have at least one claim.
  - **Money is exact.** It is `Decimal` in Python and a plain JSON number on the wire, with at most 2 decimals. Whole amounts stay whole (96000). Open question: would C4 rather have strings?
  - **Confidence is broken down.** `confidence` keeps every term of C1's v1 formula, and `llm_calls` records each model id, prompt version, temperature and redacted output.
  - **No personal data.** The trace carries amounts, rule ids and generated text only.
  - **Why the template was shown.** `explanation.rewrite_status` (required) is `not_requested`, `failed`, `rejected` or `accepted`. `failed` means the LLM call itself raised (e.g. quota), so the skeleton text was shown but nothing was rejected; `rejected` means the rewrite failed C1's post-check (`used_fallback` true). The flags must agree with it. C4 can use it for an LLM failure rate in drift monitoring.
- **Amended (2026-10-17, still v0.1.0, no version bump while a proposal):** added `explanation.rewrite_status`, the matching schema, and `"rewrite_status": "not_requested"` in the three examples.
- **Review needed from:** De Zoysa K R D P (C4, consumer). For awareness: Elvitigala C S (C3, rule ids and citations) and Senevirathne R S N N (C2, `inputs[].source` and `document_id`).

## ocr_output — v0.2 (additive)

- **Date:** 2026-10-21
- **Author:** Senevirathne R S N N (C2)
- **Change:** `document_type` adds `quotation`. New field names: `document_number`, `due_date`, `seller_reg_no`, `subtotal`, `discount`, `amount_paid`, `balance_due`, `previous_balance`. Three new optional lists: `line_items` (description exactly as printed, quantity, unit, unit price, tax rate, amount, whether the amount includes tax), `tax_components` (VAT, SSCL, SGST ... with rate and amount) and `attributes` (printed key-value facts such as `System Size: 5 kW`). `schema_version` may be `"0.1"` or `"0.2"`. Mock: `mock_data/ocr_output_examples/invoice_solar_mock_01.json` (synthetic).
- **Reason:** Supervisor asked for tax deductions/reliefs (e.g. a solar panel installation). C2 now reads invoices, receipts and quotations so C1/C3 can decide relief eligibility from the line items; C2 itself decides nothing about relief.
- **Compatibility:** Payslips and bank statements still return `schema_version: "0.1"` with exactly the same fields, so C1's current check (`schema_version == "0.1"`, payslip only) keeps working. Invoices, receipts and quotations return `"0.2"`. A `quotation` is a price offer, not proof of spending.
- **Review needed from:** Fernando W I C S (C1), Elvitigala C S (C3), De Zoysa K R D P (C4).
