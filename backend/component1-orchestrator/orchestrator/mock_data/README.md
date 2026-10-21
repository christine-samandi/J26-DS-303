# C1 orchestrator mock data

Used by the `Mock…Client` adapters until the real C2/C3/C4 APIs are live.

> ⚠️ Stand-ins for C3, marked `[C1 MOCK]`. Provisions are copied from C3's verified ground truth
> (`backend/component3-graphrag/ground_truth/`: V3/B3 for 2024/25, V4/B4 for 2025/26); in a live run
> they come from C3. `MockRetrievalClient` picks the file for the YA of the query's `as_of_date`.

| File | Stands in for | Contract |
|---|---|---|
| `retrieval_ya2024_25.json` | C3's answer for YA 2024/25 (relief + bands, as of 1 April 2024) | `j26_contracts.retrieval.RetrievalOutput` v1.0.0. Values match the 2024/25 seed rules (Act No. 45 of 2022). |
| `retrieval_ya2025_26.json` | C3's answer for YA 2025/26 (relief + bands, as of 1 April 2025) | `j26_contracts.retrieval.RetrievalOutput` v1.0.0. Values match the seed rules; C3's own mocks are dated 2026 and use the old six-band table. |
| `ocr/payslip_clean.json` | C2 reading a clean annual payslip (GS-001 figures) | **Provisional** C1-internal shape (`orchestrator.models.OcrResult`) until C2's `ocr_output` contract exists |
| `ocr/payslip_noisy_low_confidence.json` | Same payslip, low OCR confidence (should escalate) | Provisional, as above |
| `retrieval_reliefs_ya2024_25.json`, `retrieval_reliefs_ya2025_26.json` | C3's answer to relief queries (solar panel + rental relief, as of 1 April; plus the expenditure relief, which ended 2022-12-31, returned only with `include_history`) | `RetrievalOutput` v1.0.0, copied from C3's graphrag answer on dev `c590e0b` (expenditure relief: C3's `retrieve` on the graph rows of all Acts, query "Asiri Hospital medical expenses", `include_history: true`, 2026-10-21). `MockRetrievalClient` keeps the reliefs the query names. |
| `relief_keywords.json` | C3's words that name each relief (`knowledge_graph/reliefs.py`) | Mock only: picks which mock relief a query names |
| `ocr/solar_invoice_2025.json` | C2 reading a solar panel invoice (Rs. 800,000, 15 Aug 2025) | Provisional `OcrResult` with `issuer` / `document_date` / `total_amount` (pre-fill only; the user confirms) |
| `ocr/receipt_unreadable_issuer.json` | A receipt whose issuer C2 could not read (C1 asks what it was for) | Provisional, as above |
| `ocr/medical_receipt_2025.json` | A medical receipt (Rs. 45,000, 10 Sep 2025): the expenditure relief it names was withdrawn (last valid 2022-12-31) | Provisional, as above |
| `ocr/bank_statement_2025.json` | A bank statement (listed as not analysed yet) | Provisional, as above |

Money values in the OCR files are strings so they load as exact `Decimal`s.
