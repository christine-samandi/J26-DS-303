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

Money values in the OCR files are strings so they load as exact `Decimal`s.
