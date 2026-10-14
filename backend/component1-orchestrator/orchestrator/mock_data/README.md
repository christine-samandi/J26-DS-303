# C1 orchestrator mock data

Used by the `Mock…Client` adapters until the real C2/C3/C4 APIs are live.

> ⚠️ Placeholders. Section numbers are `TBC` and documents are marked `[C1 MOCK]`.
> Shapes only; real rules and citations come from C3.

| File | Stands in for | Contract |
|---|---|---|
| `retrieval_ya2025_26.json` | C3's answer for YA 2025/26 (relief + bands, as of 1 April 2025) | `j26_contracts.retrieval.RetrievalOutput` v1.0.0. Values match the seed rules; C3's own mocks are dated 2026 and use the old six-band table. |
| `ocr/payslip_clean.json` | C2 reading a clean annual payslip (GS-001 figures) | **Provisional** C1-internal shape (`orchestrator.models.OcrResult`) until C2's `ocr_output` contract exists |
| `ocr/payslip_noisy_low_confidence.json` | Same payslip, low OCR confidence (should escalate) | Provisional, as above |

Money values in the OCR files are strings so they load as exact `Decimal`s.
