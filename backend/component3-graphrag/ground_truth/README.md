# Ground truth: rule versions (PP1 slice)

Hand-verified history of the two rules in the PP1 slice. This is what C3's versioning is
measured against (objective 2.2.3: versioning accuracy) and what the point-in-time demo must
reproduce.

| File | Rule | Versions |
|---|---|---|
| `personal_relief_versions.csv` | Personal relief, individuals (Fifth Schedule para 2(a)) | V1–V4 |
| `individual_tax_bands.csv` | Progressive income tax rates, individuals (First Schedule para 1) | B1–B4 |

Compiled 2026-10-11 from the official Acts (documents.gov.lk, ird.gov.lk). Every row names its
source document (same IDs as `data_collection/corpus_sources.json`) and provision.

## How the dates were set

- **Dates are when the rule applied, not when the Act was published.** Act No. 10 of 2021 was
  certified on 13 May 2021 but applies from 1 January 2020. `valid_from` is 2020-01-01.
- **Changes in the middle of a year of assessment.** The 2020 and 2023 changes split a year of
  assessment (1 April – 31 March). Act No. 45 of 2022 states the split explicitly: for the year
  starting 1 April 2022, relief is Rs. 2,250,000 for the first nine months (3,000,000 × 9/12)
  and Rs. 300,000 for the last three (1,200,000 × 3/12), with bands scaled the same way
  (Rs. 125,000 per quarter = Rs. 500,000 a year). The CSVs record the **annual** values and the
  date the new annual value took effect (1 January 2023).
- `valid_to` is the last day in force (inclusive); empty means still in force.
- Amounts and band limits are in rupees; `upper_lkr` empty means no upper limit.

## Verification

All rows are verified against the official Act PDFs (`verified` = `yes`). B1 was checked by
hand on 2026-10-12 against the First Schedule, paragraph 1(1) of the 2017 Act (PDF page 212,
printed page 200): bands of Rs. 600,000 at 4, 8, 12, 16, 20 and 24 %, with fixed amounts
Rs. 24,000 / 72,000 / 144,000 / 240,000 / 360,000, which `tests/test_ground_truth.py`
recomputes from the bands.

Acts checked and excluded (no change to these two rules): No. 4 of 2023, No. 14 of 2023,
No. 11 of 2026. Act No. 11 of 2026 was checked on its parsed text: its section 36 changes
First Schedule para 1(2)(a) (gains on investment assets, 10% to 15%), not the progressive
bands in para 1(1); its sections 5 and 40 change qualifying payments (Fifth Schedule para 1),
not personal relief (para 2).
