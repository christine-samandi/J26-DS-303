# Retrieval contract — mock examples (C3)

Example requests and responses for `contracts/schemas/retrieval_request.schema.json` and
`retrieval_output.schema.json`. Use them to build and test C1 and C4 against C3 before the real
retrieval API exists. Every file validates against its schema.

> ⚠️ **The amounts, dates, section numbers and amendment documents in these files are
> placeholders.** They show the *shape* of the data only. Do not use them in tax calculations.
> Verified values will come from the real knowledge graph (C3 Phase 1 onward).
> Anything marked `TBC` or `[MOCK ...]` is deliberately unfilled.

| File | Shows |
|---|---|
| `request_01_current_personal_relief.json` → `01_current_personal_relief.json` | Normal query for current law: one active rule |
| `request_02_point_in_time_with_history.json` → `02_point_in_time_with_history.json` | Past `as_of_date` with `include_history`: the rule in force then is `replaced` today but `in_force_on_as_of_date: true`; later versions appear with `in_force_on_as_of_date: false` |
| (no request file) → `03_tax_slabs_low_confidence.json` | Slab table in `values.bands`, low confidence, `LOW_CONFIDENCE` + `AMBIGUOUS_QUERY` warnings |
| (no request file) → `04_no_match.json` | Empty `results` with `NO_MATCH` — callers must handle this |
| `request_03_c4_recheck_by_rule_id.json` | C4 re-fetching a cited rule by `rule_id` |

## Rules for consumers

- **Which rule applies?** Use the result with `validity.in_force_on_as_of_date == true`, not
  `status == "active"`. `status` is about today; `in_force_on_as_of_date` is about the date you asked.
- **Numbers** come from `values` only. Never parse amounts out of `text` or `evidence.matched_text`.
- **Citations** for the user come from `source.citation`.
- **Confidence**: use `confidence.level` for decisions; `confidence.components` is there for C4's
  risk scoring and drift monitoring.
- Always check `warnings`, even when `results` is not empty.
