# eval/ — golden scenarios and evaluation harness

Shared by the whole team. A **golden scenario** is a tax case worked out by hand and frozen
as the correct answer. The engine, the pure-LLM baseline, the explanations and C4's
assurance checks are all measured against the same set.

If the engine disagrees with a verified scenario, the engine is wrong. A scenario only
changes if the hand calculation itself was wrong, and that change needs a second reviewer.

## Layout

```
eval/
  scenarios/GS-NNN_<slug>.json   one scenario per file
  run_engine.py                  engine calculation accuracy (target 100%)
  run_baseline_llm.py            pure-LLM baseline: the LLM does the arithmetic, same scoring
  results/                       run outputs, git-ignored
```

More metric scripts (`run_explanations.py`, `run_latency.py`, ...) will be added here.

## Running

From the repo root, with the C1 package installed (`pip install -e backend/component1-orchestrator`):

```bash
python eval/run_engine.py
```

Writes `results/engine_<UTC time>_<git sha>.csv` and `.md`. A `-dirty` suffix on the SHA
means there were uncommitted changes, so don't report numbers from a dirty run. Only
`verified` scenarios count towards accuracy. The script exits 1 if any of them fails.

### Pure-LLM baseline

```bash
pip install -e "backend/component1-orchestrator[llm]"
cp .env.example .env        # then set GEMINI_API_KEY
python eval/run_baseline_llm.py
```

The LLM gets the same seed rules as the engine, inside the prompt, and must return the four
totals as JSON. It is scored exactly like the engine, to the cent. Temperature 0, seed 0, one
retry on a malformed reply. The results record the model that was requested, the model
version that actually answered, and the prompt version. Raw replies are saved to
`results/baseline_llm_*.raw.jsonl`. Wrong answers are the result being measured, so the script
always exits 0.

## Scenario format

| Field | Meaning |
|---|---|
| `scenario_id` | `GS-NNN`, matches the file name prefix |
| `title`, `tags` | What the case covers (band reached, payable/refund, edge case, ...) |
| `year_of_assessment` | e.g. `"2025/26"`; picks the rule set |
| `input` | `employment_income`, `apit_paid` (LKR) |
| `expected` | `taxable_income`, `total_tax`, `tax_payable`, `refund`, and every `steps[]` in order |
| `expected.steps[].cites` | Rule **keys** the step must cite (`personal_relief`, `individual_income_tax_slabs`). Keys rather than rule ids, so the same scenario works with seed rules and with C3's rules |
| `hand_calculation` | The working, line by line, so a reviewer can redo it |
| `verification` | `status`: `unverified` or `verified`; `verified_by`: who checked it and when |

Money follows the contract rules: LKR, plain JSON numbers, no commas. Read files with
`json.load(f, parse_float=Decimal)` so no float reaches the engine.

## Rules

- Never put real taxpayer data here. Scenarios are made up.
- A scenario counts towards reported results only once `verification.status` is `verified`.
- Figures depend on the YA rules. YA 2025/26 uses Rs. 1,800,000 relief and bands of
  6 / 18 / 24 / 30 / 36% (1M, then 500k each), from the Inland Revenue (Amendment) Act
  No. 2 of 2025.
