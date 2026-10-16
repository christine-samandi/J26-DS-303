# Component 2 decision log

One entry per decision. Format: date · decision · chosen · rejected · reason.

| Date | Decision | Chosen | Rejected | Reason |
|---|---|---|---|---|
| 2026-10-16 | Output contract | JSON + JSON Schema 2020-12 (`ocr_output_schema.json` v0.1) | XML, Protobuf | Native to FastAPI/Pydantic, readable by people and LLM agents, validatable in tests; Protobuf adds code generation with no benefit at this scale |
| 2026-10-16 | Schema enforcement | Pydantic models mirroring the schema | Manual checks | Invalid output fails before it reaches Components 1, 3, 4 |
| 2026-10-16 | bbox format | `[x, y, w, h]` in original-image pixels | Normalised 0–1, polygons | Traceability must point at the user's original photo; polygons logged as future extension |
| 2026-10-16 | Review threshold | confidence < 0.95 → `needs_review` | No flag | FR7 and main objective (95% threshold) |
| 2026-10-16 | Token format | One `Token` (text, confidence 0–1, bbox, line_id) for all engines | Engine-specific outputs | Extraction and benchmark code stay engine-independent |
| 2026-10-16 | Data storage | `data/` git-ignored, redacted before storage | Committing samples | NFR4; real documents hold personal financial data |
| 2026-10-16 | data/ folder creation | Scripts create `data/` and `outputs/` subfolders at runtime (`mkdir(parents=True, exist_ok=True)`) | Un-ignoring `.gitkeep` files in `data/` | Keeps `data/` fully git-ignored so nothing under it can be committed by accident; a fresh clone works without manual setup |
