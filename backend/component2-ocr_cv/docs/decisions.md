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
| 2026-10-16 | Preprocessing order | denoise → resize → perspective → deskew → CLAHE → adaptive threshold + speck removal | Resize before denoise | Upscaling noisy pixels turns noise into blobs that look like ink; denoising first at native resolution removed this (ink share 24% → 2.5% on a noisy synthetic photo) |
| 2026-10-16 | Denoise strength | Set from measured noise (Immerkaer estimate, h = 1.1·σ, 3–30) | Fixed strength | Clean scans are barely touched; dark, noisy photos are cleaned harder |
| 2026-10-16 | Skew estimation | Projection-profile search (±20°, 1° then 0.1°) on a locally thresholded ink mask | Otsu-based mask; Hough lines | Otsu marked shadowed regions as ink and gave 21° instead of −7°; projection profile is simple and explainable |
| 2026-10-16 | Perspective correction | Largest convex quad with ≥60% continuous edge support; skipped otherwise | Always warp the largest contour | Text blocks were mistaken for page edges; edge-support check rejects them and leaves scans/cropped photos untouched |
| 2026-10-16 | Rotation fill | Median background brightness | Pure white; replicate edge | White leaves a seam; replicate creates streaks read as dashes |
| 2026-10-16 | bbox traceability | Every geometric step stored as a 3×3 homography; boxes mapped back with its inverse | Re-running OCR on the original | Keeps contract rule "bbox in original-image pixels" while OCR runs on the cleaned image |
| 2026-10-16 | Ablation support | `preprocess(img, steps=...)` with named, switchable steps | Fixed pipeline | Needed for experiment E1 (contribution of each step) |
| 2026-10-16 | Dataset split | Per document, stratified by type, every 3rd document → test (≥1 test doc per type); existing splits never move | Random per-image split | Photos of the same document on both sides would leak answers into the test score |
| 2026-10-16 | Ground-truth format | One JSON per document, same field names as the output schema, validated by docs/ground_truth_schema.json | Spreadsheet of values | Direct comparison with predictions; schema catches typos such as "150,000" typed as text |
| 2026-10-16 | Ground-truth checking | Automatic payslip arithmetic check (gross − deductions = net) on hand-typed values | Trusting manual entry | A wrong answer key silently lowers every accuracy number |
| 2026-10-16 | File naming | `<DOC_ID>_<capture>.<ext>` (e.g. PS_001_photo_dark.jpg) | Free names + manual manifest | Manifest is generated, so capture condition and document id can never disagree |
