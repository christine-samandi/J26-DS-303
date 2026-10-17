"""Phase 2: lightweight (non-LLM) tax entity extraction with spaCy rules and regex.

chunking.py   parsed Acts -> retrieval chunks (one per subsection / schedule paragraph)
patterns.py   the rules as data: regex for amounts, rates, dates, bands, legal references;
              a term lexicon for reliefs, deductions, exemptions, income types, taxpayers
extract.py    runs the rules with spaCy and writes data/extracted/<document_id>.json
annotation.py builds the blind hand-annotation workbook (fixed sample of 30 chunks)
evaluate.py   precision / recall / F1 per label against those annotations
"""
