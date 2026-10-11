"""Phase 1: collect the official IRD documents and parse them into structured JSON.

python -m data_collection.download      # fetch the corpus, write raw_regulatory_docs/manifest.csv
python -m data_collection.parse         # PDFs -> data/processed/<document_id>.json
python -m data_collection.inspect_pdf <pdf>   # show how the layout was read (for tuning)
python -m data_collection.discover <url>      # list PDF links on a page (find new Acts)
"""
