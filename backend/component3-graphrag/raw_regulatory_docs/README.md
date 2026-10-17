# Raw regulatory documents

The original official PDFs, exactly as downloaded. Never edit them: parsed output goes to
`data/processed/`.

- Which documents, and from where: `data_collection/corpus_sources.json`
- What was collected, when, and its SHA-256 fingerprint: `manifest.csv` (written by
  `python -m data_collection.download`)

Every rule in the knowledge graph traces back to a file here through its `document_id`.
These are public government publications, committed so that results can be reproduced.
