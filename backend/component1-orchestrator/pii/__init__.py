"""C1 PII redact-restore (CLAUDE.md section 10): personal identifiers are swapped for
placeholders before text goes to an external LLM and put back in its reply. Regex only, plus
names the caller already knows (e.g. an OCR employee-name field); no NLP model.
"""

from pii.redact import Redaction, redact, restore

__all__ = ["Redaction", "redact", "restore"]
