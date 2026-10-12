"""The corpus registry: which official documents C3 collects, and where they come from.

The list lives in `corpus_sources.json` (data, not code) so adding a document never needs a
code change. Every entry is validated here before anything is downloaded.
"""

from __future__ import annotations

import json
from datetime import date
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

REGISTRY_PATH = Path(__file__).with_name("corpus_sources.json")

DocumentId = Annotated[str, StringConstraints(pattern=r"^DOC-[A-Z0-9_-]+$")]
FileName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+\.pdf$")]


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: DocumentId
    title: str = Field(min_length=1)
    document_type: Literal["act", "amendment_act", "gazette", "circular", "notice", "guide"]
    act_number: str | None = None
    certified_date: date
    operative_from: date | None = None  # None: sections take effect on different dates
    url: str
    filename: FileName
    relevance: str = ""

    @field_validator("url")
    @classmethod
    def _https_only(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("source URLs must use https")
        return value


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    documents: list[SourceDocument] = Field(min_length=1)

    @field_validator("documents")
    @classmethod
    def _unique(cls, docs: list[SourceDocument]) -> list[SourceDocument]:
        for attr in ("document_id", "filename"):
            values = [getattr(d, attr) for d in docs]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {attr} in corpus registry")
        return docs

    def get(self, document_id: str) -> SourceDocument:
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        raise KeyError(f"{document_id} is not in the corpus registry")


@cache
def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    return Registry.model_validate(json.loads(path.read_text(encoding="utf-8")))
