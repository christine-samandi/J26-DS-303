"""Settings for Component 3, read from environment variables (or a local .env file).

Copy `.env.example` to `.env` and fill in your values. Never commit `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

COMPONENT_DIR = Path(__file__).resolve().parents[1]


def _find_contracts_dir() -> Path:
    override = os.getenv("C3_CONTRACTS_DIR")
    if override:
        return Path(override).resolve()
    for parent in COMPONENT_DIR.parents:
        candidate = parent / "contracts"
        if (candidate / "schemas").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the repo's contracts/ folder; set C3_CONTRACTS_DIR.")


@dataclass(frozen=True)
class Settings:
    mode: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    chroma_dir: Path
    raw_docs_dir: Path
    contracts_dir: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(COMPONENT_DIR / ".env")
    return Settings(
        mode=os.getenv("C3_MODE", "mock"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        chroma_dir=Path(os.getenv("C3_CHROMA_DIR", COMPONENT_DIR / ".chroma")),
        raw_docs_dir=Path(os.getenv("C3_RAW_DOCS_DIR", COMPONENT_DIR / "raw_regulatory_docs")),
        contracts_dir=_find_contracts_dir(),
    )
