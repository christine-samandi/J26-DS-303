"""Law-change log: what changed in the law C3 holds, and what the monitor did.

Two kinds, written by two commands:
- `new_document_found` (for the maintainer): data_collection.monitor found a new document on
  the IRD site and says what happened to it (added automatically, or held with the reason).
- `law_change` (for users): an Act was added and rule versions changed; each change says which
  rule, what it is now, and the Act's sentence that caused it.

They are kept in data/notifications/notifications.json (newest last) and served by the API at
GET /notifications. Future work: deliver each law_change to the users whose saved
calculations it affects (an in-app inbox, once the app has user accounts).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from c3_common.config import COMPONENT_DIR

DEFAULT_PATH = COMPONENT_DIR / "data" / "notifications" / "notifications.json"


@dataclass
class RuleChange:
    rule_id: str
    rule_key: str
    title: str
    change: str  # added | ended | replaced | changed | removed
    summary: str  # one plain sentence for a person
    valid_from: str | None = None
    valid_to: str | None = None
    evidence: str = ""  # the Act's sentence, or what replaced it


@dataclass
class Notification:
    id: str
    created_at: str
    kind: str  # law_change | new_document_found
    audience: str  # users | admin
    title: str
    message: str
    document_id: str | None = None
    url: str | None = None
    changes: list[RuleChange] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Notification:
        return cls(**{**d, "changes": [RuleChange(**c) for c in d.get("changes", [])]})


def _path() -> Path:
    return Path(os.getenv("C3_NOTIFICATIONS_FILE", DEFAULT_PATH))


def load(path: Path | None = None) -> list[Notification]:
    path = path or _path()
    if not path.is_file():
        return []
    return [Notification.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def new_id(existing: list[Notification], now: datetime) -> str:
    stem = f"N-{now:%Y%m%d-%H%M%S}"
    taken = {n.id for n in existing}
    n = 1
    while f"{stem}-{n}" in taken:
        n += 1
    return f"{stem}-{n}"


def add(notification: Notification, path: Path | None = None) -> Notification:
    """Keep a notification."""
    path = path or _path()
    items = load(path)
    items.append(notification)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(n) for n in items], indent=2) + "\n", encoding="utf-8")
    return notification


def make(kind: str, audience: str, title: str, message: str, **extra) -> Notification:
    now = datetime.now(UTC)
    return Notification(new_id(load(), now), now.isoformat(timespec="seconds"), kind, audience,
                        title, message, **extra)  # fmt: skip


def select(items: list[Notification], since: datetime | None = None,
           audience: str | None = None) -> list[Notification]:  # fmt: skip
    out = []
    for n in items:
        if audience and n.audience != audience:
            continue
        if since and datetime.fromisoformat(n.created_at) <= since:
            continue
        out.append(n)
    return out
