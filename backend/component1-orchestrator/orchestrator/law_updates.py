"""Law-change notifications from C3, and whether they affect a calculation already made.

C3 records a notification when a newly added Act changes rule versions (GET /notifications,
see component3-graphrag/c3_common/notifications.py). C1 uses them two ways:
- a "Law updates" box listing the latest changes;
- a personal check of the last calculation: it is AFFECTED when it used a rule version that
  the change ended or replaced before the calculation's date, or when a new rule (e.g. a new
  relief) is in force on that date and was not used. The user is asked to recalculate.

Standard library only. Any failure raises ComponentUnavailable; the UI shows it and carries on.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime

from orchestrator.clients import C3_DEFAULT_URL, ComponentUnavailable, _http

ENDED = {"ended", "replaced", "changed", "removed"}


def fetch_notifications(
    base_url: str = C3_DEFAULT_URL,
    since: datetime | None = None,
    audience: str = "users",
    timeout_s: float = 5.0,
) -> list[dict]:
    """C3's notifications, newest first."""
    params = {"audience": audience}
    if since is not None:
        params["since"] = since.isoformat()
    url = f"{base_url.rstrip('/')}/notifications?{urllib.parse.urlencode(params)}"
    raw = _http("C3", urllib.request.Request(url), timeout_s)
    try:
        return list(json.loads(raw)["notifications"])
    except (ValueError, KeyError, TypeError) as e:
        raise ComponentUnavailable(f"C3's notifications could not be read: {e}") from e


def _day(text: str | None) -> date | None:
    return date.fromisoformat(text) if text else None


def affects(change: dict, used_rule_ids: set[str], as_of: date) -> bool:
    """True when this change matters for a calculation made for `as_of` with these rules."""
    kind, rid = change.get("change"), change.get("rule_id")
    starts, ends = _day(change.get("valid_from")), _day(change.get("valid_to"))
    if kind in ENDED:
        if rid not in used_rule_ids:
            return False
        return (
            kind == "removed"
            or (ends is not None and as_of > ends)
            or (starts is not None and as_of < starts)
        )
    if kind == "added":
        return (starts or date.min) <= as_of <= (ends or date.max) and rid not in used_rule_ids
    return False


@dataclass(frozen=True)
class Impact:
    notification_title: str
    created_at: str
    summary: str
    rule_id: str
    change: str


def impacts(notifications: list[dict], used_rule_ids: set[str], as_of: date) -> list[Impact]:
    """Every change, in these notifications, that matters for a calculation on `as_of`."""
    out = []
    for n in notifications:
        for c in n.get("changes", []):
            if affects(c, used_rule_ids, as_of):
                out.append(Impact(n.get("title", ""), n.get("created_at", ""),
                                  c.get("summary", ""), c.get("rule_id", ""),
                                  c.get("change", "")))  # fmt: skip
    return out
