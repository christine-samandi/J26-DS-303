"""The demo's explanation badge follows rewrite_status, not the Gemini toggle."""

import pytest

from explanation_agent.badge import explanation_badge
from explanation_agent.models import Explanation


def explanation(status: str) -> Explanation:
    return Explanation(
        lines=("x",),
        claims=(),
        rewrite_status=status,
        used_fallback=status == "rejected",
    )


@pytest.mark.parametrize(
    "status,label,color",
    [
        ("failed", "Template · Gemini unavailable", "gray"),
        ("rejected", "Template · Gemini rewrite rejected by post-check", "red"),
        ("accepted", "Gemini rewrite · verified by post-check", "green"),
        ("not_requested", "Template (deterministic)", "blue"),
    ],
)
def test_badge_follows_rewrite_status(status, label, color):
    assert explanation_badge(explanation(status)) == (label, color)


def test_demo_rewriter_name_is_only_wording():
    assert explanation_badge(explanation("rejected"), "rewrite") == (
        "Template · rewrite rejected by post-check",
        "red",
    )
