"""The explanation badge the demo shows, from what actually happened to the rewrite.

Kept free of Streamlit so it can be unit-tested; app.py passes the result to st.badge.
"""

from __future__ import annotations

from explanation_agent.models import Explanation


def explanation_badge(exp: Explanation, rewriter: str = "Gemini rewrite") -> tuple[str, str]:
    """(label, colour) for `exp`. `rewriter` only names the LLM in the label."""
    match exp.rewrite_status:
        case "failed":  # nothing was produced, so nothing was rejected: neutral, not red
            return "Template · Gemini unavailable", "gray"
        case "rejected":
            return f"Template · {rewriter} rejected by post-check", "red"
        case "accepted":
            return f"{rewriter} · verified by post-check", "green"
        case _:
            return "Template (deterministic)", "blue"
