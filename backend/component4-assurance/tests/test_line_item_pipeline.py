
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_line_item_total
from consistency_checker.checker import ErrorCode
from decision_engine.engine import AssuranceAction


def test_matching_line_items_pass():
    result = assure_line_item_total(
        reported_total=Decimal("150000"),
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
    )

    assert result.decision.action == AssuranceAction.PASS
    assert result.decision.failed_checks == 0
    assert result.findings[0].passed is True


def test_mismatching_line_items_escalate():
    result = assure_line_item_total(
        reported_total=Decimal("180000"),
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.decision.failed_checks == 1
    assert result.findings[0].error_code == (
        ErrorCode.LINE_ITEM_SUM_MISMATCH
    )
    assert result.findings[0].expected_value == Decimal("150000")


def test_empty_line_items_rejected():
    with pytest.raises(ValueError, match="At least one"):
        assure_line_item_total(
            reported_total=Decimal("0"),
            line_items=[],
        )


def test_invalid_line_item_rejected():
    with pytest.raises(ValueError, match="finite"):
        assure_line_item_total(
            reported_total=Decimal("100"),
            line_items=[Decimal("NaN")],
        )


def test_critical_failure_escalates_even_with_retries_available():
    result = assure_line_item_total(
        reported_total=Decimal("180000"),
        line_items=[
            Decimal("50000"),
            Decimal("70000"),
            Decimal("30000"),
        ],
        regeneration_attempts=0,
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
