
from decimal import Decimal

import pytest

from consistency_checker.checker import (
    ErrorCode,
    Severity,
    check_line_item_sum,
)


def test_matching_line_items():
    result = check_line_item_sum(
        reported_total=Decimal("150000.00"),
        line_items=[
            Decimal("50000.00"),
            Decimal("70000.00"),
            Decimal("30000.00"),
        ],
    )

    assert result.passed is True
    assert result.expected_value == Decimal("150000.00")
    assert result.error_code == ErrorCode.NONE


def test_mismatching_line_items():
    result = check_line_item_sum(
        reported_total=Decimal("180000.00"),
        line_items=[
            Decimal("50000.00"),
            Decimal("70000.00"),
            Decimal("30000.00"),
        ],
    )

    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.error_code == ErrorCode.LINE_ITEM_SUM_MISMATCH
    assert result.expected_value == Decimal("150000.00")
    assert result.actual_value == Decimal("180000.00")


def test_zero_line_items():
    result = check_line_item_sum(
        reported_total=Decimal("0"),
        line_items=[Decimal("0"), Decimal("0")],
    )

    assert result.passed is True


def test_negative_adjustment():
    result = check_line_item_sum(
        reported_total=Decimal("90000"),
        line_items=[
            Decimal("100000"),
            Decimal("-10000"),
        ],
    )

    assert result.passed is True


def test_empty_line_items_rejected():
    with pytest.raises(ValueError, match="At least one"):
        check_line_item_sum(
            reported_total=Decimal("0"),
            line_items=[],
        )


@pytest.mark.parametrize(
    "invalid_item",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
    ],
)
def test_invalid_line_item_rejected(invalid_item):
    with pytest.raises(ValueError, match="finite"):
        check_line_item_sum(
            reported_total=Decimal("100"),
            line_items=[invalid_item],
        )


def test_invalid_reported_total_rejected():
    with pytest.raises(ValueError, match="finite"):
        check_line_item_sum(
            reported_total=Decimal("NaN"),
            line_items=[Decimal("100")],
        )
