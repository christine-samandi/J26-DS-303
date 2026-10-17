
from decimal import Decimal

import pytest

from consistency_checker.checker import (
    ErrorCode,
    Severity,
    check_total_consistency,
)


def test_matching_totals():
    result = check_total_consistency(
        reported_total=Decimal("150000.00"),
        calculated_total=Decimal("150000.00"),
    )

    assert result.passed is True
    assert result.severity == Severity.INFO
    assert result.error_code == ErrorCode.NONE
    assert result.expected_value == Decimal("150000.00")
    assert result.actual_value == Decimal("150000.00")


def test_mismatching_totals():
    result = check_total_consistency(
        reported_total=Decimal("180000.00"),
        calculated_total=Decimal("150000.00"),
    )

    assert result.passed is False
    assert result.severity == Severity.CRITICAL
    assert result.error_code == ErrorCode.TOTAL_MISMATCH
    assert result.expected_value == Decimal("150000.00")
    assert result.actual_value == Decimal("180000.00")
    assert "Total mismatch" in result.message


def test_zero_totals():
    result = check_total_consistency(
        reported_total=Decimal("0"),
        calculated_total=Decimal("0"),
    )

    assert result.passed is True


def test_negative_totals():
    result = check_total_consistency(
        reported_total=Decimal("-100"),
        calculated_total=Decimal("-100"),
    )

    assert result.passed is True


@pytest.mark.parametrize(
    "invalid_value",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")],
)
def test_non_finite_reported_total(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        check_total_consistency(
            reported_total=invalid_value,
            calculated_total=Decimal("150000"),
        )


def test_non_finite_calculated_total():
    with pytest.raises(ValueError, match="finite"):
        check_total_consistency(
            reported_total=Decimal("150000"),
            calculated_total=Decimal("NaN"),
        )
