
from decimal import Decimal

from consistency_checker.checker import check_total_consistency


def test_matching_totals():
    result = check_total_consistency(
        reported_total=Decimal("150000.00"),
        calculated_total=Decimal("150000.00"),
    )

    assert result.passed is True


def test_mismatching_totals():
    result = check_total_consistency(
        reported_total=Decimal("180000.00"),
        calculated_total=Decimal("150000.00"),
    )

    assert result.passed is False
    assert "Total mismatch" in result.message
