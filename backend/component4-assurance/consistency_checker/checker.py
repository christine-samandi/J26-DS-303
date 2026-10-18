
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Sequence

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class ErrorCode(str, Enum):
    NONE = "NONE"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"
    LINE_ITEM_SUM_MISMATCH = "LINE_ITEM_SUM_MISMATCH"
    MISSING_CITATION = "MISSING_CITATION"
    UNKNOWN_SOURCE = "UNKNOWN_SOURCE"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"

@dataclass(frozen=True)
class VerificationFinding:
    """Structured result of one assurance check."""

    check_name: str
    passed: bool
    message: str
    severity: Severity = Severity.INFO
    error_code: ErrorCode = ErrorCode.NONE
    expected_value: Decimal | None = None
    actual_value: Decimal | None = None


def check_total_consistency(
    reported_total: Decimal,
    calculated_total: Decimal,
) -> VerificationFinding:
    """Compare the reported total with a reference total."""

    if not reported_total.is_finite() or not calculated_total.is_finite():
        raise ValueError("Both totals must be finite decimal values.")

    if reported_total == calculated_total:
        return VerificationFinding(
            check_name="total_consistency",
            passed=True,
            message="Reported total matches calculated total.",
            expected_value=calculated_total,
            actual_value=reported_total,
        )

    return VerificationFinding(
        check_name="total_consistency",
        passed=False,
        message=(
            f"Total mismatch: reported={reported_total}, "
            f"calculated={calculated_total}."
        ),
        severity=Severity.CRITICAL,
        error_code=ErrorCode.TOTAL_MISMATCH,
        expected_value=calculated_total,
        actual_value=reported_total,
    )

def check_line_item_sum(
    reported_total: Decimal,
    line_items: Sequence[Decimal],
) -> VerificationFinding:
    """Independently recompute a total from calculation line items."""

    if not line_items:
        raise ValueError("At least one line item is required.")

    if any(not item.is_finite() for item in line_items):
        raise ValueError("All line items must be finite decimal values.")

    calculated_total = sum(line_items, Decimal("0"))

    if not calculated_total.is_finite():
        raise ValueError("Calculated total must be finite.")

    comparison = check_total_consistency(
        reported_total=reported_total,
        calculated_total=calculated_total,
    )

    if comparison.passed:
        return VerificationFinding(
            check_name="line_item_sum",
            passed=True,
            message="Reported total matches the sum of line items.",
            severity=Severity.INFO,
            error_code=ErrorCode.NONE,
            expected_value=calculated_total,
            actual_value=reported_total,
        )

    return VerificationFinding(
        check_name="line_item_sum",
        passed=False,
        message=(
            f"Line-item sum mismatch: reported={reported_total}, "
            f"recomputed={calculated_total}."
        ),
        severity=Severity.CRITICAL,
        error_code=ErrorCode.LINE_ITEM_SUM_MISMATCH,
        expected_value=calculated_total,
        actual_value=reported_total,
    )
