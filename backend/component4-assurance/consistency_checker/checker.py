
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ErrorCode(str, Enum):
    NONE = "NONE"
    TOTAL_MISMATCH = "TOTAL_MISMATCH"


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
