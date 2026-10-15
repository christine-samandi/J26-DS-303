
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class VerificationFinding:
    """Represents one issue detected during assurance."""

    check_name: str
    passed: bool
    message: str


def check_total_consistency(
    reported_total: Decimal,
    calculated_total: Decimal,
) -> VerificationFinding:
    """Compare a reported total against an independently supplied total."""

    if reported_total == calculated_total:
        return VerificationFinding(
            check_name="total_consistency",
            passed=True,
            message="Reported total matches calculated total.",
        )

    return VerificationFinding(
        check_name="total_consistency",
        passed=False,
        message=(
            f"Total mismatch: reported={reported_total}, "
            f"calculated={calculated_total}."
        ),
    )
