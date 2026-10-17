
import pytest

from consistency_checker.checker import (
    ErrorCode,
    Severity,
    VerificationFinding,
)
from decision_engine.engine import (
    AssuranceAction,
    decide_assurance_action,
)


def make_finding(passed=True, severity=Severity.INFO):
    return VerificationFinding(
        check_name="test_check",
        passed=passed,
        message="Test finding",
        severity=severity,
        error_code=ErrorCode.NONE,
    )


def test_all_checks_pass():
    decision = decide_assurance_action([make_finding()])

    assert decision.action == AssuranceAction.PASS
    assert decision.failed_checks == 0


def test_no_findings_escalates():
    decision = decide_assurance_action([])

    assert decision.action == AssuranceAction.HUMAN_REVIEW


def test_critical_failure_escalates():
    decision = decide_assurance_action([
        make_finding(passed=False, severity=Severity.CRITICAL)
    ])

    assert decision.action == AssuranceAction.HUMAN_REVIEW
    assert decision.failed_checks == 1


def test_warning_failure_regenerates():
    decision = decide_assurance_action([
        make_finding(passed=False, severity=Severity.WARNING)
    ])

    assert decision.action == AssuranceAction.REGENERATE


def test_retry_limit_escalates():
    decision = decide_assurance_action(
        [make_finding(passed=False, severity=Severity.WARNING)],
        regeneration_attempts=2,
        max_regenerations=2,
    )

    assert decision.action == AssuranceAction.HUMAN_REVIEW


def test_multiple_failures_counted():
    decision = decide_assurance_action([
        make_finding(passed=False, severity=Severity.WARNING),
        make_finding(passed=False, severity=Severity.CRITICAL),
    ])

    assert decision.failed_checks == 2
    assert decision.action == AssuranceAction.HUMAN_REVIEW


@pytest.mark.parametrize("attempts,limit", [(-1, 2), (0, -1)])
def test_negative_retry_settings_rejected(attempts, limit):
    with pytest.raises(ValueError):
        decide_assurance_action(
            [make_finding()],
            regeneration_attempts=attempts,
            max_regenerations=limit,
        )
