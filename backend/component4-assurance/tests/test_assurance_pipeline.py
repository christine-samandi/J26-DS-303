
from decimal import Decimal

import pytest

from assurance_pipeline.pipeline import assure_tax_total
from decision_engine.engine import AssuranceAction


def test_matching_total_passes():
    result = assure_tax_total(
        reported_total=Decimal("150000"),
        calculated_total=Decimal("150000"),
    )

    assert result.decision.action == AssuranceAction.PASS
    assert len(result.findings) == 1
    assert result.findings[0].passed is True


def test_mismatching_total_escalates():
    result = assure_tax_total(
        reported_total=Decimal("180000"),
        calculated_total=Decimal("150000"),
    )

    assert result.decision.action == AssuranceAction.HUMAN_REVIEW
    assert result.findings[0].passed is False
    assert result.decision.failed_checks == 1


def test_invalid_total_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        assure_tax_total(
            reported_total=Decimal("NaN"),
            calculated_total=Decimal("150000"),
        )
