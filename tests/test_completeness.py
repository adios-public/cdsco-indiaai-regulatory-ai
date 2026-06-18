"""Tests for completeness checker."""
import pytest
from src.completeness.checker import CompletenessChecker
from src.completeness.schemas import CompletenessRequest, SchemaType


@pytest.fixture
def checker():
    return CompletenessChecker()


def test_complete_sae_report(checker):
    doc = {
        "case_id": "SAE-2024-001",
        "patient_age": 45,
        "patient_sex": "M",
        "suspect_drug": "Drug X 10mg",
        "event_description": "Anaphylaxis following first dose.",
        "event_onset_date": "2024-01-15",
        "outcome": "Recovered",
        "causality_assessment": "Probable",
        "reporter_name": "[REDACTED]",
    }
    req = CompletenessRequest(document=doc, schema_type=SchemaType.sae_report)
    result = checker.assess(req)
    assert result.is_complete is True
    assert result.completeness_score == 1.0
    assert "ACCEPT" in result.review_recommendation


def test_incomplete_nda_flags_missing(checker):
    doc = {"applicant_name": "PharmaCo Ltd"}
    req = CompletenessRequest(document=doc, schema_type=SchemaType.new_drug_application)
    result = checker.assess(req)
    assert result.is_complete is False
    assert result.completeness_score < 0.5
    assert any(m.severity == "mandatory" for m in result.missing_fields)
    assert "RETURN" in result.review_recommendation


def test_score_between_zero_and_one(checker):
    req = CompletenessRequest(document={}, schema_type=SchemaType.clinical_trial)
    result = checker.assess(req)
    assert 0.0 <= result.completeness_score <= 1.0
