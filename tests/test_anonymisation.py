"""Tests for PII/PHI anonymisation engine."""
import pytest
from src.anonymisation.anonymiser import AnonymisationEngine
from src.anonymisation.schemas import AnonymisationRequest, AnonymisationMode, DocumentType


@pytest.fixture
def engine():
    return AnonymisationEngine()


def test_detects_person_name(engine):
    req = AnonymisationRequest(
        text="Dr. Anjali Sharma reviewed the case.",
        mode=AnonymisationMode.pseudonymise,
    )
    result = engine.process(req)
    assert any(e.entity_type == "PERSON" for e in result.entities_detected)
    assert "Anjali Sharma" not in result.anonymised_text


def test_detects_aadhaar(engine):
    req = AnonymisationRequest(
        text="Patient Aadhaar: 1234-5678-9012 enrolled in trial.",
        mode=AnonymisationMode.anonymise,
    )
    result = engine.process(req)
    assert any(e.entity_type == "AADHAAR_NUMBER" for e in result.entities_detected)
    assert "1234-5678-9012" not in result.anonymised_text
    assert "DPDP-2023" in " ".join(result.compliance_flags)


def test_pseudonymise_produces_tokens(engine):
    req = AnonymisationRequest(
        text="Subject John Doe (PT-00123) in Hyderabad.",
        mode=AnonymisationMode.pseudonymise,
    )
    result = engine.process(req)
    assert len(result.token_map) > 0
    assert all(k.startswith("<TOK-") for k in result.token_map)


def test_anonymise_irreversible(engine):
    req = AnonymisationRequest(
        text="Patient Jane Smith, DOB 01/01/1990.",
        mode=AnonymisationMode.anonymise,
    )
    result = engine.process(req)
    # Irreversible mode should produce redaction labels, not tokens
    assert "REDACTED" in result.anonymised_text
    assert len(result.token_map) == 0


def test_k_anonymity_estimate_positive(engine):
    req = AnonymisationRequest(text="Hello world.", mode=AnonymisationMode.anonymise)
    result = engine.process(req)
    assert result.k_anonymity_estimate >= 5
