"""Tests for SAE classifier — mocks the LLM to avoid API calls in CI."""
import json
import pytest
from unittest.mock import patch
from src.classification.sae_classifier import SAEClassifier
from src.classification.schemas import ClassificationRequest, SAESeverity, ReviewPriority


@pytest.fixture
def classifier():
    return SAEClassifier()


@pytest.mark.parametrize("severity,expected_priority", [
    ("death", ReviewPriority.critical),
    ("life_threatening", ReviewPriority.critical),
    ("disability", ReviewPriority.high),
    ("hospitalisation", ReviewPriority.high),
    ("other", ReviewPriority.medium),
])
def test_priority_mapping(classifier, severity, expected_priority):
    mock_response = json.dumps({
        "severity": severity,
        "confidence": 0.92,
        "rationale": "Test rationale.",
        "expedited_reporting_required": severity in ("death", "life_threatening"),
    })
    with patch("src.classification.sae_classifier.complete", return_value=mock_response):
        req = ClassificationRequest(case_narration="A patient experienced adverse event.")
        result = classifier.classify(req)
        assert result.priority == expected_priority


def test_death_requires_expedited(classifier):
    mock_response = json.dumps({
        "severity": "death",
        "confidence": 0.99,
        "rationale": "Fatal outcome.",
        "expedited_reporting_required": True,
    })
    with patch("src.classification.sae_classifier.complete", return_value=mock_response):
        req = ClassificationRequest(case_narration="Patient died after administration.")
        result = classifier.classify(req)
        assert result.expedited_reporting_required is True
        assert result.severity == SAESeverity.death
