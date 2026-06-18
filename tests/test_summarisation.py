"""Tests for document summariser — mocks LLM."""
import json
import pytest
from unittest.mock import patch
from src.summarisation.document_summariser import DocumentSummariser
from src.summarisation.schemas import SummarisationRequest, SourceType


@pytest.fixture
def summariser():
    return DocumentSummariser()


def test_sugam_checklist_summary(summariser):
    mock = json.dumps({
        "summary": "New drug application for Drug X submitted by PharmaCo.",
        "key_decisions": ["Fast-track designation requested"],
        "action_items": ["Request clinical data module"],
        "flagged_concerns": [],
    })
    with patch("src.summarisation.document_summariser.complete", return_value=mock):
        req = SummarisationRequest(
            text="Application data...",
            source_type=SourceType.sugam_checklist,
        )
        result = summariser.summarise(req)
        assert "PharmaCo" in result.summary
        assert len(result.key_decisions) == 1
        assert result.word_count > 0


def test_malformed_llm_response_handled(summariser):
    with patch("src.summarisation.document_summariser.complete", return_value="Not JSON"):
        req = SummarisationRequest(
            text="Some narration.",
            source_type=SourceType.sae_narration,
        )
        result = summariser.summarise(req)
        assert len(result.flagged_concerns) > 0
