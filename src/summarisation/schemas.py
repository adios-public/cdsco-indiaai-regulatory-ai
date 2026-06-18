from enum import Enum
from pydantic import BaseModel


class SourceType(str, Enum):
    sugam_checklist = "sugam_checklist"
    sae_narration = "sae_narration"
    meeting_transcript = "meeting_transcript"


class OutputFormat(str, Enum):
    structured = "structured"
    prose = "prose"


class SummarisationRequest(BaseModel):
    text: str
    source_type: SourceType
    output_format: OutputFormat = OutputFormat.structured
    max_summary_words: int = 300


class SummarisationResponse(BaseModel):
    source_type: SourceType
    summary: str
    key_decisions: list[str]
    action_items: list[str]
    flagged_concerns: list[str]
    word_count: int
