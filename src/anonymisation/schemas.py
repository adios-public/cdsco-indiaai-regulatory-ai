from enum import Enum
from pydantic import BaseModel
from typing import Any


class AnonymisationMode(str, Enum):
    pseudonymise = "pseudonymise"   # step 1: replace with reversible token
    anonymise = "anonymise"         # step 2: irreversible generalisation


class DocumentType(str, Enum):
    clinical_trial = "clinical_trial"
    new_drug_application = "new_drug_application"
    sae_report = "sae_report"
    meeting_transcript = "meeting_transcript"
    general = "general"


class DetectedEntity(BaseModel):
    text: str
    entity_type: str
    start: int
    end: int
    score: float


class AnonymisationRequest(BaseModel):
    text: str
    mode: AnonymisationMode = AnonymisationMode.pseudonymise
    document_type: DocumentType = DocumentType.general

    model_config = {"json_schema_extra": {"examples": [{
        "text": "Patient Ramesh Kumar (DOB: 12/03/1985, Aadhaar: 1234-5678-9012) enrolled in Trial CT-2024-001.",
        "mode": "pseudonymise",
        "document_type": "clinical_trial"
    }]}}


class AnonymisationResponse(BaseModel):
    original_length: int
    anonymised_text: str
    mode: AnonymisationMode
    entities_detected: list[DetectedEntity]
    token_map: dict[str, str]   # pseudonym -> entity_type (token value never stored)
    k_anonymity_estimate: int
    compliance_flags: list[str]
