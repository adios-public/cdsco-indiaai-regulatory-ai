from enum import Enum
from pydantic import BaseModel
from typing import Any


class SchemaType(str, Enum):
    new_drug_application = "new_drug_application"
    clinical_trial = "clinical_trial"
    sae_report = "sae_report"
    medical_device = "medical_device"


class MissingField(BaseModel):
    field: str
    section: str
    severity: str   # "mandatory" | "recommended"
    reason: str


class CompletenessRequest(BaseModel):
    document: dict[str, Any]
    schema_type: SchemaType
    flag_inconsistencies: bool = True


class CompletenessResponse(BaseModel):
    schema_type: SchemaType
    is_complete: bool
    completeness_score: float   # 0.0–1.0
    missing_fields: list[MissingField]
    inconsistencies: list[str]
    review_recommendation: str


class ChangeType(str, Enum):
    addition = "addition"
    deletion = "deletion"
    modification = "modification"


class DocumentChange(BaseModel):
    change_type: ChangeType
    section: str
    original: str
    revised: str
    is_substantive: bool
    significance: str   # "high" | "medium" | "low"


class ComparisonRequest(BaseModel):
    document_v1: str
    document_v2: str
    highlight_substantive: bool = True


class ComparisonResponse(BaseModel):
    total_changes: int
    substantive_changes: int
    changes: list[DocumentChange]
    similarity_score: float
    reviewer_summary: str
