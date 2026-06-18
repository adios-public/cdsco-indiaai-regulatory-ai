from enum import Enum
from pydantic import BaseModel


class SAESeverity(str, Enum):
    death = "death"
    disability = "disability"
    hospitalisation = "hospitalisation"
    life_threatening = "life_threatening"
    congenital_anomaly = "congenital_anomaly"
    other = "other"


class ReviewPriority(str, Enum):
    critical = "critical"     # death / life-threatening
    high = "high"             # disability / hospitalisation
    medium = "medium"         # other serious
    low = "low"               # non-serious / duplicate


class ClassificationRequest(BaseModel):
    case_narration: str
    check_duplicate: bool = True
    existing_case_ids: list[str] = []


class ClassificationResponse(BaseModel):
    severity: SAESeverity
    priority: ReviewPriority
    confidence: float
    is_duplicate: bool
    duplicate_case_ids: list[str]
    rationale: str
    expedited_reporting_required: bool   # 15-day expedited per Schedule Y
