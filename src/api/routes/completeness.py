from fastapi import APIRouter
from src.completeness.checker import CompletenessChecker
from src.completeness.comparator import DocumentComparator
from src.completeness.schemas import (
    CompletenessRequest, CompletenessResponse,
    ComparisonRequest, ComparisonResponse,
)

router = APIRouter(tags=["Completeness & Comparison"])
_checker = CompletenessChecker()
_comparator = DocumentComparator()


@router.post("/assess-completeness", response_model=CompletenessResponse)
def assess(req: CompletenessRequest) -> CompletenessResponse:
    """Verify mandatory fields and flag missing/inconsistent data."""
    return _checker.assess(req)


@router.post("/compare", response_model=ComparisonResponse)
def compare(req: ComparisonRequest) -> ComparisonResponse:
    """Identify substantive changes between two document versions."""
    return _comparator.compare(req)
