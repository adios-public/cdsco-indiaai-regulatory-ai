from fastapi import APIRouter
from src.anonymisation.anonymiser import AnonymisationEngine
from src.anonymisation.schemas import AnonymisationRequest, AnonymisationResponse

router = APIRouter(tags=["Anonymisation"])
_engine = AnonymisationEngine()


@router.post("/anonymise", response_model=AnonymisationResponse)
def anonymise(req: AnonymisationRequest) -> AnonymisationResponse:
    """Detect and de-identify PII/PHI in clinical documents."""
    return _engine.process(req)
