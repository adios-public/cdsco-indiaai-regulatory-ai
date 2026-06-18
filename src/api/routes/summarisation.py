from fastapi import APIRouter
from src.summarisation.document_summariser import DocumentSummariser
from src.summarisation.schemas import SummarisationRequest, SummarisationResponse

router = APIRouter(tags=["Summarisation"])
_summariser = DocumentSummariser()


@router.post("/summarise", response_model=SummarisationResponse)
def summarise(req: SummarisationRequest) -> SummarisationResponse:
    """Extract and synthesise key regulatory information from source documents."""
    return _summariser.summarise(req)
