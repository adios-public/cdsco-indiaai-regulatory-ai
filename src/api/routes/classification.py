from fastapi import APIRouter
from src.classification.sae_classifier import SAEClassifier
from src.classification.schemas import ClassificationRequest, ClassificationResponse

router = APIRouter(tags=["SAE Classification"])
_classifier = SAEClassifier()


@router.post("/classify-sae", response_model=ClassificationResponse)
def classify(req: ClassificationRequest) -> ClassificationResponse:
    """Classify SAE severity, detect duplicates, and assign priority."""
    return _classifier.classify(req)
