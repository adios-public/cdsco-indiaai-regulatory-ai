from fastapi import APIRouter
from src.inspection.report_generator import InspectionReportGenerator
from src.inspection.schemas import InspectionRequest, InspectionResponse

router = APIRouter(tags=["Inspection Reports"])
_generator = InspectionReportGenerator()


@router.post("/generate-inspection-report", response_model=InspectionResponse)
def generate(req: InspectionRequest) -> InspectionResponse:
    """Convert site inspection observations to standardised CDSCO report."""
    return _generator.generate(req)
