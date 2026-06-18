from pydantic import BaseModel


class InspectionRequest(BaseModel):
    observations_raw: str   # unstructured / handwritten observations text
    site_name: str
    inspection_date: str
    inspector_name: str = "[REDACTED]"
    inspection_type: str = "GMP"   # GMP | GCP | GLP | GDP


class InspectionResponse(BaseModel):
    site_name: str
    inspection_date: str
    inspection_type: str
    executive_summary: str
    critical_observations: list[str]
    major_observations: list[str]
    minor_observations: list[str]
    recommendations: list[str]
    formatted_report: str
