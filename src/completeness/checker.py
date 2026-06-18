"""Rule-based completeness checker against CDSCO mandatory field schemas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.completeness.schemas import (
    CompletenessRequest, CompletenessResponse, MissingField, SchemaType
)

_SCHEMA_DIR = Path(__file__).parent.parent.parent / "data" / "schemas"

# Inline fallback schemas when files aren't loaded
_MANDATORY_FIELDS: dict[str, list[dict]] = {
    SchemaType.new_drug_application: [
        {"field": "applicant_name", "section": "Administrative", "severity": "mandatory"},
        {"field": "drug_substance_name", "section": "Drug Information", "severity": "mandatory"},
        {"field": "proposed_indication", "section": "Clinical", "severity": "mandatory"},
        {"field": "dosage_form", "section": "Drug Information", "severity": "mandatory"},
        {"field": "route_of_administration", "section": "Drug Information", "severity": "mandatory"},
        {"field": "manufacturing_site", "section": "Manufacturing", "severity": "mandatory"},
        {"field": "clinical_trial_data", "section": "Clinical", "severity": "mandatory"},
        {"field": "safety_data", "section": "Safety", "severity": "mandatory"},
        {"field": "proposed_labelling", "section": "Labelling", "severity": "recommended"},
    ],
    SchemaType.sae_report: [
        {"field": "case_id", "section": "Administrative", "severity": "mandatory"},
        {"field": "patient_age", "section": "Patient", "severity": "mandatory"},
        {"field": "patient_sex", "section": "Patient", "severity": "mandatory"},
        {"field": "suspect_drug", "section": "Drug", "severity": "mandatory"},
        {"field": "event_description", "section": "Event", "severity": "mandatory"},
        {"field": "event_onset_date", "section": "Event", "severity": "mandatory"},
        {"field": "outcome", "section": "Event", "severity": "mandatory"},
        {"field": "causality_assessment", "section": "Assessment", "severity": "mandatory"},
        {"field": "reporter_name", "section": "Reporter", "severity": "mandatory"},
        {"field": "concomitant_medications", "section": "Drug", "severity": "recommended"},
    ],
    SchemaType.clinical_trial: [
        {"field": "protocol_number", "section": "Administrative", "severity": "mandatory"},
        {"field": "sponsor_name", "section": "Administrative", "severity": "mandatory"},
        {"field": "investigational_product", "section": "Product", "severity": "mandatory"},
        {"field": "study_phase", "section": "Study Design", "severity": "mandatory"},
        {"field": "primary_endpoint", "section": "Study Design", "severity": "mandatory"},
        {"field": "sample_size", "section": "Study Design", "severity": "mandatory"},
        {"field": "ethics_approval", "section": "Regulatory", "severity": "mandatory"},
        {"field": "informed_consent_process", "section": "Regulatory", "severity": "mandatory"},
    ],
}


class CompletenessChecker:
    def assess(self, req: CompletenessRequest) -> CompletenessResponse:
        required = _MANDATORY_FIELDS.get(req.schema_type, [])
        doc = req.document

        missing: list[MissingField] = []
        for field_def in required:
            field = field_def["field"]
            value = doc.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(MissingField(
                    field=field,
                    section=field_def["section"],
                    severity=field_def["severity"],
                    reason=f"Field '{field}' is absent or empty in the submission.",
                ))

        mandatory_total = sum(1 for f in required if f["severity"] == "mandatory")
        mandatory_missing = sum(1 for m in missing if m.severity == "mandatory")
        score = round(1.0 - (mandatory_missing / mandatory_total) if mandatory_total else 1.0, 3)

        inconsistencies: list[str] = []
        if req.flag_inconsistencies:
            inconsistencies = self._check_inconsistencies(doc, req.schema_type)

        if score == 1.0 and not inconsistencies:
            recommendation = "ACCEPT for technical screening"
        elif score >= 0.8:
            recommendation = "QUERY applicant for missing recommended fields before proceeding"
        else:
            recommendation = "RETURN to applicant — mandatory fields incomplete"

        return CompletenessResponse(
            schema_type=req.schema_type,
            is_complete=score == 1.0,
            completeness_score=score,
            missing_fields=missing,
            inconsistencies=inconsistencies,
            review_recommendation=recommendation,
        )

    def _check_inconsistencies(self, doc: dict, schema_type: str) -> list[str]:
        issues = []
        if schema_type == SchemaType.sae_report:
            onset = doc.get("event_onset_date", "")
            report = doc.get("report_date", "")
            if onset and report and onset > report:
                issues.append("event_onset_date is after report_date — chronology inconsistency")
        return issues
