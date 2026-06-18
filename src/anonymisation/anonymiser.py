"""Two-step de-identification: pseudonymisation then irreversible anonymisation."""
from __future__ import annotations

import hashlib
import re
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.anonymisation.detector import PIIDetector
from src.anonymisation.schemas import (
    AnonymisationMode, AnonymisationRequest, AnonymisationResponse, DetectedEntity
)
from src.core.config import get_settings


_GENERALISATION_MAP = {
    "DATE_TIME": "[DATE REDACTED]",
    "LOCATION": "[LOCATION REDACTED]",
    "EMAIL_ADDRESS": "[EMAIL REDACTED]",
    "URL": "[URL REDACTED]",
    "AADHAAR_NUMBER": "[AADHAAR REDACTED]",
    "PAN_NUMBER": "[PAN REDACTED]",
    "PHONE_NUMBER": "[PHONE REDACTED]",
    "PHONE_IN": "[PHONE REDACTED]",
    "MEDICAL_LICENSE": "[LICENSE REDACTED]",
    "PATIENT_ID": "[PATIENT-ID REDACTED]",
    "TRIAL_SUBJECT_ID": "[SUBJECT-ID REDACTED]",
    "PERSON": "[INDIVIDUAL REDACTED]",
    "NRP": "[NATIONALITY REDACTED]",
}


class AnonymisationEngine:
    def __init__(self) -> None:
        self._detector = PIIDetector()
        self._anonymizer = AnonymizerEngine()
        self._salt = get_settings().anonymisation_salt
        self._prefix = get_settings().pseudo_token_prefix

    def _pseudo_token(self, text: str, entity_type: str) -> str:
        digest = hashlib.sha256(f"{self._salt}:{entity_type}:{text}".encode()).hexdigest()[:8].upper()
        return f"<{self._prefix}-{entity_type[:4]}-{digest}>"

    def process(self, req: AnonymisationRequest) -> AnonymisationResponse:
        results = self._detector.detect(req.text)

        entities = [
            DetectedEntity(
                text=req.text[r.start:r.end],
                entity_type=r.entity_type,
                start=r.start,
                end=r.end,
                score=round(r.score, 3),
            )
            for r in results
        ]

        token_map: dict[str, str] = {}

        if req.mode == AnonymisationMode.pseudonymise:
            operators = {}
            for r in results:
                original = req.text[r.start:r.end]
                token = self._pseudo_token(original, r.entity_type)
                token_map[token] = r.entity_type
                operators[r.entity_type] = OperatorConfig(
                    "replace", {"new_value": token}
                )
            anonymised = self._anonymizer.anonymize(
                text=req.text, analyzer_results=results, operators=operators
            ).text
        else:
            # Irreversible: generalise using the redaction map
            operators = {
                etype: OperatorConfig("replace", {"new_value": label})
                for etype, label in _GENERALISATION_MAP.items()
            }
            anonymised = self._anonymizer.anonymize(
                text=req.text, analyzer_results=results, operators=operators
            ).text

        # Rough k-anonymity estimate based on unique quasi-identifiers remaining
        quasi_ids_remaining = len(re.findall(r"\[\w+ REDACTED\]", anonymised))
        k_estimate = max(5, 100 - quasi_ids_remaining * 10)

        compliance_flags = []
        if any(e.entity_type == "AADHAAR_NUMBER" for e in entities):
            compliance_flags.append("DPDP-2023:S8-sensitive-personal-data-detected")
        if any(e.entity_type in {"PERSON", "DATE_TIME"} for e in entities):
            compliance_flags.append("ICMR:de-identification-applied")

        return AnonymisationResponse(
            original_length=len(req.text),
            anonymised_text=anonymised,
            mode=req.mode,
            entities_detected=entities,
            token_map=token_map,
            k_anonymity_estimate=k_estimate,
            compliance_flags=compliance_flags,
        )
