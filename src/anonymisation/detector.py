"""PII/PHI entity detection using Microsoft Presidio + custom Indian-context recognisers."""
from __future__ import annotations

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Additional Indian-specific patterns
INDIAN_PATTERNS = {
    "AADHAAR_NUMBER": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "PAN_NUMBER": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "PHONE_IN": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "PATIENT_ID": r"\b(?:PT|PAT|MR)-?\d{4,10}\b",
    "TRIAL_SUBJECT_ID": r"\b(?:SUB|SUBJ|SID)-?\d{4,10}\b",
}


class PIIDetector:
    def __init__(self) -> None:
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"],
        )
        self._register_indian_recognisers()

    def _register_indian_recognisers(self) -> None:
        from presidio_analyzer import PatternRecognizer, Pattern
        for name, pattern in INDIAN_PATTERNS.items():
            recogniser = PatternRecognizer(
                supported_entity=name,
                patterns=[Pattern(name=name, regex=pattern, score=0.85)],
            )
            self._analyzer.registry.add_recognizer(recogniser)

    def detect(self, text: str) -> list[RecognizerResult]:
        return self._analyzer.analyze(
            text=text,
            language="en",
            entities=[
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
                "DATE_TIME", "NRP", "MEDICAL_LICENSE", "URL",
                "AADHAAR_NUMBER", "PAN_NUMBER", "PHONE_IN",
                "PATIENT_ID", "TRIAL_SUBJECT_ID",
            ],
        )
