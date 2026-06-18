"""SAE severity classification using local Ollama model with rule-based priority escalation."""
from __future__ import annotations

import json
import re
from src.core.llm import complete
from src.classification.schemas import (
    ClassificationRequest, ClassificationResponse, SAESeverity, ReviewPriority
)

_SYSTEM = """You are a pharmacovigilance expert classifying Serious Adverse Events (SAEs)
for CDSCO regulatory review under Schedule Y of the Drugs and Cosmetics Act.

Analyse the case narration and return a JSON object with keys:
- severity: one of "death" | "disability" | "hospitalisation" | "life_threatening" | "congenital_anomaly" | "other"
- confidence: float 0.0-1.0
- rationale: brief explanation (max 100 words) of classification reasoning
- expedited_reporting_required: true if the event requires 15-day expedited reporting per Schedule Y

Return only valid JSON. No preamble. No markdown fences."""

_PRIORITY_MAP = {
    SAESeverity.death: ReviewPriority.critical,
    SAESeverity.life_threatening: ReviewPriority.critical,
    SAESeverity.disability: ReviewPriority.high,
    SAESeverity.congenital_anomaly: ReviewPriority.high,
    SAESeverity.hospitalisation: ReviewPriority.high,
    SAESeverity.other: ReviewPriority.medium,
}


class SAEClassifier:
    def classify(self, req: ClassificationRequest) -> ClassificationResponse:
        raw = complete(system=_SYSTEM, user=req.case_narration)

        # Strip reasoning traces from models like deepseek-r1 / qwen3.6
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        json_str = json_match.group(1).strip() if json_match else raw

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            parsed = {
                "severity": "other",
                "confidence": 0.5,
                "rationale": "Classification uncertain — manual review required.",
                "expedited_reporting_required": False,
            }

        severity = SAESeverity(parsed.get("severity", "other"))
        priority = _PRIORITY_MAP.get(severity, ReviewPriority.medium)
        confidence = float(parsed.get("confidence", 0.5))

        is_duplicate = False
        duplicate_ids: list[str] = []
        if req.check_duplicate and req.existing_case_ids:
            # Stage 2: vector similarity search against CDSCO case database
            is_duplicate = False

        if is_duplicate:
            priority = ReviewPriority.low

        return ClassificationResponse(
            severity=severity,
            priority=priority,
            confidence=round(confidence, 3),
            is_duplicate=is_duplicate,
            duplicate_case_ids=duplicate_ids,
            rationale=parsed.get("rationale", ""),
            expedited_reporting_required=bool(parsed.get("expedited_reporting_required", False)),
        )
