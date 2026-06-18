"""Convert unstructured inspection observations to standardised CDSCO report format.

Uses the powerful local model for long-form structured generation.
"""
from __future__ import annotations

import json
import re
from src.core.llm import complete_powerful
from src.inspection.schemas import InspectionRequest, InspectionResponse

_SYSTEM = """You are a CDSCO inspection officer drafting a formal inspection report.
Convert the unstructured site inspection observations into a standardised CDSCO report structure.

Return a JSON object with keys:
- executive_summary: concise overview (max 150 words)
- critical_observations: list of strings — observations that may pose immediate risk to patient safety
- major_observations: list of strings — significant non-compliance requiring corrective action
- minor_observations: list of strings — minor deviations for improvement
- recommendations: list of strings — specific corrective/preventive actions recommended

Classify each observation per CDSCO inspection guidelines (Schedule M / GCP / GLP as applicable).
Return only valid JSON. No preamble. No markdown fences."""


class InspectionReportGenerator:
    def generate(self, req: InspectionRequest) -> InspectionResponse:
        user_prompt = (
            f"Inspection Type: {req.inspection_type}\n"
            f"Site: {req.site_name}\n"
            f"Date: {req.inspection_date}\n\n"
            f"Raw Observations:\n{req.observations_raw}"
        )

        raw = complete_powerful(system=_SYSTEM, user=user_prompt)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        json_str = json_match.group(1).strip() if json_match else raw

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            parsed = {
                "executive_summary": raw[:500],
                "critical_observations": [],
                "major_observations": [],
                "minor_observations": [],
                "recommendations": ["Manual review required — automated parsing failed."],
            }

        critical = parsed.get("critical_observations", [])
        major    = parsed.get("major_observations", [])
        minor    = parsed.get("minor_observations", [])
        recs     = parsed.get("recommendations", [])

        def _section(title: str, items: list[str]) -> str:
            if not items:
                return f"\n## {title}\nNil\n"
            lines = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(items))
            return f"\n## {title}\n{lines}\n"

        formatted = (
            f"# CDSCO Inspection Report\n"
            f"**Site:** {req.site_name}  \n"
            f"**Date:** {req.inspection_date}  \n"
            f"**Type:** {req.inspection_type}  \n"
            f"**Inspector:** {req.inspector_name}\n\n"
            f"## Executive Summary\n{parsed.get('executive_summary', '')}\n"
            + _section("Critical Observations", critical)
            + _section("Major Observations", major)
            + _section("Minor Observations", minor)
            + _section("Recommendations", recs)
        )

        return InspectionResponse(
            site_name=req.site_name,
            inspection_date=req.inspection_date,
            inspection_type=req.inspection_type,
            executive_summary=parsed.get("executive_summary", ""),
            critical_observations=critical,
            major_observations=major,
            minor_observations=minor,
            recommendations=recs,
            formatted_report=formatted,
        )
