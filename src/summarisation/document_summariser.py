"""Document summarisation for SUGAM checklists, SAE narrations, and meeting transcripts.

Uses the powerful local model (qwen3.6) for long-form synthesis tasks.
"""
from __future__ import annotations

import json
from src.core.llm import complete_powerful
from src.summarisation.schemas import SummarisationRequest, SummarisationResponse, SourceType

_SYSTEM_PROMPTS: dict[str, str] = {
    SourceType.sugam_checklist: """You are a CDSCO regulatory affairs specialist.
Analyse the SUGAM portal application checklist data provided.
Return a JSON object with keys:
- summary: concise prose overview (max {max_words} words)
- key_decisions: list of strings — regulatory decision points identified
- action_items: list of strings — outstanding items the reviewer must act on
- flagged_concerns: list of strings — data gaps, inconsistencies, or risk flags
Return only valid JSON. No preamble.""",

    SourceType.sae_narration: """You are a pharmacovigilance expert reviewing a Serious Adverse Event (SAE) case narration for CDSCO.
Analyse the SAE narrative provided.
Return a JSON object with keys:
- summary: concise case summary (max {max_words} words) covering patient profile, event, timeline, causality, outcome
- key_decisions: list — causality assessments and regulatory signals identified
- action_items: list — follow-up required (additional data, expedited reporting, label change etc.)
- flagged_concerns: list — missing data, inconsistencies, duplicate signals
Return only valid JSON. No preamble.""",

    SourceType.meeting_transcript: """You are a regulatory meeting secretary for CDSCO.
Analyse the meeting transcript or audio transcript provided.
Return a JSON object with keys:
- summary: concise meeting summary (max {max_words} words)
- key_decisions: list — decisions made during the meeting
- action_items: list — action items with implicit owners where mentioned
- flagged_concerns: list — open issues or unresolved points
Return only valid JSON. No preamble.""",
}


class DocumentSummariser:
    def summarise(self, req: SummarisationRequest) -> SummarisationResponse:
        system = _SYSTEM_PROMPTS[req.source_type].format(max_words=req.max_summary_words)
        # Use powerful model for summarisation — it handles long documents better
        raw = complete_powerful(system=system, user=req.text)

        # Strip <think>...</think> blocks that reasoning models (deepseek-r1, qwen3.6) emit
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Extract JSON from response (model may wrap it in markdown fences)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        json_str = json_match.group(1).strip() if json_match else raw

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            parsed = {
                "summary": raw[:1000],
                "key_decisions": [],
                "action_items": [],
                "flagged_concerns": ["Model returned non-JSON; manual review required"],
            }

        summary = parsed.get("summary", "")
        return SummarisationResponse(
            source_type=req.source_type,
            summary=summary,
            key_decisions=parsed.get("key_decisions", []),
            action_items=parsed.get("action_items", []),
            flagged_concerns=parsed.get("flagged_concerns", []),
            word_count=len(summary.split()),
        )
