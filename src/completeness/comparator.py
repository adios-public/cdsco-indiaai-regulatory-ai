"""Semantic + lexical document comparison with substantive change detection."""
from __future__ import annotations

import difflib
from sentence_transformers import SentenceTransformer, util

from src.completeness.schemas import (
    ChangeType, ComparisonRequest, ComparisonResponse, DocumentChange
)

_SUBSTANTIVE_THRESHOLD = 0.75   # cosine similarity below this = substantive change


class DocumentComparator:
    def __init__(self) -> None:
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def compare(self, req: ComparisonRequest) -> ComparisonResponse:
        v1_lines = req.document_v1.splitlines()
        v2_lines = req.document_v2.splitlines()

        differ = difflib.unified_diff(v1_lines, v2_lines, lineterm="", n=0)
        raw_diff = list(differ)

        changes: list[DocumentChange] = []
        section = "Body"
        for line in raw_diff:
            if line.startswith("@@"):
                section = line
                continue
            if line.startswith("+") and not line.startswith("+++"):
                original = ""
                revised = line[1:].strip()
                ctype = ChangeType.addition
            elif line.startswith("-") and not line.startswith("---"):
                original = line[1:].strip()
                revised = ""
                ctype = ChangeType.deletion
            else:
                continue

            # Determine substantiveness via semantic similarity
            if original and revised:
                emb1 = self._model.encode(original, convert_to_tensor=True)
                emb2 = self._model.encode(revised, convert_to_tensor=True)
                sim = float(util.cos_sim(emb1, emb2))
                is_substantive = sim < _SUBSTANTIVE_THRESHOLD
                significance = "high" if sim < 0.5 else ("medium" if sim < _SUBSTANTIVE_THRESHOLD else "low")
            else:
                is_substantive = req.highlight_substantive
                sim = 0.0
                significance = "medium"

            changes.append(DocumentChange(
                change_type=ctype,
                section=section,
                original=original,
                revised=revised,
                is_substantive=is_substantive,
                significance=significance,
            ))

        # Overall similarity
        emb_v1 = self._model.encode(req.document_v1, convert_to_tensor=True)
        emb_v2 = self._model.encode(req.document_v2, convert_to_tensor=True)
        overall_sim = round(float(util.cos_sim(emb_v1, emb_v2)), 3)

        substantive = [c for c in changes if c.is_substantive]
        summary = (
            f"{len(changes)} total changes detected ({len(substantive)} substantive). "
            f"Document similarity: {overall_sim:.0%}. "
            + ("Major revisions present — reviewer attention required." if substantive else "Minor edits only.")
        )

        return ComparisonResponse(
            total_changes=len(changes),
            substantive_changes=len(substantive),
            changes=changes,
            similarity_score=overall_sim,
            reviewer_summary=summary,
        )
