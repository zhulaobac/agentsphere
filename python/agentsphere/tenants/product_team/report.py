"""Structured research / competitive-analysis artifact produced by the tenant agent."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field


class CompetitorRow(BaseModel):
    """Generic offering row. Names are category roles (e.g. Incumbent suite), not brands."""

    name: str
    positioning: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    topic: str
    problem_context: str
    user_need_findings: list[str] = Field(default_factory=list)
    competitor_comparison: list[CompetitorRow] = Field(default_factory=list)
    implications_or_recommendations: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    refused: bool = False
    reason: str = ""

    def has_required_sections(self) -> bool:
        if self.refused:
            return False
        return bool(
            self.topic.strip()
            and self.problem_context.strip()
            and self.user_need_findings
            and self.competitor_comparison
            and self.implications_or_recommendations
        )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty research report")
    fence = _FENCE.search(raw)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def parse_research_report(text: str | ResearchReport | dict[str, Any]) -> ResearchReport:
    if isinstance(text, ResearchReport):
        return text
    if isinstance(text, dict):
        return ResearchReport.model_validate(text)
    return ResearchReport.model_validate(_extract_json(str(text)))


def try_parse_research_report(text: str | ResearchReport | dict[str, Any] | None) -> ResearchReport | None:
    if text is None:
        return None
    try:
        return parse_research_report(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
