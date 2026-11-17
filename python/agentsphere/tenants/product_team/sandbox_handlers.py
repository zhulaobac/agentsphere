"""Sandbox-safe handlers. Secrets arrive via the worker FD, never environment variables."""

from __future__ import annotations

import json
from typing import Any, Mapping

from agentsphere.tenants.product_team.report import CompetitorRow, ResearchReport


def assemble_report(
    arguments: Mapping[str, Any],
    secrets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Merge tool notes into the structured research artifact. Do not echo secrets."""

    del secrets  # vault tokens must not enter the report
    topic = str(arguments.get("topic") or "").strip()
    notes = arguments.get("notes") or {}
    lookup = _as_dict(notes.get("research_lookup_topic") or arguments.get("lookup"))
    listed = _as_dict(notes.get("research_list_offerings") or arguments.get("offerings"))
    compared = _as_dict(notes.get("research_compare_offerings") or arguments.get("comparison"))

    findings = _as_str_list(lookup.get("user_need_findings"))
    recommendations = _as_str_list(compared.get("recommendations")) or _as_str_list(
        lookup.get("recommendations")
    )
    rows = _as_rows(compared.get("rows") or listed.get("offerings"))
    sources = _as_str_list(lookup.get("sources")) or ["fixture-catalog"]
    context = str(lookup.get("problem_context") or arguments.get("problem_context") or "").strip()

    report = ResearchReport(
        topic=topic or str(lookup.get("topic") or "unspecified topic"),
        problem_context=context,
        user_need_findings=findings,
        competitor_comparison=rows,
        implications_or_recommendations=recommendations,
        sources=sources,
    )
    return json.loads(report.model_dump_json())


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _as_rows(value: Any) -> list[CompetitorRow]:
    if not isinstance(value, list):
        return []
    rows: list[CompetitorRow] = []
    for item in value:
        if isinstance(item, CompetitorRow):
            rows.append(item)
        elif isinstance(item, dict):
            rows.append(CompetitorRow.model_validate(item))
        else:
            rows.append(CompetitorRow(name=str(item)))
    return rows
