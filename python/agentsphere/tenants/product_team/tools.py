"""Tool registrations for the research workflow. Handlers go through Tool Registry / sandbox."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentsphere.tenants.product_team.constants import (
    SECRET_NAME,
    TENANT_OWNER,
    TOOL_ASSEMBLE,
    TOOL_COMPARE,
    TOOL_LIST,
    TOOL_LOOKUP,
    TOOL_VERSION,
)
from agentsphere.tenants.product_team.sources import ResearchSource
from agentsphere.tool_registry.registry import ToolRegistry
from agentsphere.tool_registry.spec import RiskLevel, ToolSpec

ASSEMBLE_SANDBOX_HANDLER = "agentsphere.tenants.product_team.sandbox_handlers:assemble_report"


def register_research_tools(registry: ToolRegistry, source: ResearchSource) -> list[ToolSpec]:
    """Bind fixture (or other) source behind ToolSpec handlers. Do not call source from the model."""

    def lookup(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return source.lookup_topic(str(arguments.get("topic") or ""))

    def list_offerings(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return source.list_offerings(str(arguments.get("category") or arguments.get("topic") or ""))

    def compare(arguments: Mapping[str, Any]) -> dict[str, Any]:
        names = arguments.get("names")
        name_list = [str(n) for n in names] if isinstance(names, list) else None
        return source.compare_offerings(str(arguments.get("topic") or ""), name_list)

    specs = [
        ToolSpec(
            name=TOOL_LOOKUP,
            version=TOOL_VERSION,
            owner=TENANT_OWNER,
            description="Look up problem context and user-need findings for a product category or research topic.",
            json_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            risk_level=RiskLevel.LOW,
            idempotent=True,
            required_credentials=[SECRET_NAME],
            handler=lookup,
        ),
        ToolSpec(
            name=TOOL_LIST,
            version=TOOL_VERSION,
            owner=TENANT_OWNER,
            description="List generic competing offerings (role labels, not commercial brands) for a category.",
            json_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "topic": {"type": "string"},
                },
                "additionalProperties": False,
            },
            risk_level=RiskLevel.LOW,
            idempotent=True,
            handler=list_offerings,
        ),
        ToolSpec(
            name=TOOL_COMPARE,
            version=TOOL_VERSION,
            owner=TENANT_OWNER,
            description="Compare offerings for a topic: positioning, strengths, and gaps.",
            json_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "names": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
            risk_level=RiskLevel.LOW,
            idempotent=True,
            handler=compare,
        ),
        ToolSpec(
            name=TOOL_ASSEMBLE,
            version=TOOL_VERSION,
            owner=TENANT_OWNER,
            description="Assemble a structured research and competitive-analysis artifact in the sandbox.",
            json_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "notes": {"type": "object", "additionalProperties": True},
                    "lookup": {"type": "object"},
                    "offerings": {"type": "object"},
                    "comparison": {"type": "object"},
                    "problem_context": {"type": "string"},
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
            risk_level=RiskLevel.LOW,
            idempotent=True,
            sandboxed=True,
            sandbox_handler=ASSEMBLE_SANDBOX_HANDLER,
        ),
    ]
    for spec in specs:
        registry.register(spec)
    return specs


def tool_asset_specs(specs: list[ToolSpec]) -> list[tuple[ToolSpec, dict[str, Any]]]:
    """JSON-serializable Control Plane Tool assets (handlers are not serialized)."""

    records: list[tuple[ToolSpec, dict[str, Any]]] = []
    for spec in specs:
        records.append(
            (
                spec,
                {
                    "name": spec.name,
                    "version": spec.version,
                    "description": spec.description,
                    "json_schema": spec.json_schema,
                    "risk_level": spec.risk_level.value,
                    "idempotent": spec.idempotent,
                    "sandboxed": spec.sandboxed,
                    "sandbox_handler": spec.sandbox_handler,
                    "required_credentials": list(spec.required_credentials),
                },
            )
        )
    return records
