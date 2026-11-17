from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agentsphere.controlplane import Principal
from agentsphere.tool_registry import RiskLevel, ToolRegistry, ToolSpec


def echo_handler(arguments: Mapping[str, Any]) -> str:
    return str(arguments.get("text", ""))


@pytest.fixture
def echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo",
        version="1.0.0",
        owner="platform",
        description="Echo text back",
        json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        risk_level=RiskLevel.LOW,
        idempotent=True,
        handler=echo_handler,
    )


@pytest.fixture
def registry(echo_spec: ToolSpec) -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(echo_spec)
    return tools


@pytest.fixture
def principal() -> Principal:
    return Principal(name="alice", tenant="acme", owner="alice", team="platform")


@pytest.fixture
def other_tenant() -> Principal:
    return Principal(name="bob", tenant="other", owner="bob", team="ops")
