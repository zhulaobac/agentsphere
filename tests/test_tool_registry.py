from __future__ import annotations

import pytest

from agentsphere.tool_registry import (
    RiskLevel,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
)


def _spec(version: str, *, handler=None) -> ToolSpec:
    return ToolSpec(
        name="search",
        version=version,
        owner="search-team",
        description="Search",
        json_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        risk_level=RiskLevel.MEDIUM,
        idempotent=True,
        handler=handler or (lambda args: args["q"]),
    )


def test_register_and_lookup_latest() -> None:
    registry = ToolRegistry()
    registry.register(_spec("1.0.0"))
    registry.register(_spec("1.2.0"))
    registry.register(_spec("1.1.0"))
    latest = registry.get("search")
    assert latest.version == "1.2.0"
    assert latest.owner == "search-team"
    assert latest.risk_level is RiskLevel.MEDIUM
    assert latest.idempotent is True


def test_version_pin() -> None:
    registry = ToolRegistry()
    registry.register(_spec("1.0.0"))
    registry.register(_spec("2.0.0"))
    pinned = registry.get("search", version="1.0.0")
    assert pinned.version == "1.0.0"


def test_unknown_tool_and_duplicate() -> None:
    registry = ToolRegistry()
    registry.register(_spec("1.0.0"))
    with pytest.raises(ToolNotFoundError):
        registry.get("missing")
    with pytest.raises(ToolNotFoundError):
        registry.get("search", version="9.9.9")
    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(_spec("1.0.0"))


def test_rejects_non_semver() -> None:
    with pytest.raises(ValueError, match="SemVer"):
        ToolSpec(name="x", version="nonsemver", owner="o")


def test_list_definitions_uses_latest() -> None:
    registry = ToolRegistry()
    registry.register(_spec("1.0.0"))
    registry.register(_spec("1.1.0"))
    defs = registry.list_definitions()
    assert len(defs) == 1
    assert defs[0].name == "search"


def test_resolve_invokes_handler() -> None:
    registry = ToolRegistry()
    registry.register(_spec("1.0.0", handler=lambda args: f"hit:{args['q']}"))
    bound = registry.resolve("search", "1.0.0")
    assert bound.invoke({"q": "abc"}) == "hit:abc"
    assert bound.idempotent is True
