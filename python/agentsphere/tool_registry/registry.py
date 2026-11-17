"""Tool Registry v1: register, lookup by name, version pin. Discovery is exact-name, not semantic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentsphere.model_adapter.types import ToolDefinition
from agentsphere.tool_registry.spec import ToolSpec


class ToolNotFoundError(KeyError):
    pass


class ToolAlreadyRegisteredError(ValueError):
    pass


class BoundTool:
    """Executable view consumed by the Kernel ToolDispatcher."""

    def __init__(self, spec: ToolSpec) -> None:
        self.name = spec.name
        self.version = spec.version
        self.parameter_schema = spec.json_schema
        self.idempotent = spec.idempotent
        self.spec = spec

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        if self.spec.handler is None:
            raise RuntimeError(f"tool {self.name}@{self.version} has no handler bound")
        return self.spec.handler(arguments)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[tuple[str, str], ToolSpec] = {}

    def register(self, spec: ToolSpec, *, replace: bool = False) -> ToolSpec:
        key = (spec.name, spec.version)
        if key in self._tools and not replace:
            raise ToolAlreadyRegisteredError(f"{spec.name}@{spec.version} is already registered")
        self._tools[key] = spec
        return spec

    def get(self, name: str, version: str | None = None) -> ToolSpec:
        if version is not None:
            spec = self._tools.get((name, version))
            if spec is None:
                raise ToolNotFoundError(f"tool {name}@{version} not found")
            return spec
        versions = [spec for (n, _), spec in self._tools.items() if n == name]
        if not versions:
            raise ToolNotFoundError(f"tool {name} not found")
        return max(versions, key=lambda s: s.parsed_version)

    def resolve(self, name: str, version: str | None = None) -> BoundTool:
        return BoundTool(self.get(name, version))

    def list(self, name: str | None = None) -> list[ToolSpec]:
        specs = list(self._tools.values())
        if name is not None:
            specs = [s for s in specs if s.name == name]
        return sorted(specs, key=lambda s: (s.name, s.parsed_version))

    def list_definitions(self) -> list[ToolDefinition]:
        """Latest version of each tool, as model-facing definitions."""
        latest: dict[str, ToolSpec] = {}
        for spec in self._tools.values():
            current = latest.get(spec.name)
            if current is None or spec.parsed_version > current.parsed_version:
                latest[spec.name] = spec
        return [
            ToolDefinition(name=s.name, description=s.description, parameters=s.json_schema)
            for s in sorted(latest.values(), key=lambda s: s.name)
        ]
