"""Research input backends. CI and tests default to the fixture catalog (no network)."""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from agentsphere.tenants.product_team.catalog import resolve_entry


class LiveResearchNotConfigured(RuntimeError):
    """Live provider was requested but no fetch seam / endpoint is bound."""


@runtime_checkable
class ResearchSource(Protocol):
    """Seam for research / competitive inputs. Implementations must not be called from the model."""

    name: str

    def lookup_topic(self, topic: str) -> dict[str, Any]: ...

    def list_offerings(self, category: str) -> dict[str, Any]: ...

    def compare_offerings(self, topic: str, names: list[str] | None = None) -> dict[str, Any]: ...


class FixtureResearchSource:
    """Scripted catalog. Safe for CI; never opens a socket."""

    name = "fixture"

    def lookup_topic(self, topic: str) -> dict[str, Any]:
        entry = resolve_entry(topic)
        return {
            "topic": topic,
            "category": entry["category"],
            "problem_context": entry["problem_context"],
            "user_need_findings": list(entry["user_need_findings"]),
            "recommendations": list(entry["recommendations"]),
            "sources": list(entry["sources"]),
            "backend": self.name,
        }

    def list_offerings(self, category: str) -> dict[str, Any]:
        entry = resolve_entry(category)
        return {
            "category": entry["category"],
            "topic": category,
            "offerings": [dict(row) for row in entry["offerings"]],
            "backend": self.name,
        }

    def compare_offerings(self, topic: str, names: list[str] | None = None) -> dict[str, Any]:
        entry = resolve_entry(topic)
        rows = [dict(row) for row in entry["offerings"]]
        if names:
            wanted = {n.strip().lower() for n in names if n}
            filtered = [row for row in rows if row["name"].lower() in wanted]
            if filtered:
                rows = filtered
        return {
            "topic": topic,
            "rows": rows,
            "recommendations": list(entry["recommendations"]),
            "backend": self.name,
        }


class LiveResearchSource:
    """Optional live provider. Tests must not instantiate this without an injected fetch.

    No HTTP client is bundled. Bind ``fetch`` in a real deployment; otherwise this
    class refuses to run so CI cannot accidentally call a paid search API.
    """

    name = "live"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        fetch: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._fetch = fetch
        if not endpoint and fetch is None:
            raise LiveResearchNotConfigured(
                "live research source requires endpoint or an injected fetch seam"
            )

    @classmethod
    def from_env(cls) -> LiveResearchSource:
        endpoint = os.environ.get("AGENTSPHERE_RESEARCH_ENDPOINT")
        if not endpoint:
            raise LiveResearchNotConfigured(
                "AGENTSPHERE_RESEARCH_ENDPOINT is unset; use the fixture backend"
            )
        return cls(endpoint=endpoint)

    def lookup_topic(self, topic: str) -> dict[str, Any]:
        return self._call("lookup_topic", {"topic": topic})

    def list_offerings(self, category: str) -> dict[str, Any]:
        return self._call("list_offerings", {"category": category})

    def compare_offerings(self, topic: str, names: list[str] | None = None) -> dict[str, Any]:
        return self._call("compare_offerings", {"topic": topic, "names": names or []})

    def _call(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._fetch is None:
            raise LiveResearchNotConfigured(
                "live provider has no fetch seam bound; tests must use the fixture backend"
            )
        return self._fetch(operation, payload, endpoint=self.endpoint)


def resolve_research_source(kind: str | None = None) -> ResearchSource:
    """Default is fixture. ``live`` stays behind this seam and is off in CI."""

    selected = (kind or os.environ.get("AGENTSPHERE_RESEARCH_SOURCE") or "fixture").strip().lower()
    if selected in {"fixture", "catalog", "offline", ""}:
        return FixtureResearchSource()
    if selected == "live":
        return LiveResearchSource.from_env()
    raise ValueError(f"unknown research source {selected!r}; use 'fixture' or 'live'")
