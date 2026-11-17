"""In-memory Control Plane store. JSON dump/load stands in for a future SQL/etcd backend."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.controlplane.assets import AssetRecord, DependencyEdge
from agentsphere.controlplane.events import ControlEvent
from agentsphere.controlplane.release import DeploymentRecord
from agentsphere.controlplane.types import AssetKind, GateDecision, SLOSignal


def _asset_key(kind: AssetKind | str, tenant: str, name: str, version: str) -> tuple[str, str, str, str]:
    kind_value = kind.value if isinstance(kind, AssetKind) else kind
    return (kind_value, tenant, name, version)


class ControlStore:
    """Persistence port. A later Go control-plane service can speak the same operations."""

    def put_asset(self, record: AssetRecord) -> AssetRecord:
        raise NotImplementedError

    def get_asset(self, kind: AssetKind, tenant: str, name: str, version: str) -> AssetRecord | None:
        raise NotImplementedError

    def list_assets(
        self,
        *,
        kind: AssetKind | None = None,
        tenant: str | None = None,
        name: str | None = None,
    ) -> list[AssetRecord]:
        raise NotImplementedError

    def put_artifact(self, artifact: AgentArtifact) -> AgentArtifact:
        raise NotImplementedError

    def get_artifact(self, tenant: str, name: str, version: str) -> AgentArtifact | None:
        raise NotImplementedError

    def list_artifacts(self, *, tenant: str | None = None, name: str | None = None) -> list[AgentArtifact]:
        raise NotImplementedError

    def put_deployment(self, record: DeploymentRecord) -> DeploymentRecord:
        raise NotImplementedError

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        raise NotImplementedError

    def list_deployments(
        self,
        *,
        tenant: str | None = None,
        name: str | None = None,
        environment: str | None = None,
    ) -> list[DeploymentRecord]:
        raise NotImplementedError

    def add_edge(self, edge: DependencyEdge) -> DependencyEdge:
        raise NotImplementedError

    def list_edges(
        self,
        *,
        source: tuple[str, str, str, str] | None = None,
        target: tuple[str, str, str, str] | None = None,
    ) -> list[DependencyEdge]:
        raise NotImplementedError

    def append_event(self, event: ControlEvent) -> ControlEvent:
        raise NotImplementedError

    def list_events(
        self,
        *,
        tenant: str | None = None,
        resource: str | None = None,
    ) -> list[ControlEvent]:
        raise NotImplementedError

    def put_gate(self, gate_id: str, decision: GateDecision) -> GateDecision:
        raise NotImplementedError

    def get_gate(self, gate_id: str) -> GateDecision | None:
        raise NotImplementedError

    def add_slo(self, signal: SLOSignal) -> SLOSignal:
        raise NotImplementedError

    def list_slo(self, *, deployment_id: str | None = None) -> list[SLOSignal]:
        raise NotImplementedError


class InMemoryControlStore(ControlStore):
    def __init__(self) -> None:
        self._assets: dict[tuple[str, str, str, str], AssetRecord] = {}
        self._artifacts: dict[tuple[str, str, str], AgentArtifact] = {}
        self._deployments: dict[str, DeploymentRecord] = {}
        self._edges: list[DependencyEdge] = []
        self._events: list[ControlEvent] = []
        self._gates: dict[str, GateDecision] = {}
        self._slo: list[SLOSignal] = []
        self._lock = threading.Lock()

    def put_asset(self, record: AssetRecord) -> AssetRecord:
        stored = record.model_copy(deep=True)
        with self._lock:
            self._assets[_asset_key(stored.kind, stored.tenant, stored.name, stored.version)] = stored
        return stored.model_copy(deep=True)

    def get_asset(self, kind: AssetKind, tenant: str, name: str, version: str) -> AssetRecord | None:
        item = self._assets.get(_asset_key(kind, tenant, name, version))
        return item.model_copy(deep=True) if item else None

    def list_assets(
        self,
        *,
        kind: AssetKind | None = None,
        tenant: str | None = None,
        name: str | None = None,
    ) -> list[AssetRecord]:
        items = [a.model_copy(deep=True) for a in self._assets.values()]
        if kind is not None:
            items = [a for a in items if a.kind is kind]
        if tenant is not None:
            items = [a for a in items if a.tenant == tenant]
        if name is not None:
            items = [a for a in items if a.name == name]
        return sorted(items, key=lambda a: (a.kind.value, a.name, a.version))

    def put_artifact(self, artifact: AgentArtifact) -> AgentArtifact:
        stored = artifact.model_copy(deep=True)
        key = (stored.metadata.tenant, stored.metadata.name, stored.version)
        with self._lock:
            self._artifacts[key] = stored
        return stored.model_copy(deep=True)

    def get_artifact(self, tenant: str, name: str, version: str) -> AgentArtifact | None:
        item = self._artifacts.get((tenant, name, version))
        return item.model_copy(deep=True) if item else None

    def list_artifacts(self, *, tenant: str | None = None, name: str | None = None) -> list[AgentArtifact]:
        items = [a.model_copy(deep=True) for a in self._artifacts.values()]
        if tenant is not None:
            items = [a for a in items if a.metadata.tenant == tenant]
        if name is not None:
            items = [a for a in items if a.metadata.name == name]
        return sorted(items, key=lambda a: (a.metadata.name, a.version))

    def put_deployment(self, record: DeploymentRecord) -> DeploymentRecord:
        stored = record.model_copy(deep=True)
        with self._lock:
            self._deployments[stored.deployment_id] = stored
        return stored.model_copy(deep=True)

    def get_deployment(self, deployment_id: str) -> DeploymentRecord | None:
        item = self._deployments.get(deployment_id)
        return item.model_copy(deep=True) if item else None

    def list_deployments(
        self,
        *,
        tenant: str | None = None,
        name: str | None = None,
        environment: str | None = None,
    ) -> list[DeploymentRecord]:
        items = [d.model_copy(deep=True) for d in self._deployments.values()]
        if tenant is not None:
            items = [d for d in items if d.tenant == tenant]
        if name is not None:
            items = [d for d in items if d.agent_name == name]
        if environment is not None:
            items = [d for d in items if d.environment == environment]
        return sorted(items, key=lambda d: d.created_at)

    def add_edge(self, edge: DependencyEdge) -> DependencyEdge:
        stored = edge.model_copy(deep=True)
        with self._lock:
            self._edges.append(stored)
        return stored.model_copy(deep=True)

    def list_edges(
        self,
        *,
        source: tuple[str, str, str, str] | None = None,
        target: tuple[str, str, str, str] | None = None,
    ) -> list[DependencyEdge]:
        items = [e.model_copy(deep=True) for e in self._edges]
        if source is not None:
            items = [e for e in items if e.source.exact_key() == source]
        if target is not None:
            items = [e for e in items if e.target.exact_key() == target]
        return items

    def append_event(self, event: ControlEvent) -> ControlEvent:
        stored = event.model_copy(deep=True)
        with self._lock:
            self._events.append(stored)
        return stored.model_copy(deep=True)

    def list_events(
        self,
        *,
        tenant: str | None = None,
        resource: str | None = None,
    ) -> list[ControlEvent]:
        items = [e.model_copy(deep=True) for e in self._events]
        if tenant is not None:
            items = [e for e in items if e.tenant == tenant]
        if resource is not None:
            items = [e for e in items if e.resource == resource]
        return items

    def put_gate(self, gate_id: str, decision: GateDecision) -> GateDecision:
        stored = decision.model_copy(deep=True)
        with self._lock:
            self._gates[gate_id] = stored
        return stored.model_copy(deep=True)

    def get_gate(self, gate_id: str) -> GateDecision | None:
        item = self._gates.get(gate_id)
        return item.model_copy(deep=True) if item else None

    def add_slo(self, signal: SLOSignal) -> SLOSignal:
        stored = signal.model_copy(deep=True)
        with self._lock:
            self._slo.append(stored)
        return stored.model_copy(deep=True)

    def list_slo(self, *, deployment_id: str | None = None) -> list[SLOSignal]:
        items = [s.model_copy(deep=True) for s in self._slo]
        if deployment_id is not None:
            items = [s for s in items if s.deployment_id == deployment_id]
        return items

    def dump(self) -> dict[str, Any]:
        return {
            "assets": [a.model_dump(mode="json") for a in self._assets.values()],
            "artifacts": [a.model_dump(mode="json") for a in self._artifacts.values()],
            "deployments": [d.model_dump(mode="json") for d in self._deployments.values()],
            "edges": [e.model_dump(mode="json") for e in self._edges],
            "events": [e.model_dump(mode="json") for e in self._events],
            "gates": {k: v.model_dump(mode="json") for k, v in self._gates.items()},
            "slo": [s.model_dump(mode="json") for s in self._slo],
        }

    @classmethod
    def load(cls, payload: dict[str, Any]) -> InMemoryControlStore:
        store = cls()
        for item in payload.get("assets") or []:
            store.put_asset(AssetRecord.model_validate(item))
        for item in payload.get("artifacts") or []:
            store.put_artifact(AgentArtifact.model_validate(item))
        for item in payload.get("deployments") or []:
            store.put_deployment(DeploymentRecord.model_validate(item))
        for item in payload.get("edges") or []:
            store.add_edge(DependencyEdge.model_validate(item))
        for item in payload.get("events") or []:
            store.append_event(ControlEvent.model_validate(item))
        for gate_id, item in (payload.get("gates") or {}).items():
            store.put_gate(gate_id, GateDecision.model_validate(item))
        for item in payload.get("slo") or []:
            store.add_slo(SLOSignal.model_validate(item))
        return store

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.dump(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> InMemoryControlStore:
        return cls.load(json.loads(Path(path).read_text(encoding="utf-8")))
