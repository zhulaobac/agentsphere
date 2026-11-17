"""Versioned evaluation datasets with slice tags. Sources are labels, not an ingest pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentsphere.controlplane.assets import AssetRecord, DependencyEdge
from agentsphere.controlplane.errors import AssetNotFoundError, ImmutableAssetError
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import AssetKind, AssetMetadata, AssetRef, Principal, require_semver
from agentsphere.evaluation.types import DatasetCase


class DatasetVersion(BaseModel):
    name: str
    version: str
    owner: str
    tenant: str
    team: str = ""
    labels: dict[str, str] = Field(default_factory=dict)
    cases: list[DatasetCase] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("version")
    @classmethod
    def _semver(cls, value: str) -> str:
        return require_semver(value)

    def metadata(self) -> AssetMetadata:
        return AssetMetadata(
            name=self.name,
            owner=self.owner,
            tenant=self.tenant,
            team=self.team,
            labels=self.labels,
        )

    def spec(self) -> dict[str, Any]:
        return {
            "cases": [c.model_dump(mode="json") for c in self.cases],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_asset(cls, record: AssetRecord) -> DatasetVersion:
        return cls(
            name=record.name,
            version=record.version,
            owner=record.owner,
            tenant=record.tenant,
            team=record.metadata.team,
            labels=dict(record.labels),
            cases=[DatasetCase.model_validate(c) for c in record.spec.get("cases") or []],
        )


class DatasetService:
    def __init__(self, control: ControlPlane) -> None:
        self.control = control

    async def create(
        self,
        *,
        name: str,
        version: str,
        principal: Principal,
        cases: list[DatasetCase],
        labels: dict[str, str] | None = None,
        team: str = "",
    ) -> DatasetVersion:
        dataset = DatasetVersion(
            name=name,
            version=version,
            owner=principal.owner,
            tenant=principal.tenant,
            team=team or principal.team,
            labels=labels or {},
            cases=list(cases),
        )
        await self.control.put_asset(
            kind=AssetKind.DATASET,
            metadata=dataset.metadata(),
            version=dataset.version,
            spec=dataset.spec(),
            principal=principal,
        )
        self._record_source_edges(dataset, principal)
        return dataset

    async def update(
        self,
        *,
        name: str,
        version: str,
        principal: Principal,
        cases: list[DatasetCase] | None = None,
        labels: dict[str, str] | None = None,
    ) -> DatasetVersion:
        current = await self.get(name, version, principal)
        if self._pinned_by_experiment(name, version, principal.tenant):
            raise ImmutableAssetError(
                f"dataset {name}@{version} is pinned by an experiment; create a new version"
            )
        if cases is not None:
            current.cases = list(cases)
        if labels is not None:
            current.labels = dict(labels)
        await self.control.put_asset(
            kind=AssetKind.DATASET,
            metadata=current.metadata(),
            version=current.version,
            spec=current.spec(),
            principal=principal,
            replace=True,
        )
        return current

    async def get(self, name: str, version: str, principal: Principal) -> DatasetVersion:
        record = await self.control.get_asset(AssetKind.DATASET, name, version, principal)
        return DatasetVersion.from_asset(record)

    def _pinned_by_experiment(self, name: str, version: str, tenant: str) -> bool:
        store = getattr(self.control.store, "list_edges", None)
        if store is None:
            return False
        ref = AssetRef(kind=AssetKind.DATASET, name=name, version=version, tenant=tenant)
        used = self.control.store.list_edges(target=ref.exact_key())
        return any(e.relation == "evaluates" for e in used)

    def _record_source_edges(self, dataset: DatasetVersion, principal: Principal) -> None:
        source_label = dataset.labels.get("source")
        if not source_label:
            return
        origin = AssetRef(
            kind=AssetKind.DATASET,
            name=f"{dataset.name}-source",
            version=dataset.version,
            tenant=dataset.tenant,
        )
        self.control.store.add_edge(
            DependencyEdge(
                source=AssetRef(
                    kind=AssetKind.DATASET,
                    name=dataset.name,
                    version=dataset.version,
                    tenant=dataset.tenant,
                ),
                target=origin,
                relation=f"source:{source_label}",
                actor=principal.name,
            )
        )


class DatasetNotFoundError(AssetNotFoundError):
    pass
