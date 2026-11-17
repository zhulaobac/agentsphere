"""Control Plane façade: Artifact Registry, lifecycle, release, tenant scoping, lineage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from packaging.version import Version

from agentsphere.controlplane.artifact import AgentArtifact, AgentStatus, DeployedVersion
from agentsphere.controlplane.assets import STUB_KINDS, AssetRecord, DependencyEdge
from agentsphere.controlplane.auth import assert_can_mutate, assert_can_read, assert_same_tenant
from agentsphere.controlplane.errors import (
    ArtifactNotFoundError,
    AssetNotFoundError,
    AuthorizationError,
    DependencyError,
    DuplicateAssetError,
    GateFailedError,
    ImmutableAssetError,
    LifecycleError,
)
from agentsphere.controlplane.events import ControlEvent, ControlEventType
from agentsphere.controlplane.gate import derived_passed, verify_gate_decision
from agentsphere.controlplane.lifecycle import assert_can_release, assert_transition_allowed
from agentsphere.controlplane.release import CANARY_WEIGHT, FULL_WEIGHT, DeploymentRecord
from agentsphere.controlplane.runtime import ControlRuntimeProvider, DeploymentBinding
from agentsphere.controlplane.store import ControlStore, InMemoryControlStore
from agentsphere.controlplane.types import (
    AssetKind,
    AssetMetadata,
    AssetRef,
    DeploymentStatus,
    GateDecision,
    GateThresholds,
    LifecyclePhase,
    Principal,
    ReleaseChannel,
    SLOSignal,
)
from agentsphere.kernel.ids import new_id


def _as_release_channel(channel: Any) -> ReleaseChannel:
    if isinstance(channel, ReleaseChannel):
        return channel
    try:
        return ReleaseChannel(channel)
    except (TypeError, ValueError) as exc:
        raise LifecycleError(f"invalid release channel {channel!r}") from exc


def _snapshot_experiment_run(run: Any) -> Any:
    from agentsphere.evaluation.experiment import ExperimentRun

    try:
        if isinstance(run, ExperimentRun):
            stored = run.model_copy(deep=True)
        elif hasattr(run, "model_dump"):
            stored = ExperimentRun.model_validate(run.model_dump())
        else:
            stored = ExperimentRun.model_validate(run, from_attributes=True)
    except (TypeError, ValueError) as exc:
        raise LifecycleError("experiment run must be a snapshotable ExperimentRun") from exc
    return stored.model_copy(deep=True)


class ControlPlane:
    def __init__(
        self,
        store: ControlStore | None = None,
        *,
        runtime: ControlRuntimeProvider | None = None,
    ) -> None:
        self.store = store or InMemoryControlStore()
        self.runtime = runtime
        self._experiments: dict[str, Any] = {}

    def register_experiment(self, run: Any, *, tenant: str | None = None) -> Any:
        """Index a stored ExperimentRun so evaluation and Gate can look it up by id."""
        run_id = getattr(run, "run_id", None)
        if not run_id:
            raise LifecycleError("experiment run is missing run_id")
        key = str(run_id)
        if key in self._experiments:
            raise LifecycleError(f"experiment run {key} is immutable")
        stored = _snapshot_experiment_run(run)
        spec = getattr(stored, "spec", None)
        bound_tenant = tenant or (getattr(spec, "tenant", None) or "")
        if not bound_tenant:
            raise LifecycleError("experiment run is missing tenant")
        if spec is not None and getattr(spec, "tenant", None) != bound_tenant:
            stored = stored.model_copy(
                update={"spec": spec.model_copy(update={"tenant": bound_tenant})},
                deep=True,
            )
        self._experiments[key] = stored
        return stored.model_copy(deep=True)

    def get_experiment(self, run_id: str) -> Any | None:
        item = self._experiments.get(run_id)
        if item is None:
            return None
        return item.model_copy(deep=True)

    async def put_asset(
        self,
        *,
        kind: AssetKind,
        metadata: AssetMetadata,
        version: str,
        spec: dict[str, Any] | None = None,
        principal: Principal,
        replace: bool = False,
    ) -> AssetRecord:
        resource = f"{kind.value}/{metadata.name}@{version}"
        existing = self.store.get_asset(kind, metadata.tenant, metadata.name, version)
        if existing is None:
            assert_can_mutate(principal, metadata, action="create", resource=resource)
        else:
            if existing.immutable or self._is_experiment_pinned(existing):
                raise ImmutableAssetError(f"{kind.value}/{metadata.name}@{version} is immutable")
            if not replace:
                raise DuplicateAssetError(f"{kind.value}/{metadata.name}@{version} already exists")
            assert_can_mutate(principal, existing.metadata, action="update", resource=resource)
        record = AssetRecord(
            kind=kind,
            metadata=metadata,
            version=version,
            spec=spec or {},
            stub=kind in STUB_KINDS,
            updated_at=datetime.now(timezone.utc),
            created_at=existing.created_at if existing is not None else datetime.now(timezone.utc),
        )
        stored = self.store.put_asset(record)
        await self._emit(
            ControlEventType.ASSET_UPDATED if existing is not None else ControlEventType.ASSET_CREATED,
            principal,
            resource=f"{kind.value}/{metadata.name}@{version}",
            tenant=metadata.tenant,
            details={"kind": kind.value, "replace": replace, "stub": stored.stub},
        )
        return stored

    async def get_asset(
        self,
        kind: AssetKind,
        name: str,
        version: str,
        principal: Principal,
        *,
        tenant: str | None = None,
    ) -> AssetRecord:
        tenant_id = tenant or principal.tenant
        record = self.store.get_asset(kind, tenant_id, name, version)
        if record is None:
            raise AssetNotFoundError(f"{kind.value}/{name}@{version} not found")
        assert_can_read(principal, record.metadata, resource=f"{kind.value}/{name}@{version}")
        return record

    async def list_assets(
        self,
        principal: Principal,
        *,
        kind: AssetKind | None = None,
        name: str | None = None,
    ) -> list[AssetRecord]:
        return self.store.list_assets(kind=kind, tenant=principal.tenant, name=name)

    async def put_artifact(self, artifact: AgentArtifact, principal: Principal) -> AgentArtifact:
        meta = artifact.metadata
        resource = artifact.resource_key()
        existing = self.store.get_artifact(meta.tenant, meta.name, artifact.version)
        if existing is not None:
            if existing.status.phase is not LifecyclePhase.DRAFT:
                raise ImmutableAssetError(
                    f"artifact {resource} is {existing.status.phase.value}; create a new version"
                )
            assert_can_mutate(principal, existing.metadata, action="update", resource=resource)
            artifact = artifact.model_copy(
                update={"status": existing.status.model_copy(deep=True)},
                deep=True,
            )
            event_type = ControlEventType.ARTIFACT_UPDATED
        else:
            assert_can_mutate(principal, meta, action="create", resource=resource)
            artifact = artifact.model_copy(
                update={"status": AgentStatus(phase=LifecyclePhase.DRAFT)},
                deep=True,
            )
            event_type = ControlEventType.ARTIFACT_CREATED

        stored = self.store.put_artifact(artifact)
        await self.put_asset(
            kind=AssetKind.AGENT,
            metadata=stored.metadata,
            version=stored.version,
            spec=stored.spec.model_dump(mode="json"),
            principal=principal,
            replace=True,
        )
        await self._record_lineage(stored, principal)
        await self._emit(
            event_type,
            principal,
            resource=resource,
            tenant=meta.tenant,
            details={"phase": stored.status.phase.value},
        )
        return stored

    async def get_artifact(self, name: str, version: str, principal: Principal) -> AgentArtifact:
        artifact = self.store.get_artifact(principal.tenant, name, version)
        if artifact is None:
            raise ArtifactNotFoundError(f"{principal.tenant}/{name}@{version} not found")
        assert_can_read(principal, artifact.metadata, resource=artifact.resource_key())
        return artifact

    async def list_artifacts(self, principal: Principal, *, name: str | None = None) -> list[AgentArtifact]:
        return self.store.list_artifacts(tenant=principal.tenant, name=name)

    async def build(self, name: str, version: str, principal: Principal) -> AgentArtifact:
        artifact = await self._require_mutable(name, version, principal, action="build")
        assert_transition_allowed(artifact.status.phase, LifecyclePhase.BUILT)
        missing = await self._missing_dependencies(artifact, principal)
        if missing:
            raise DependencyError(
                "cannot Build; missing dependencies: " + ", ".join(r.display() for r in missing)
            )
        return await self._transition(artifact, LifecyclePhase.BUILT, principal, details={"resolved": True})

    async def record_evaluation(
        self,
        name: str,
        version: str,
        principal: Principal,
        *,
        experiment_run_id: str,
    ) -> AgentArtifact:
        artifact = await self._require_mutable(name, version, principal, action="evaluate")
        assert_transition_allowed(artifact.status.phase, LifecyclePhase.EVALUATED)
        if not experiment_run_id:
            raise LifecycleError("cannot mark Evaluated without an experiment run id")
        run = self.get_experiment(experiment_run_id)
        if run is None:
            raise LifecycleError(f"unknown experiment run {experiment_run_id}")
        spec = getattr(run, "spec", None)
        if spec is not None and (spec.agent_name != name or spec.agent_version != version):
            raise LifecycleError(
                f"experiment run {experiment_run_id} does not belong to {name}@{version}"
            )
        self._assert_experiment_tenant(run, artifact.metadata.tenant, experiment_run_id)
        artifact.status.lastExperimentRunId = experiment_run_id
        await self._emit(
            ControlEventType.EVALUATION_RECORDED,
            principal,
            resource=artifact.resource_key(),
            tenant=artifact.metadata.tenant,
            details={"experiment_run_id": experiment_run_id},
        )
        return await self._transition(
            artifact,
            LifecyclePhase.EVALUATED,
            principal,
            details={"experiment_run_id": experiment_run_id},
        )

    async def approve(self, name: str, version: str, principal: Principal, *, reason: str = "") -> AgentArtifact:
        artifact = await self._require_mutable(name, version, principal, action="approve")
        assert_transition_allowed(artifact.status.phase, LifecyclePhase.APPROVED)
        await self._emit(
            ControlEventType.APPROVAL_RECORDED,
            principal,
            resource=artifact.resource_key(),
            tenant=artifact.metadata.tenant,
            details={"reason": reason},
        )
        return await self._transition(artifact, LifecyclePhase.APPROVED, principal, details={"reason": reason})

    async def record_gate(
        self,
        name: str,
        version: str,
        principal: Principal,
        *,
        experiment_run_id: str | None = None,
        thresholds: GateThresholds | None = None,
        baseline_run_id: str | None = None,
    ) -> tuple[AgentArtifact, GateDecision]:
        """Persist a GateDecision produced only by ReleaseGate.evaluate on a stored run."""
        artifact = await self._require_mutable(name, version, principal, action="record_gate")
        run_id = experiment_run_id or artifact.status.lastExperimentRunId
        if not run_id:
            raise LifecycleError("cannot record a Gate without an experiment run id")
        run = self.get_experiment(run_id)
        if run is None:
            raise LifecycleError(f"unknown experiment run {run_id}")
        spec = getattr(run, "spec", None)
        if spec is not None and (spec.agent_name != name or spec.agent_version != version):
            raise LifecycleError(f"experiment run {run_id} does not belong to {name}@{version}")
        self._assert_experiment_tenant(run, artifact.metadata.tenant, run_id)
        if not artifact.status.lastExperimentRunId:
            raise LifecycleError("cannot record a Gate before evaluation")
        if run_id != artifact.status.lastExperimentRunId:
            raise LifecycleError(
                "Gate experiment run does not match the recorded evaluation"
            )
        baseline = self.get_experiment(baseline_run_id) if baseline_run_id else None
        from agentsphere.evaluation.gate import ReleaseGate

        decision = ReleaseGate().evaluate(run, thresholds=thresholds, baseline=baseline)
        if decision.passed != derived_passed(decision):
            raise GateFailedError(
                "inconsistent GateDecision: passed does not match non-skipped checks",
                decision=decision,
            )
        gate_id = new_id("gate")
        self.store.put_gate(gate_id, decision)
        artifact.status.lastGateResultId = gate_id
        stored_artifact = self.store.put_artifact(artifact)
        await self._sync_agent_asset(stored_artifact, principal)
        await self._emit(
            ControlEventType.GATE_RECORDED,
            principal,
            resource=stored_artifact.resource_key(),
            tenant=stored_artifact.metadata.tenant,
            details={
                "gate_id": gate_id,
                "passed": decision.passed,
                "experiment_run_id": decision.experiment_run_id,
            },
        )
        return stored_artifact, decision

    async def release(
        self,
        name: str,
        version: str,
        principal: Principal,
        gate: GateDecision | None = None,
        *,
        gate_id: str | None = None,
        channel: ReleaseChannel = ReleaseChannel.CANARY,
        weight: float | None = None,
        environment: str = "default",
        reason: str = "",
    ) -> tuple[AgentArtifact, DeploymentRecord]:
        artifact = await self._require_mutable(name, version, principal, action="release")
        if artifact.status.phase is not LifecyclePhase.APPROVED:
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={
                    "phase": artifact.status.phase.value,
                    "gate_passed": bool(gate and gate.passed),
                    "reason": "gate_or_phase",
                },
            )
            raise LifecycleError(f"cannot Release from phase {artifact.status.phase.value}; must be Approved")
        if gate is not None and gate.passed != derived_passed(gate):
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={"reason": "inconsistent_gate", "gate_passed": gate.passed},
            )
            raise GateFailedError(
                "inconsistent GateDecision: passed does not match non-skipped checks",
                decision=gate,
            )
        try:
            stored_gate = self._load_stored_gate(artifact, gate_id=gate_id)
            assert_can_release(artifact.status.phase, stored_gate)
        except (LifecycleError, GateFailedError) as exc:
            decision = getattr(exc, "decision", None)
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={
                    "phase": artifact.status.phase.value,
                    "gate_passed": bool(decision.passed) if decision is not None else False,
                    "reason": str(exc),
                },
            )
            raise

        verified = verify_gate_decision(stored_gate)
        try:
            channel = _as_release_channel(channel)
        except LifecycleError:
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={"reason": "invalid_channel", "channel": str(channel)},
            )
            raise
        if channel is not ReleaseChannel.CANARY:
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={"reason": "canary_required", "channel": channel.value},
            )
            raise LifecycleError(
                "cannot Release to Full; promote an active Canary instead"
            )
        if weight is not None and weight >= 1.0:
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={"reason": "canary_weight", "weight": weight},
            )
            raise LifecycleError("canary weight must be < 1.0; promote for Full")
        if verified.experiment_run_id != artifact.status.lastExperimentRunId:
            await self._emit(
                ControlEventType.RELEASE_BLOCKED,
                principal,
                resource=artifact.resource_key(),
                tenant=artifact.metadata.tenant,
                details={
                    "reason": "stale_gate",
                    "gate_run": verified.experiment_run_id,
                    "artifact_run": artifact.status.lastExperimentRunId,
                },
            )
            raise GateFailedError(
                "Gate decision is not for this artifact version's experiment run",
                decision=verified,
            )

        desired_weight = CANARY_WEIGHT if weight is None else weight
        for existing in self.store.list_deployments(
            tenant=artifact.metadata.tenant, name=name, environment=environment
        ):
            if existing.status is DeploymentStatus.ACTIVE and existing.agent_version != version:
                if channel is ReleaseChannel.FULL:
                    existing.status = DeploymentStatus.SUPERSEDED
                    existing.updated_at = datetime.now(timezone.utc)
                    self.store.put_deployment(existing)

        record = DeploymentRecord(
            tenant=artifact.metadata.tenant,
            agent_name=name,
            agent_version=version,
            environment=environment,
            channel=channel,
            weight=desired_weight,
            reason=reason,
        )
        stored_dep = self.store.put_deployment(record)
        deployed = DeployedVersion(
            version=version,
            environment=environment,
            channel=channel,
            weight=desired_weight,
            deployment_id=stored_dep.deployment_id,
            status=DeploymentStatus.ACTIVE,
        )
        artifact.status.deployedVersions = [
            d for d in artifact.status.deployedVersions if d.deployment_id != stored_dep.deployment_id
        ]
        artifact.status.deployedVersions.append(deployed)
        artifact.status.releaseChannel = channel
        artifact = await self._transition(
            artifact,
            LifecyclePhase.RELEASED,
            principal,
            details={
                "channel": channel.value,
                "weight": desired_weight,
                "deployment_id": stored_dep.deployment_id,
            },
        )
        event_type = (
            ControlEventType.RELEASE_CANARY if channel is ReleaseChannel.CANARY else ControlEventType.RELEASE_FULL
        )
        await self._emit(
            event_type,
            principal,
            resource=artifact.resource_key(),
            tenant=artifact.metadata.tenant,
            details={"deployment_id": stored_dep.deployment_id, "weight": desired_weight},
        )
        await self._bind_runtime(artifact, stored_dep, principal)
        return artifact, stored_dep

    async def promote(
        self,
        name: str,
        version: str,
        principal: Principal,
        *,
        environment: str = "default",
    ) -> tuple[AgentArtifact, DeploymentRecord]:
        artifact = await self._require_mutable(name, version, principal, action="promote")
        if artifact.status.phase is not LifecyclePhase.RELEASED:
            raise LifecycleError(f"cannot promote from phase {artifact.status.phase.value}")
        active = [
            d
            for d in self.store.list_deployments(
                tenant=artifact.metadata.tenant, name=name, environment=environment
            )
            if d.status is DeploymentStatus.ACTIVE and d.agent_version == version
        ]
        if not active:
            raise LifecycleError(f"no active canary deployment for {name}@{version}")
        current = active[-1]
        if current.channel is ReleaseChannel.FULL:
            raise LifecycleError(f"{name}@{version} is already Full in {environment}")
        if current.channel is not ReleaseChannel.CANARY:
            raise LifecycleError("promote requires an active Canary deployment")

        for other in self.store.list_deployments(
            tenant=artifact.metadata.tenant, name=name, environment=environment
        ):
            if other.deployment_id != current.deployment_id and other.status is DeploymentStatus.ACTIVE:
                other.status = DeploymentStatus.SUPERSEDED
                other.updated_at = datetime.now(timezone.utc)
                self.store.put_deployment(other)

        current.channel = ReleaseChannel.FULL
        current.weight = FULL_WEIGHT
        current.updated_at = datetime.now(timezone.utc)
        stored = self.store.put_deployment(current)
        artifact.status.releaseChannel = ReleaseChannel.FULL
        artifact.status.deployedVersions = [
            DeployedVersion(
                version=d.version,
                environment=d.environment,
                channel=stored.channel if d.deployment_id == stored.deployment_id else d.channel,
                weight=stored.weight if d.deployment_id == stored.deployment_id else d.weight,
                deployment_id=d.deployment_id,
                status=d.status,
            )
            for d in artifact.status.deployedVersions
        ]
        for item in artifact.status.deployedVersions:
            if item.deployment_id == stored.deployment_id:
                item.channel = ReleaseChannel.FULL
                item.weight = FULL_WEIGHT
        self.store.put_artifact(artifact)
        await self._sync_agent_asset(artifact, principal)
        await self._emit(
            ControlEventType.RELEASE_FULL,
            principal,
            resource=artifact.resource_key(),
            tenant=artifact.metadata.tenant,
            details={"deployment_id": stored.deployment_id, "promoted_from": "Canary"},
        )
        await self._bind_runtime(artifact, stored, principal)
        return artifact, stored

    async def deprecate(self, name: str, version: str, principal: Principal, *, reason: str = "") -> AgentArtifact:
        artifact = await self._require_mutable(name, version, principal, action="deprecate")
        assert_transition_allowed(artifact.status.phase, LifecyclePhase.DEPRECATED)
        return await self._transition(artifact, LifecyclePhase.DEPRECATED, principal, details={"reason": reason})

    async def retire(self, name: str, version: str, principal: Principal, *, reason: str = "") -> AgentArtifact:
        artifact = await self._require_mutable(name, version, principal, action="retire")
        assert_transition_allowed(artifact.status.phase, LifecyclePhase.RETIRED)
        for dep in self.store.list_deployments(tenant=artifact.metadata.tenant, name=name):
            if dep.agent_version == version and dep.status is DeploymentStatus.ACTIVE:
                dep.status = DeploymentStatus.SUPERSEDED
                dep.updated_at = datetime.now(timezone.utc)
                self.store.put_deployment(dep)
                if self.runtime is not None:
                    await self.runtime.unbind_deployment(dep.deployment_id)
        return await self._transition(artifact, LifecyclePhase.RETIRED, principal, details={"reason": reason})

    async def rollback(
        self,
        name: str,
        principal: Principal,
        *,
        to_version: str,
        environment: str = "default",
        reason: str = "",
        automatic: bool = False,
        from_version: str | None = None,
    ) -> tuple[AgentArtifact, DeploymentRecord]:
        target = await self.get_artifact(name, to_version, principal)
        assert_can_mutate(principal, target.metadata, action="rollback", resource=target.resource_key())
        if target.status.phase not in {LifecyclePhase.RELEASED, LifecyclePhase.DEPRECATED}:
            raise LifecycleError(
                f"cannot rollback to {name}@{to_version} in phase {target.status.phase.value}"
            )

        current_active = [
            d
            for d in self.store.list_deployments(
                tenant=target.metadata.tenant, name=name, environment=environment
            )
            if d.status is DeploymentStatus.ACTIVE
        ]
        from_ver = from_version or (current_active[-1].agent_version if current_active else None)
        if current_active:
            latest_active = current_active[-1]
            if (
                latest_active.agent_version == to_version
                and latest_active.channel is ReleaseChannel.CANARY
            ):
                raise LifecycleError(
                    "cannot rollback the active canary of this version; promote instead"
                )
        restore_full = self._version_was_full(
            tenant=target.metadata.tenant,
            name=name,
            version=to_version,
            environment=environment,
        )
        restore_channel = ReleaseChannel.FULL if restore_full else ReleaseChannel.CANARY
        restore_weight = FULL_WEIGHT if restore_full else CANARY_WEIGHT
        for dep in current_active:
            dep.status = DeploymentStatus.ROLLED_BACK
            dep.updated_at = datetime.now(timezone.utc)
            dep.rolled_back_to = to_version
            dep.reason = reason
            self.store.put_deployment(dep)
            if self.runtime is not None:
                await self.runtime.unbind_deployment(dep.deployment_id)

        if from_ver and from_ver != to_version:
            failing = self.store.get_artifact(principal.tenant, name, from_ver)
            if failing is not None and failing.status.phase is LifecyclePhase.RELEASED:
                failing.status.phase = LifecyclePhase.DEPRECATED
                failing.status.releaseChannel = None
                for item in failing.status.deployedVersions:
                    if item.environment == environment:
                        item.status = DeploymentStatus.ROLLED_BACK
                self.store.put_artifact(failing)
                await self._sync_agent_asset(failing, principal)
                await self._emit(
                    ControlEventType.LIFECYCLE_TRANSITION,
                    principal,
                    resource=failing.resource_key(),
                    tenant=failing.metadata.tenant,
                    details={"from": LifecyclePhase.RELEASED.value, "to": LifecyclePhase.DEPRECATED.value, "reason": reason},
                )

        restored = DeploymentRecord(
            tenant=target.metadata.tenant,
            agent_name=name,
            agent_version=to_version,
            environment=environment,
            channel=restore_channel,
            weight=restore_weight,
            reason=reason,
            rolled_back_from=from_ver,
        )
        stored = self.store.put_deployment(restored)
        target.status.phase = LifecyclePhase.RELEASED
        if restore_full:
            target.status.releaseChannel = ReleaseChannel.FULL
        else:
            still_full = any(
                d.status is DeploymentStatus.ACTIVE and d.channel is ReleaseChannel.FULL
                for d in self.store.list_deployments(
                    tenant=target.metadata.tenant, name=name
                )
                if d.environment != environment
            )
            if not still_full:
                target.status.releaseChannel = restore_channel
        target.status.deployedVersions.append(
            DeployedVersion(
                version=to_version,
                environment=environment,
                channel=restore_channel,
                weight=restore_weight,
                deployment_id=stored.deployment_id,
                status=DeploymentStatus.ACTIVE,
            )
        )
        self.store.put_artifact(target)
        await self._sync_agent_asset(target, principal)
        event_type = (
            ControlEventType.RELEASE_AUTO_ROLLBACK if automatic else ControlEventType.RELEASE_ROLLBACK
        )
        await self._emit(
            event_type,
            principal,
            resource=target.resource_key(),
            tenant=target.metadata.tenant,
            details={
                "deployment_id": stored.deployment_id,
                "from_version": from_ver,
                "to_version": to_version,
                "reason": reason,
            },
        )
        await self._bind_runtime(target, stored, principal)
        return target, stored

    async def record_slo(
        self,
        signal: SLOSignal,
        principal: Principal,
        *,
        auto_rollback: bool = True,
        rollback_to: str | None = None,
    ) -> SLOSignal:
        deployment = self.store.get_deployment(signal.deployment_id)
        if deployment is None:
            raise AssetNotFoundError(f"deployment not found: {signal.deployment_id}")
        assert_same_tenant(
            principal, deployment.tenant, action="record_slo", resource=signal.deployment_id
        )
        if auto_rollback and signal.breached:
            current = self.store.get_artifact(
                deployment.tenant, deployment.agent_name, deployment.agent_version
            )
            if current is not None:
                assert_can_mutate(
                    principal,
                    current.metadata,
                    action="record_slo",
                    resource=current.resource_key(),
                )
        stored = self.store.add_slo(signal)
        await self._emit(
            ControlEventType.SLO_RECORDED,
            principal,
            resource=signal.deployment_id,
            tenant=deployment.tenant,
            details={
                "metric": signal.metric,
                "value": signal.value,
                "threshold": signal.threshold,
                "breached": stored.breached,
            },
        )
        if auto_rollback and stored.breached:
            previous = rollback_to or self._previous_released_version(
                tenant=deployment.tenant, name=deployment.agent_name, current=deployment.agent_version
            )
            if previous:
                await self.rollback(
                    deployment.agent_name,
                    principal,
                    to_version=previous,
                    environment=deployment.environment,
                    reason=f"auto-rollback: {signal.metric} {signal.value} vs {signal.threshold}",
                    automatic=True,
                    from_version=deployment.agent_version,
                )
        return stored

    def lineage(
        self, ref: AssetRef, principal: Principal, *, tenant: str | None = None
    ) -> dict[str, list[AssetRef]]:
        tenant_id = principal.tenant if tenant is None else tenant
        assert_same_tenant(principal, tenant_id, action="read", resource=ref.display())
        if ref.tenant is not None and ref.tenant != tenant_id:
            raise AuthorizationError(
                f"tenant {principal.tenant!r} cannot read {ref.display()} owned by tenant {ref.tenant!r}"
            )
        record = self.store.get_asset(ref.kind, tenant_id, ref.name, ref.version)
        if record is None:
            return {"uses": [], "used_by": []}
        assert_can_read(principal, record.metadata, resource=ref.display())
        scoped = ref.model_copy(update={"tenant": tenant_id})
        key = scoped.key(tenant_id)
        outgoing = [e.target for e in self.store.list_edges(source=key)]
        incoming = [e.source for e in self.store.list_edges(target=key)]
        return {"uses": outgoing, "used_by": incoming}

    def events_for(
        self, resource: str, principal: Principal, *, tenant: str | None = None
    ) -> list[ControlEvent]:
        tenant_id = tenant or principal.tenant
        assert_same_tenant(principal, tenant_id, action="read", resource=resource)
        return self.store.list_events(tenant=tenant_id, resource=resource)

    async def _require_mutable(
        self, name: str, version: str, principal: Principal, *, action: str
    ) -> AgentArtifact:
        artifact = await self.get_artifact(name, version, principal)
        assert_can_mutate(principal, artifact.metadata, action=action, resource=artifact.resource_key())
        return artifact

    async def _transition(
        self,
        artifact: AgentArtifact,
        target: LifecyclePhase,
        principal: Principal,
        *,
        details: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        previous = artifact.status.phase
        artifact.status.phase = target
        stored = self.store.put_artifact(artifact)
        await self._sync_agent_asset(stored, principal)
        await self._emit(
            ControlEventType.LIFECYCLE_TRANSITION,
            principal,
            resource=stored.resource_key(),
            tenant=stored.metadata.tenant,
            details={"from": previous.value, "to": target.value, **(details or {})},
        )
        return stored

    async def _sync_agent_asset(self, artifact: AgentArtifact, principal: Principal) -> None:
        del principal
        self.store.put_asset(
            AssetRecord(
                kind=AssetKind.AGENT,
                metadata=artifact.metadata,
                version=artifact.version,
                spec=artifact.spec.model_dump(mode="json"),
                immutable=artifact.status.phase is not LifecyclePhase.DRAFT,
            )
        )

    async def _record_lineage(self, artifact: AgentArtifact, principal: Principal) -> None:
        consumer_tenant = artifact.metadata.tenant
        source = AssetRef(
            kind=AssetKind.AGENT,
            name=artifact.metadata.name,
            version=artifact.version,
            tenant=consumer_tenant,
        )
        existing = {e.target.display() for e in self.store.list_edges(source=source.exact_key())}
        for ref in artifact.spec.dependency_refs():
            if ref.tenant is not None and ref.tenant != consumer_tenant:
                continue
            if ref.display() in existing:
                continue
            target = ref.model_copy(update={"tenant": consumer_tenant})
            edge = DependencyEdge(source=source, target=target, relation="uses", actor=principal.name)
            self.store.add_edge(edge)
            await self._emit(
                ControlEventType.LINEAGE_EDGE,
                principal,
                resource=artifact.resource_key(),
                tenant=consumer_tenant,
                details={"from": source.display(), "to": target.display(), "relation": "uses"},
            )

    async def _missing_dependencies(
        self, artifact: AgentArtifact, principal: Principal
    ) -> list[AssetRef]:
        missing: list[AssetRef] = []
        tenant = artifact.metadata.tenant
        for ref in artifact.spec.dependency_refs():
            if ref.tenant is not None and ref.tenant != tenant:
                raise AuthorizationError(
                    f"tenant {principal.tenant!r} cannot read {ref.display()} owned by tenant {ref.tenant!r}"
                )
            found = self.store.get_asset(ref.kind, tenant, ref.name, ref.version)
            if found is None:
                missing.append(ref)
                continue
            assert_can_read(principal, found.metadata, resource=ref.display())
        return missing

    async def _bind_runtime(
        self, artifact: AgentArtifact, deployment: DeploymentRecord, principal: Principal
    ) -> None:
        if self.runtime is None:
            return
        prompt = await self._system_prompt_for(artifact, principal)
        binding = DeploymentBinding(
            deployment_id=deployment.deployment_id,
            agent_name=deployment.agent_name,
            agent_version=deployment.agent_version,
            tenant=deployment.tenant,
            channel=deployment.channel,
            weight=deployment.weight,
            system_prompt=prompt,
            metadata=dict(deployment.metadata),
        )
        await self.runtime.bind_deployment(binding)

    async def _system_prompt_for(self, artifact: AgentArtifact, principal: Principal) -> str:
        parts: list[str] = []
        tenant = artifact.metadata.tenant
        for ref in artifact.spec.prompts:
            if ref.tenant is not None and ref.tenant != tenant:
                raise AuthorizationError(
                    f"tenant {principal.tenant!r} cannot read {ref.display()} owned by tenant {ref.tenant!r}"
                )
            record = self.store.get_asset(ref.kind, tenant, ref.name, ref.version)
            if record is None:
                continue
            assert_can_read(principal, record.metadata, resource=ref.display())
            text = record.spec.get("text") or record.spec.get("template") or ""
            if text:
                parts.append(str(text))
        return "\n\n".join(parts)

    def _previous_released_version(self, *, tenant: str, name: str, current: str) -> str | None:
        deployments = [
            d
            for d in self.store.list_deployments(tenant=tenant, name=name)
            if d.agent_version != current
            and d.channel is ReleaseChannel.FULL
            and d.status in {DeploymentStatus.ACTIVE, DeploymentStatus.SUPERSEDED, DeploymentStatus.ROLLED_BACK}
        ]
        if not deployments:
            artifacts = [
                a
                for a in self.store.list_artifacts(tenant=tenant, name=name)
                if a.version != current
                and a.status.phase in {LifecyclePhase.RELEASED, LifecyclePhase.DEPRECATED}
            ]
            if not artifacts:
                return None
            artifacts.sort(key=lambda a: Version(a.version))
            return artifacts[-1].version
        return deployments[-1].agent_version

    def _version_was_full(
        self, *, tenant: str, name: str, version: str, environment: str
    ) -> bool:
        return any(
            d.agent_version == version and d.channel is ReleaseChannel.FULL
            for d in self.store.list_deployments(
                tenant=tenant, name=name, environment=environment
            )
        )

    def _assert_experiment_tenant(self, run: Any, tenant: str, run_id: str) -> None:
        spec = getattr(run, "spec", None)
        run_tenant = getattr(spec, "tenant", None) or ""
        if run_tenant != tenant:
            raise AuthorizationError(
                f"tenant {tenant!r} cannot use experiment run {run_id} owned by tenant {run_tenant!r}"
            )

    def _load_stored_gate(self, artifact: AgentArtifact, *, gate_id: str | None) -> GateDecision:
        latest = artifact.status.lastGateResultId
        if gate_id is not None and gate_id != latest:
            raise GateFailedError("gate_id is not the latest stored Gate for this version")
        resolved_id = latest
        if not resolved_id:
            raise GateFailedError("cannot Release without a stored Gate decision")
        stored = self.store.get_gate(resolved_id)
        if stored is None:
            raise GateFailedError(f"stored Gate decision not found: {resolved_id}")
        return stored

    def _is_experiment_pinned(self, record: AssetRecord) -> bool:
        ref = record.ref()
        return any(
            edge.relation == "evaluates"
            for edge in self.store.list_edges(target=ref.exact_key())
        )

    def mark_immutable(self, kind: AssetKind, tenant: str, name: str, version: str) -> AssetRecord | None:
        record = self.store.get_asset(kind, tenant, name, version)
        if record is None:
            return None
        stored = self.store.put_asset(record.model_copy(update={"immutable": True}))
        return stored

    async def _emit(
        self,
        event_type: ControlEventType,
        principal: Principal | None,
        *,
        resource: str,
        tenant: str,
        details: dict[str, Any] | None = None,
    ) -> ControlEvent:
        event = ControlEvent(
            type=event_type,
            actor=principal,
            tenant=tenant,
            resource=resource,
            details=details or {},
        )
        return self.store.append_event(event)
