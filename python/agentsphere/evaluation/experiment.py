"""Experiment: one run = (Agent version, dataset version, evaluator version, environment)."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentsphere.controlplane.artifact import AgentArtifact
from agentsphere.controlplane.assets import DependencyEdge
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.types import AssetKind, AssetRef, Principal
from agentsphere.evaluation.dataset import DatasetService, DatasetVersion
from agentsphere.evaluation.evaluator import Evaluator, EvaluatorRegistry
from agentsphere.evaluation.metrics import metrics_from_events, output_from_events
from agentsphere.evaluation.types import DatasetCase, EnvironmentConfig, EvalResult, EvalSample, SampleMetrics
from agentsphere.kernel.events import Event
from agentsphere.kernel.ids import new_id
from agentsphere.runtime.interfaces import RuntimeProvider
from agentsphere.runtime.store import InMemoryEventStore


class ExperimentSpec(BaseModel):
    agent_name: str
    agent_version: str
    dataset_name: str
    dataset_version: str
    evaluator_name: str
    evaluator_version: str
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    samples: int = 1

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class SampleRecord(BaseModel):
    sample_index: int
    output: str = ""
    passed: bool
    result: EvalResult
    events: list[Event] = Field(default_factory=list)
    metrics: SampleMetrics = Field(default_factory=SampleMetrics)
    session_id: str | None = None


class CaseRun(BaseModel):
    case: DatasetCase
    samples: list[SampleRecord] = Field(default_factory=list)

    @property
    def pass_fraction(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.passed) / len(self.samples)

    @property
    def pass_at_k(self) -> bool:
        return any(s.passed for s in self.samples)


class ExperimentSummary(BaseModel):
    n_cases: int
    n_samples: int
    pass_rate: float
    pass_at_k: float
    variance: float
    by_slice: dict[str, float] = Field(default_factory=dict)
    mean_cost: float | None = None
    mean_latency_ms: float | None = None
    fingerprint: str = ""


class ExperimentRun(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("exp"))
    spec: ExperimentSpec
    summary: ExperimentSummary
    cases: list[CaseRun] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_fingerprint: str | None = None


Executor = Callable[[AgentArtifact, DatasetCase, int], Awaitable[tuple[str, list[Event], SampleMetrics, str | None]]]


class ArtifactRuntimeFactory:
    """Build a Managed Runtime for an Agent Artifact. Not a second runtime."""

    def __init__(
        self,
        adapter: Any,
        registry: Any,
        *,
        build_runtime: Callable[..., Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.registry = registry
        self._build = build_runtime
        self.extra = dict(extra or {})

    def runtime_for(self, artifact: AgentArtifact) -> RuntimeProvider:
        from agentsphere.kernel.loop import get_strategy_class
        from agentsphere.runtime.service import build_runtime as _build

        builder = self._build or _build
        strategy_name = artifact.spec.harness.strategy
        strategy_cls = get_strategy_class(strategy_name)
        config = dict(artifact.spec.harness.config)
        try:
            strategy = strategy_cls(**{k: v for k, v in config.items() if k in {"max_steps"}})
        except TypeError:
            strategy = strategy_cls()
        store = InMemoryEventStore()
        return builder(
            adapter=self.adapter,
            registry=self.registry,
            strategy=strategy,
            store=store,
            **self.extra,
        )


class RuntimeCaseExecutor:
    def __init__(
        self,
        factory: ArtifactRuntimeFactory,
        *,
        system_prompt: str = "",
    ) -> None:
        self.factory = factory
        self.system_prompt = system_prompt

    async def __call__(
        self,
        artifact: AgentArtifact,
        case: DatasetCase,
        sample_index: int,
    ) -> tuple[str, list[Event], SampleMetrics, str | None]:
        runtime = self.factory.runtime_for(artifact)
        session = await runtime.create_session(
            system_prompt=self.system_prompt,
            metadata={
                "agent_name": artifact.metadata.name,
                "agent_version": artifact.version,
                "case_id": case.case_id,
                "sample_index": sample_index,
                "experiment": True,
            },
        )
        events = [event async for event in runtime.run_turn(session.session_id, case.input)]
        stored = await runtime.list_events(session.session_id)
        trace = stored or events
        return output_from_events(trace), trace, metrics_from_events(trace), session.session_id


class ScriptedExecutor:
    """Deterministic executor for tests that should not start a Runtime."""

    def __init__(
        self,
        outputs: dict[str, str] | Callable[[DatasetCase, int], str],
        *,
        events: dict[str, list[Event]] | None = None,
        metrics: dict[str, SampleMetrics] | None = None,
    ) -> None:
        self.outputs = outputs
        self.events = events or {}
        self.metrics = metrics or {}

    async def __call__(
        self,
        artifact: AgentArtifact,
        case: DatasetCase,
        sample_index: int,
    ) -> tuple[str, list[Event], SampleMetrics, str | None]:
        del artifact
        if callable(self.outputs):
            output = self.outputs(case, sample_index)
        else:
            output = self.outputs.get(case.case_id, self.outputs.get(case.input, ""))
        events = list(self.events.get(case.case_id, []))
        metrics = self.metrics.get(case.case_id, SampleMetrics())
        return output, events, metrics, None


class ExperimentStore:
    def __init__(self) -> None:
        self._runs: dict[str, ExperimentRun] = {}

    def put(self, run: ExperimentRun) -> ExperimentRun:
        stored = run.model_copy(deep=True)
        self._runs[stored.run_id] = stored
        return stored.model_copy(deep=True)

    def get(self, run_id: str) -> ExperimentRun | None:
        item = self._runs.get(run_id)
        return item.model_copy(deep=True) if item else None

    def list_for_agent(self, name: str, version: str) -> list[ExperimentRun]:
        items = [
            r.model_copy(deep=True)
            for r in self._runs.values()
            if r.spec.agent_name == name and r.spec.agent_version == version
        ]
        return sorted(items, key=lambda r: r.created_at)


class ExperimentRunner:
    def __init__(
        self,
        control: ControlPlane,
        evaluators: EvaluatorRegistry,
        *,
        datasets: DatasetService | None = None,
        store: ExperimentStore | None = None,
        executor: Executor | None = None,
    ) -> None:
        self.control = control
        self.evaluators = evaluators
        self.datasets = datasets or DatasetService(control)
        self.store = store or ExperimentStore()
        self.executor = executor

    async def run(
        self,
        spec: ExperimentSpec,
        principal: Principal,
        *,
        executor: Executor | None = None,
        evaluator: Evaluator | None = None,
    ) -> ExperimentRun:
        if spec.samples < 1:
            raise ValueError("samples must be >= 1")
        artifact = await self.control.get_artifact(spec.agent_name, spec.agent_version, principal)
        dataset = await self.datasets.get(spec.dataset_name, spec.dataset_version, principal)
        ev = evaluator or self.evaluators.get(spec.evaluator_name, spec.evaluator_version)
        run_executor = executor or self.executor
        if run_executor is None:
            raise RuntimeError("ExperimentRunner requires an executor or ArtifactRuntimeFactory binding")

        cases: list[CaseRun] = []
        for case in dataset.cases:
            samples: list[SampleRecord] = []
            for index in range(spec.samples):
                output, events, metrics, session_id = await run_executor(artifact, case, index)
                sample = EvalSample(
                    case=case,
                    output=output,
                    events=events,
                    metrics=metrics,
                    agent_name=artifact.metadata.name,
                    agent_version=artifact.version,
                    sample_index=index,
                )
                result = await ev.evaluate(sample)
                samples.append(
                    SampleRecord(
                        sample_index=index,
                        output=output,
                        passed=result.passed,
                        result=result,
                        events=events,
                        metrics=metrics,
                        session_id=session_id,
                    )
                )
            cases.append(CaseRun(case=case, samples=samples))

        summary = summarize(cases, samples_k=spec.samples, fingerprint=spec.fingerprint())
        run = ExperimentRun(
            spec=spec,
            summary=summary,
            cases=cases,
            agent_fingerprint=_artifact_fingerprint(artifact),
        )
        stored = self.store.put(run)
        self._pin_dataset(artifact, dataset, principal)
        return stored

    def _pin_dataset(self, artifact: AgentArtifact, dataset: DatasetVersion, principal: Principal) -> None:
        source = AssetRef(
            kind=AssetKind.AGENT,
            name=artifact.metadata.name,
            version=artifact.version,
            tenant=artifact.metadata.tenant,
        )
        target = AssetRef(
            kind=AssetKind.DATASET,
            name=dataset.name,
            version=dataset.version,
            tenant=dataset.tenant,
        )
        self.control.store.add_edge(
            DependencyEdge(source=source, target=target, relation="evaluates", actor=principal.name)
        )


def summarize(cases: list[CaseRun], *, samples_k: int, fingerprint: str) -> ExperimentSummary:
    n_cases = len(cases)
    n_samples = samples_k
    if n_cases == 0:
        return ExperimentSummary(
            n_cases=0,
            n_samples=n_samples,
            pass_rate=0.0,
            pass_at_k=0.0,
            variance=0.0,
            fingerprint=fingerprint,
        )
    pass_rate = sum(c.pass_fraction for c in cases) / n_cases
    pass_at_k = sum(1 for c in cases if c.pass_at_k) / n_cases
    fractions = [c.pass_fraction for c in cases]
    variance = statistics.pvariance(fractions) if len(fractions) > 1 else 0.0
    by_slice: dict[str, list[float]] = {}
    for case_run in cases:
        tags = case_run.case.slice
        keys = [
            f"scene:{tags.scene}" if tags.scene else "",
            f"difficulty:{tags.difficulty}" if tags.difficulty else "",
            f"risk:{tags.risk}" if tags.risk else "",
        ]
        for key in keys:
            if key:
                by_slice.setdefault(key, []).append(case_run.pass_fraction)
    costs = [s.metrics.cost for c in cases for s in c.samples if s.metrics.cost is not None]
    latencies = [s.metrics.latency_ms for c in cases for s in c.samples if s.metrics.latency_ms is not None]
    return ExperimentSummary(
        n_cases=n_cases,
        n_samples=n_samples,
        pass_rate=pass_rate,
        pass_at_k=pass_at_k,
        variance=variance,
        by_slice={k: (sum(v) / len(v) if v else 0.0) for k, v in by_slice.items()},
        mean_cost=(sum(costs) / len(costs)) if costs else None,
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        fingerprint=fingerprint,
    )


def _artifact_fingerprint(artifact: AgentArtifact) -> str:
    payload = artifact.spec.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


