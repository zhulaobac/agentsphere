"""Evaluation foundation: datasets, evaluators, experiments, regression, Release Gate."""

from agentsphere.evaluation.code_check import CodeCheckEvaluator
from agentsphere.evaluation.dataset import DatasetService, DatasetVersion
from agentsphere.evaluation.evaluator import Evaluator, EvaluatorRegistry
from agentsphere.evaluation.experiment import (
    ArtifactRuntimeFactory,
    CaseRun,
    ExperimentRun,
    ExperimentRunner,
    ExperimentSpec,
    ExperimentStore,
    ExperimentSummary,
    RuntimeCaseExecutor,
    ScriptedExecutor,
)
from agentsphere.evaluation.fakes import FakeHumanReview, FakeLLMJudge, FakeTrajectoryEvaluator
from agentsphere.evaluation.gate import ReleaseGate
from agentsphere.evaluation.pipeline import ReleasePipeline
from agentsphere.evaluation.regression import (
    AttributedFailure,
    FailureAttribution,
    RegressionReport,
    compare_experiments,
)
from agentsphere.evaluation.rule import RuleAssertionEvaluator
from agentsphere.evaluation.trajectory import (
    EventTrajectoryEvaluator,
    TrajectoryDiff,
    compare_trajectories,
    replay_session,
)
from agentsphere.evaluation.types import (
    AssertionSpec,
    DatasetCase,
    EnvironmentConfig,
    EvalResult,
    EvalSample,
    SampleMetrics,
    SliceTags,
)

__all__ = [
    "ArtifactRuntimeFactory",
    "AssertionSpec",
    "AttributedFailure",
    "CaseRun",
    "CodeCheckEvaluator",
    "DatasetCase",
    "DatasetService",
    "DatasetVersion",
    "EnvironmentConfig",
    "EvalResult",
    "EvalSample",
    "Evaluator",
    "EvaluatorRegistry",
    "EventTrajectoryEvaluator",
    "ExperimentRun",
    "ExperimentRunner",
    "ExperimentSpec",
    "ExperimentStore",
    "ExperimentSummary",
    "FailureAttribution",
    "FakeHumanReview",
    "FakeLLMJudge",
    "FakeTrajectoryEvaluator",
    "RegressionReport",
    "ReleaseGate",
    "ReleasePipeline",
    "RuleAssertionEvaluator",
    "RuntimeCaseExecutor",
    "SampleMetrics",
    "ScriptedExecutor",
    "SliceTags",
    "TrajectoryDiff",
    "compare_experiments",
    "compare_trajectories",
    "replay_session",
]
