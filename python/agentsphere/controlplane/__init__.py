"""Python Control Plane: Artifact Registry, lifecycle, release, tenant scoping.

A later Go service can land under the reserved ``control-plane/`` directory and
speak the same Artifact / RuntimeProvider contracts. This package is the P1
implementation used in CI.
"""

from agentsphere.controlplane.artifact import (
    AgentArtifact,
    AgentEvalSpec,
    AgentGateSpec,
    AgentSpec,
    AgentStatus,
    DeployedVersion,
    HarnessSpec,
    ModelSpec,
    RuntimeSpec,
)
from agentsphere.controlplane.assets import STUB_KINDS, AssetRecord, DependencyEdge
from agentsphere.controlplane.auth import assert_can_mutate, assert_can_read
from agentsphere.controlplane.errors import (
    ArtifactNotFoundError,
    AssetNotFoundError,
    AuthorizationError,
    ControlPlaneError,
    DependencyError,
    DuplicateAssetError,
    GateFailedError,
    ImmutableAssetError,
    LifecycleError,
)
from agentsphere.controlplane.events import ControlEvent, ControlEventType
from agentsphere.controlplane.lifecycle import ALLOWED_TRANSITIONS
from agentsphere.controlplane.release import CANARY_WEIGHT, FULL_WEIGHT, DeploymentRecord
from agentsphere.controlplane.runtime import ControlRuntimeProvider, DeploymentBinding, ManagedRuntimeProvider
from agentsphere.controlplane.service import ControlPlane
from agentsphere.controlplane.store import ControlStore, InMemoryControlStore
from agentsphere.controlplane.types import (
    API_VERSION,
    AssetKind,
    AssetMetadata,
    AssetRef,
    DeploymentStatus,
    GateCheck,
    GateDecision,
    GateThresholds,
    LifecyclePhase,
    Principal,
    ReleaseChannel,
    SLOSignal,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "API_VERSION",
    "CANARY_WEIGHT",
    "FULL_WEIGHT",
    "STUB_KINDS",
    "AgentArtifact",
    "AgentEvalSpec",
    "AgentGateSpec",
    "AgentSpec",
    "AgentStatus",
    "ArtifactNotFoundError",
    "AssetKind",
    "AssetMetadata",
    "AssetNotFoundError",
    "AssetRecord",
    "AssetRef",
    "AuthorizationError",
    "ControlEvent",
    "ControlEventType",
    "ControlPlane",
    "ControlPlaneError",
    "ControlRuntimeProvider",
    "ControlStore",
    "DependencyEdge",
    "DependencyError",
    "DeployedVersion",
    "DeploymentBinding",
    "DeploymentRecord",
    "DeploymentStatus",
    "DuplicateAssetError",
    "GateCheck",
    "GateDecision",
    "GateFailedError",
    "GateThresholds",
    "HarnessSpec",
    "ImmutableAssetError",
    "InMemoryControlStore",
    "LifecycleError",
    "LifecyclePhase",
    "ManagedRuntimeProvider",
    "ModelSpec",
    "Principal",
    "ReleaseChannel",
    "RuntimeSpec",
    "SLOSignal",
    "assert_can_mutate",
    "assert_can_read",
]
