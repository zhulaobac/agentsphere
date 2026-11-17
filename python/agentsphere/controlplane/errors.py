"""Control Plane errors. Preconditions fail closed."""

from __future__ import annotations

from typing import Any


class ControlPlaneError(Exception):
    """Base error for Artifact Registry, lifecycle, and release."""


class AuthorizationError(ControlPlaneError, PermissionError):
    """Tenant/owner scoping rejected the operation."""


class AssetNotFoundError(ControlPlaneError, KeyError):
    pass


class ArtifactNotFoundError(AssetNotFoundError):
    pass


class DuplicateAssetError(ControlPlaneError, ValueError):
    pass


class ImmutableAssetError(ControlPlaneError, ValueError):
    """Built (or later) artifacts cannot be mutated in place; bump the version."""


class LifecycleError(ControlPlaneError, ValueError):
    """Illegal phase transition or failed precondition."""


class GateFailedError(LifecycleError):
    """Approved → Released is blocked when the Release Gate does not pass."""

    def __init__(self, message: str, *, decision: Any | None = None) -> None:
        super().__init__(message)
        self.decision = decision


class DependencyError(ControlPlaneError, ValueError):
    """Referenced asset is missing or the dependency graph is invalid."""
