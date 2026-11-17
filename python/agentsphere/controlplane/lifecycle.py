"""Lifecycle protocol: Draft → Built → Evaluated → Approved → Released → Deprecated → Retired.

Each transition is event-driven. Preconditions fail closed (e.g. no Release without a passing Gate).
"""

from __future__ import annotations

from agentsphere.controlplane.errors import GateFailedError, LifecycleError
from agentsphere.controlplane.gate import derived_passed
from agentsphere.controlplane.types import GateDecision, LifecyclePhase

ALLOWED_TRANSITIONS: frozenset[tuple[LifecyclePhase, LifecyclePhase]] = frozenset(
    {
        (LifecyclePhase.DRAFT, LifecyclePhase.BUILT),
        (LifecyclePhase.BUILT, LifecyclePhase.EVALUATED),
        (LifecyclePhase.EVALUATED, LifecyclePhase.APPROVED),
        (LifecyclePhase.APPROVED, LifecyclePhase.RELEASED),
        (LifecyclePhase.RELEASED, LifecyclePhase.DEPRECATED),
        (LifecyclePhase.DEPRECATED, LifecyclePhase.RETIRED),
    }
)


def assert_transition_allowed(current: LifecyclePhase, target: LifecyclePhase) -> None:
    if current is target:
        raise LifecycleError(f"already in phase {current.value}")
    if (current, target) not in ALLOWED_TRANSITIONS:
        raise LifecycleError(f"cannot transition {current.value} → {target.value}")


def assert_can_release(phase: LifecyclePhase, gate: GateDecision | None) -> None:
    if phase is not LifecyclePhase.APPROVED:
        raise LifecycleError(f"cannot Release from phase {phase.value}; must be Approved")
    if gate is None:
        raise GateFailedError("cannot Release without a Gate decision")
    if gate.passed != derived_passed(gate):
        raise GateFailedError(
            "inconsistent GateDecision: passed does not match non-skipped checks",
            decision=gate,
        )
    if not derived_passed(gate):
        raise GateFailedError("cannot Release without a passing Gate", decision=gate)
