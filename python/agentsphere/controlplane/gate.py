"""GateDecision is untrusted input. Passed is derived from non-skipped checks."""

from __future__ import annotations

from agentsphere.controlplane.errors import GateFailedError
from agentsphere.controlplane.types import GateDecision


def derived_passed(decision: GateDecision) -> bool:
    """A gate passes only when it has material checks and all of them pass."""
    material = [check for check in decision.checks if not check.skipped]
    if not material:
        return False
    return all(check.passed for check in material)


def verify_gate_decision(decision: GateDecision) -> GateDecision:
    """Reject empty or inconsistent decisions. Do not coerce ``passed`` to True."""
    derived = derived_passed(decision)
    if decision.passed != derived:
        raise GateFailedError(
            "inconsistent GateDecision: passed does not match non-skipped checks",
            decision=decision,
        )
    if not derived:
        raise GateFailedError("cannot Release without a passing Gate", decision=decision)
    return decision
