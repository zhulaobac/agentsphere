from __future__ import annotations

import pytest

from agentsphere.tool_registry import ToolSpec


def test_non_idempotent_tool_requires_compensation_or_two_phase() -> None:
    with pytest.raises(ValueError, match="compensation_action or two_phase"):
        ToolSpec(name="charge", version="1.0.0", owner="payments", idempotent=False)


def test_non_idempotent_with_compensation_is_allowed() -> None:
    spec = ToolSpec(
        name="charge",
        version="1.0.0",
        owner="payments",
        idempotent=False,
        compensation_action="refund",
    )
    assert spec.compensation_action == "refund"


def test_two_phase_non_idempotent_is_allowed() -> None:
    spec = ToolSpec(name="transfer", version="1.0.0", owner="payments", idempotent=False, two_phase=True)
    assert spec.two_phase is True
