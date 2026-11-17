"""Default-deny policy engine. High-risk actions can require HITL. OPA/Cedar can implement the same protocol."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from agentsphere.identity.agent import AgentIdentity
from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.hitl import ApprovalRequired
from agentsphere.kernel.invocation import current_invocation
from agentsphere.kernel.middleware import MiddlewareStage, NextFn
from agentsphere.tool_registry.spec import RiskLevel


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HITL = "require_hitl"


class PolicyRequest(BaseModel):
    identity: AgentIdentity
    tool_name: str
    risk_level: RiskLevel = RiskLevel.LOW
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str = ""
    turn_id: str = ""
    idempotency_key: str = ""
    trusted_instruction: bool = True


class PolicyDecision(BaseModel):
    allow: bool
    require_hitl: bool = False
    effect: PolicyEffect = PolicyEffect.DENY
    reason: str
    policy_id: str = "default-deny"
    approval_id: str | None = None


class PolicyRule(BaseModel):
    policy_id: str = "rule"
    agent_id: str | None = None
    tool_name: str | None = None
    max_risk: RiskLevel | None = None
    effect: PolicyEffect = PolicyEffect.ALLOW
    require_trusted_instruction: bool = False


class PolicyEngine(Protocol):
    def evaluate(self, request: PolicyRequest) -> PolicyDecision: ...


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class StaticPolicyEngine:
    """In-process stand-in for OPA. Default-deny: no matching allow rule means deny."""

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        *,
        hitl_at_or_above: RiskLevel = RiskLevel.HIGH,
    ) -> None:
        self.rules = list(rules or [])
        self.hitl_at_or_above = hitl_at_or_above

    def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        matched = [rule for rule in self.rules if _matches(rule, request)]
        if not matched:
            return PolicyDecision(
                allow=False,
                effect=PolicyEffect.DENY,
                reason=f"default-deny: no rule allows {request.tool_name} for agent {request.identity.agent_id}",
            )

        for rule in matched:
            if rule.effect is PolicyEffect.DENY:
                return PolicyDecision(
                    allow=False,
                    effect=PolicyEffect.DENY,
                    reason=f"denied by policy {rule.policy_id}",
                    policy_id=rule.policy_id,
                )
            if rule.require_trusted_instruction and not request.trusted_instruction:
                return PolicyDecision(
                    allow=False,
                    effect=PolicyEffect.DENY,
                    reason="high-risk tool requires a trusted instruction source",
                    policy_id=rule.policy_id,
                )

        hitl_rule = next((rule for rule in matched if rule.effect is PolicyEffect.REQUIRE_HITL), None)
        risk_requires_hitl = _RISK_ORDER[request.risk_level] >= _RISK_ORDER[self.hitl_at_or_above]
        if hitl_rule is not None or risk_requires_hitl:
            rule = hitl_rule or matched[0]
            return PolicyDecision(
                allow=True,
                require_hitl=True,
                effect=PolicyEffect.REQUIRE_HITL,
                reason=f"high-risk tool {request.tool_name} requires human approval",
                policy_id=rule.policy_id,
            )
        return PolicyDecision(
            allow=True,
            effect=PolicyEffect.ALLOW,
            reason=f"allowed by policy {matched[0].policy_id}",
            policy_id=matched[0].policy_id,
        )


def _matches(rule: PolicyRule, request: PolicyRequest) -> bool:
    if rule.agent_id is not None and rule.agent_id != request.identity.agent_id:
        return False
    if rule.tool_name is not None and rule.tool_name != request.tool_name:
        return False
    if rule.max_risk is not None and _RISK_ORDER[request.risk_level] > _RISK_ORDER[rule.max_risk]:
        return False
    return True


class ApprovalRecord(BaseModel):
    approval_id: str
    session_id: str
    turn_id: str
    tool_name: str
    idempotency_key: str
    status: Literal["pending", "granted", "denied"] = "pending"
    reason: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class ApprovalStore:
    def __init__(self) -> None:
        self._items: dict[str, ApprovalRecord] = {}

    def put(self, record: ApprovalRecord) -> ApprovalRecord:
        self._items[record.approval_id] = record
        return record

    def get(self, approval_id: str) -> ApprovalRecord | None:
        return self._items.get(approval_id)

    def decision_for(self, *, session_id: str, idempotency_key: str) -> ApprovalRecord | None:
        pending = None
        latest: ApprovalRecord | None = None
        for record in self._items.values():
            if record.session_id == session_id and record.idempotency_key == idempotency_key:
                latest = record
                if record.status == "pending":
                    pending = record
        return pending or latest

    def decide(self, approval_id: str, status: Literal["granted", "denied"]) -> ApprovalRecord:
        record = self._items.get(approval_id)
        if record is None:
            raise KeyError(f"approval not found: {approval_id}")
        record.status = status
        return record

    def pending_for_session(self, session_id: str) -> list[ApprovalRecord]:
        return [r for r in self._items.values() if r.session_id == session_id and r.status == "pending"]


class PolicyHitlMiddleware:
    """before_tool gate: evaluate policy, audit, and raise ApprovalRequired for HITL."""

    def __init__(
        self,
        engine: PolicyEngine,
        identity: AgentIdentity,
        approvals: ApprovalStore,
        *,
        risk_lookup: Any = None,
        audit: Any = None,
    ) -> None:
        self.engine = engine
        self.identity = identity
        self.approvals = approvals
        self.risk_lookup = risk_lookup
        self.audit = audit

    async def wrap(self, stage: MiddlewareStage, payload: Any, nxt: NextFn) -> Any:
        if stage is not MiddlewareStage.BEFORE_TOOL or not isinstance(payload, ToolCallRequest):
            return await nxt(payload)

        invocation = current_invocation()
        session_id = invocation.session_id if invocation else ""
        turn_id = invocation.turn_id if invocation else ""
        idempotency_key = invocation.idempotency_key if invocation else ""

        existing = None
        if session_id and idempotency_key:
            existing = self.approvals.decision_for(session_id=session_id, idempotency_key=idempotency_key)
        if existing is not None and existing.status == "granted":
            return await nxt(payload)
        if existing is not None and existing.status == "denied":
            raise PermissionError(f"human denied tool {payload.name}")
        if existing is not None and existing.status == "pending":
            raise ApprovalRequired(
                payload,
                reason="waiting for human approval",
                approval_id=existing.approval_id,
                session_id=session_id,
                turn_id=turn_id,
                idempotency_key=idempotency_key,
            )

        risk = RiskLevel.LOW
        if self.risk_lookup is not None:
            risk = self.risk_lookup(payload.name, payload.version)

        from agentsphere.identity.audit import AuditRecord

        request = PolicyRequest(
            identity=self.identity,
            tool_name=payload.name,
            risk_level=risk,
            arguments=dict(payload.arguments),
            session_id=session_id,
            turn_id=turn_id,
            idempotency_key=idempotency_key,
            trusted_instruction=True,
        )
        decision = self.engine.evaluate(request)
        if self.audit is not None:
            await self.audit.record(
                AuditRecord(
                    agent_id=self.identity.agent_id,
                    action="policy.decide",
                    resource=payload.name,
                    decision=decision.effect.value,
                    session_id=session_id or None,
                    turn_id=turn_id or None,
                    details={"reason": decision.reason, "allow": decision.allow, "require_hitl": decision.require_hitl},
                )
            )
        if not decision.allow:
            raise PermissionError(decision.reason)
        if decision.require_hitl:
            raise ApprovalRequired(
                payload,
                reason=decision.reason,
                decision=decision,
                session_id=session_id,
                turn_id=turn_id,
                idempotency_key=idempotency_key,
            )
        return await nxt(payload)
