"""Agent Identity, Credential Vault, policy, and audit. Production backends plug in behind these protocols."""

from agentsphere.identity.agent import AgentIdentity
from agentsphere.identity.audit import AuditLog, AuditRecord, InMemoryAuditLog
from agentsphere.identity.policy import (
    ApprovalRecord,
    ApprovalStore,
    PolicyDecision,
    PolicyEffect,
    PolicyEngine,
    PolicyHitlMiddleware,
    PolicyRequest,
    PolicyRule,
    StaticPolicyEngine,
)
from agentsphere.identity.redact import REDACTED, RedactingListener, contains_secret, redact
from agentsphere.identity.vault import (
    CredentialHandle,
    InMemoryVault,
    IssuedCredential,
    Vault,
    VaultError,
    bind_issued_credential,
    current_issued_credential,
    current_secret_token,
    reset_issued_credential,
)

__all__ = [
    "AgentIdentity",
    "ApprovalRecord",
    "ApprovalStore",
    "AuditLog",
    "AuditRecord",
    "CredentialHandle",
    "InMemoryAuditLog",
    "InMemoryVault",
    "IssuedCredential",
    "PolicyDecision",
    "PolicyEffect",
    "PolicyEngine",
    "PolicyHitlMiddleware",
    "PolicyRequest",
    "PolicyRule",
    "REDACTED",
    "RedactingListener",
    "StaticPolicyEngine",
    "Vault",
    "VaultError",
    "bind_issued_credential",
    "contains_secret",
    "current_issued_credential",
    "current_secret_token",
    "redact",
    "reset_issued_credential",
]
