"""Credential Vault. Short-lived tokens issued at call time; secrets never enter model context or sandbox env."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Protocol

from pydantic import BaseModel, Field

from agentsphere.kernel.ids import new_id


class _Clock(Protocol):
    def now(self) -> datetime: ...


class _RealClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)

_current_issued: ContextVar["IssuedCredential | None"] = ContextVar("agentsphere_issued_credential", default=None)


class VaultError(RuntimeError):
    pass


class CredentialHandle(BaseModel):
    """Safe-to-log reference. Does not contain the secret."""

    credential_id: str
    name: str
    agent_id: str
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime


class IssuedCredential(BaseModel):
    """Ephemeral token held only for the duration of a tool invocation."""

    handle: CredentialHandle
    token: str
    secret_name: str


class StoredSecret(BaseModel):
    name: str
    secret: str
    agent_id: str
    scopes: list[str] = Field(default_factory=list)
    expires_at: datetime
    credential_id: str


class Vault(Protocol):
    async def put(
        self,
        name: str,
        secret: str,
        *,
        agent_id: str,
        ttl_seconds: float = 3600,
        scopes: list[str] | None = None,
    ) -> CredentialHandle: ...

    async def issue(
        self,
        name: str,
        *,
        agent_id: str,
        ttl_seconds: float = 60,
        scopes: list[str] | None = None,
    ) -> IssuedCredential: ...

    async def reveal(self, token: str) -> str: ...

    async def revoke(self, credential_id: str) -> None: ...


class InMemoryVault:
    """Local/dev vault. A HashiCorp Vault / SPIFFE backend can implement the same protocol."""

    def __init__(self, clock: _Clock | None = None) -> None:
        self._clock = clock or _RealClock()
        self._secrets: dict[str, StoredSecret] = {}
        self._issued: dict[str, tuple[IssuedCredential, str, datetime]] = {}
        self._sinks: list[Callable[[str], None]] = []

    def add_sink(self, sink: Callable[[str], None]) -> None:
        self._sinks.append(sink)
        for stored in self._secrets.values():
            sink(stored.secret)

    def _enroll(self, secret: str) -> None:
        if not secret:
            return
        for sink in self._sinks:
            sink(secret)

    async def put(
        self,
        name: str,
        secret: str,
        *,
        agent_id: str,
        ttl_seconds: float = 3600,
        scopes: list[str] | None = None,
    ) -> CredentialHandle:
        now = self._clock.now()
        stored = StoredSecret(
            name=name,
            secret=secret,
            agent_id=agent_id,
            scopes=list(scopes or []),
            expires_at=now + timedelta(seconds=ttl_seconds),
            credential_id=new_id("cred"),
        )
        self._secrets[name] = stored
        self._enroll(secret)
        return CredentialHandle(
            credential_id=stored.credential_id,
            name=name,
            agent_id=agent_id,
            scopes=stored.scopes,
            expires_at=stored.expires_at,
        )

    async def issue(
        self,
        name: str,
        *,
        agent_id: str,
        ttl_seconds: float = 60,
        scopes: list[str] | None = None,
    ) -> IssuedCredential:
        stored = self._secrets.get(name)
        now = self._clock.now()
        if stored is None:
            raise VaultError(f"credential {name!r} not found")
        if stored.expires_at <= now:
            raise VaultError(f"credential {name!r} has expired")
        if stored.agent_id != agent_id:
            raise VaultError(f"credential {name!r} is not issued to agent {agent_id}")
        requested = list(scopes or [])
        if requested and not set(requested).issubset(set(stored.scopes)):
            raise VaultError(f"credential {name!r} does not grant scopes {requested}")
        token = new_id("tok")
        handle = CredentialHandle(
            credential_id=stored.credential_id,
            name=name,
            agent_id=agent_id,
            scopes=stored.scopes,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )
        issued = IssuedCredential(handle=handle, token=token, secret_name=name)
        self._issued[token] = (issued, stored.secret, handle.expires_at)
        self._enroll(stored.secret)
        return issued

    async def reveal(self, token: str) -> str:
        item = self._issued.get(token)
        if item is None:
            raise VaultError("unknown or expired token")
        issued, secret, expires_at = item
        del issued
        if expires_at <= self._clock.now():
            self._issued.pop(token, None)
            raise VaultError("unknown or expired token")
        return secret

    async def revoke(self, credential_id: str) -> None:
        for name, stored in list(self._secrets.items()):
            if stored.credential_id == credential_id:
                self._secrets.pop(name, None)
        for token, (issued, _secret, _exp) in list(self._issued.items()):
            if issued.handle.credential_id == credential_id:
                self._issued.pop(token, None)

    def known_plaintext_secrets(self) -> set[str]:
        """Test helper: the raw secrets this vault currently holds."""
        return {stored.secret for stored in self._secrets.values()}


def bind_issued_credential(issued: IssuedCredential | None):
    token = _current_issued.set(issued)
    return token


def reset_issued_credential(token: object) -> None:
    _current_issued.reset(token)  # type: ignore[arg-type]


def current_issued_credential() -> IssuedCredential | None:
    return _current_issued.get()


def current_secret_token() -> str | None:
    issued = _current_issued.get()
    return None if issued is None else issued.token
