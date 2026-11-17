"""Redact secrets from event payloads, traces, and logs. Vault secrets must never be persisted in plaintext."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agentsphere.kernel.events import Event

_SECRET_KEY_FRAGMENTS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "credential",
    "private_key",
    "authorization",
)

REDACTED = "[redacted]"


def is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def redact(value: Any, extra_secrets: set[str] | None = None) -> Any:
    secrets = {s for s in (extra_secrets or set()) if s}
    return _redact(value, secrets)


def contains_secret(value: Any, secrets: set[str]) -> bool:
    if not secrets:
        return False
    if isinstance(value, str):
        return any(secret and secret in value for secret in secrets)
    if isinstance(value, dict):
        return any(contains_secret(v, secrets) for v in value.values()) or any(
            contains_secret(k, secrets) for k in value
        )
    if isinstance(value, (list, tuple, set)):
        return any(contains_secret(v, secrets) for v in value)
    return False


def _redact(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if is_secret_key(str(key)):
                out[key] = REDACTED
            else:
                out[key] = _redact(item, secrets)
        return out
    if isinstance(value, list):
        return [_redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, secrets) for item in value)
    if isinstance(value, str) and secrets:
        redacted = value
        for secret in secrets:
            if secret and secret in redacted:
                redacted = redacted.replace(secret, REDACTED)
        return redacted
    return value


class RedactingListener:
    """Wrap an event listener so vault plaintext is stripped before traces/logs record it."""

    def __init__(self, inner: object, secrets_fn: Callable[[], set[str]]) -> None:
        self._inner = inner
        self._secrets_fn = secrets_fn

    async def on_event(self, event: Event) -> None:
        payload = redact(event.payload, extra_secrets=self._secrets_fn())
        redacted = event.model_copy(update={"payload": payload})
        on_event = getattr(self._inner, "on_event", self._inner)
        result = on_event(redacted)
        if isinstance(result, Awaitable):
            await result
