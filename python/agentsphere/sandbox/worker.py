"""In-sandbox tool worker. Secrets arrive on file descriptor 3, never via environment variables."""

from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
from typing import Any


SECRET_FD = 3


def main() -> None:
    job = json.loads(sys.stdin.read() or "{}")
    secrets = _read_secrets()
    arguments = dict(job.get("arguments") or {})
    handler = job.get("handler")
    try:
        result = _invoke(handler, arguments, secrets)
        sys.stdout.write(json.dumps({"ok": True, "result": result}))
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)


def _read_secrets() -> dict[str, str]:
    try:
        with os.fdopen(SECRET_FD, "r") as handle:
            raw = handle.read()
    except OSError:
        return {}
    if not raw:
        return {}
    data = json.loads(raw)
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def _invoke(handler: str | None, arguments: dict[str, Any], secrets: dict[str, str]) -> Any:
    from agentsphere.identity.vault import IssuedCredential, bind_issued_credential
    from agentsphere.kernel.ids import new_id

    if not handler:
        raise RuntimeError("sandbox job is missing handler")
    module_name, _, attr = handler.partition(":")
    if not attr:
        raise RuntimeError("handler must be 'module:function'")
    module_globals = runpy.run_module(module_name, run_name=module_name)
    func = module_globals.get(attr)
    if func is None:
        import importlib

        module = importlib.import_module(module_name)
        func = getattr(module, attr)
    token = None
    if secrets:
        first_name, first_secret = next(iter(secrets.items()))
        issued = IssuedCredential.model_construct(
            handle=None,  # type: ignore[arg-type]
            token=first_secret,
            secret_name=first_name,
        )
        del first_secret
        issued.token = secrets[first_name]
        issued.secret_name = first_name
        from agentsphere.identity.vault import CredentialHandle

        issued.handle = CredentialHandle(
            credential_id=new_id("cred"),
            name=first_name,
            agent_id="sandbox",
            expires_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        token = bind_issued_credential(issued)
    try:
        return func(arguments, secrets=secrets) if _accepts_secrets(func) else func(arguments)
    finally:
        if token is not None:
            from agentsphere.identity.vault import reset_issued_credential

            reset_issued_credential(token)


def _accepts_secrets(func: Any) -> bool:
    import inspect

    try:
        return "secrets" in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
