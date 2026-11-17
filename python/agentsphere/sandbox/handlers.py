"""Importable sandbox tool handlers. Used by the sandbox worker and by P1 tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def echo(arguments: Mapping[str, Any], secrets: Mapping[str, str] | None = None) -> str:
    del secrets
    return str(arguments.get("text", ""))


def probe_paths(arguments: Mapping[str, Any], secrets: Mapping[str, str] | None = None) -> dict[str, Any]:
    del secrets
    host_path = str(arguments.get("host_path", ""))
    workspace_file = str(arguments.get("workspace_path", "note.txt"))
    result: dict[str, Any] = {"workspace": None, "host": None, "host_hidden": False}
    try:
        result["workspace"] = Path(workspace_file).read_text()
    except Exception as exc:  # noqa: BLE001
        result["workspace"] = f"error:{exc}"
    try:
        result["host"] = Path(host_path).read_text()
    except Exception:
        result["host_hidden"] = True
        result["host"] = None
    return result


def probe_secret(arguments: Mapping[str, Any], secrets: Mapping[str, str] | None = None) -> dict[str, Any]:
    secrets = dict(secrets or {})
    name = str(arguments.get("name", "weather_api"))
    value = secrets.get(name, "")
    env_blob = " ".join(f"{k}={v}" for k, v in os.environ.items())
    return {
        "secrets_has_value": bool(value),
        "secret_length": len(value),
        "env_contains_secret": bool(value) and value in env_blob,
        "text": str(arguments.get("text", "")),
    }


def leak_secret(arguments: Mapping[str, Any], secrets: Mapping[str, str] | None = None) -> str:
    secrets = dict(secrets or {})
    name = str(arguments.get("name", "weather_api"))
    return f"leaked={secrets.get(name, '')}"
