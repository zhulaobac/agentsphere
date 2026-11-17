"""Tool Dispatcher: schema validation, concurrency, timeout/retry, result truncation, idempotency cache."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import BaseModel, Field

from agentsphere.kernel.actions import ToolCallRequest
from agentsphere.kernel.hitl import ApprovalRequired
from agentsphere.kernel.invocation import InvocationContext, bind_invocation, reset_invocation
from agentsphere.kernel.middleware import Middleware, MiddlewareStage, run_chain


class ToolInvokeError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class NonIdempotentRecoveryError(RuntimeError):
    """A non-idempotent in-flight tool must not be silently re-executed on recovery."""


class ExecutableTool(Protocol):
    name: str
    version: str
    parameter_schema: Mapping[str, Any]
    idempotent: bool

    def invoke(self, arguments: Mapping[str, Any]) -> Any: ...


class ToolResolver(Protocol):
    def resolve(self, name: str, version: str | None = None) -> ExecutableTool: ...


class ResultCacheLookup(Protocol):
    async def get_result(self, idempotency_key: str) -> Any: ...

    async def put_result(self, record: Any) -> None: ...


class ToolResult(BaseModel):
    call_id: str
    name: str
    version: str = ""
    output: str = ""
    truncated: bool = False
    error: str | None = None
    attempts: int = 1
    idempotency_key: str = ""
    from_cache: bool = False
    skipped_non_idempotent: bool = False
    phase: str = "execute"
    untrusted: bool = True


class DispatcherConfig(BaseModel):
    max_concurrency: int = 4
    timeout_seconds: float = 10.0
    max_retries: int = 1
    retry_backoff_seconds: float = 0.01
    max_result_chars: int = 8_192


class ToolDispatcher:
    def __init__(
        self,
        resolver: ToolResolver,
        config: DispatcherConfig | None = None,
        middlewares: Sequence[Middleware] | None = None,
        *,
        result_cache: ResultCacheLookup | None = None,
        vault: Any = None,
        sandbox: Any = None,
        identity: Any = None,
        audit: Any = None,
        approvals: Any = None,
        workspace_path: str | None = None,
    ) -> None:
        self.resolver = resolver
        self.config = config or DispatcherConfig()
        self.middlewares = list(middlewares or [])
        self.result_cache = result_cache
        self.vault = vault
        self.sandbox = sandbox
        self.identity = identity
        self.audit = audit
        self.approvals = approvals
        self.workspace_path = workspace_path

    async def dispatch(
        self,
        calls: Sequence[ToolCallRequest],
        *,
        session_id: str,
        turn_id: str,
        step_id: str,
        step_index: int = 0,
        recovery: bool = False,
        in_flight_keys: set[str] | None = None,
        confirm_commit: bool = False,
        confirm_keys: set[str] | None = None,
        agent_id: str = "",
    ) -> list[ToolResult]:
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        in_flight = set(in_flight_keys or ())
        allowed_keys = set(confirm_keys or ())

        async def one(call: ToolCallRequest, index: int) -> ToolResult:
            async with semaphore:
                return await self._invoke(
                    call,
                    session_id,
                    turn_id,
                    step_id,
                    index,
                    step_index=step_index,
                    recovery=recovery,
                    in_flight=in_flight,
                    confirm_commit=confirm_commit,
                    confirm_keys=allowed_keys,
                    agent_id=agent_id,
                )

        gathered = await asyncio.gather(
            *[one(call, i) for i, call in enumerate(calls)],
            return_exceptions=True,
        )
        results: list[ToolResult] = []
        approval: ApprovalRequired | None = None
        for item in gathered:
            if isinstance(item, ApprovalRequired):
                approval = item
            elif isinstance(item, BaseException):
                raise item
            else:
                results.append(item)
        if approval is not None:
            raise approval
        return results

    async def _invoke(
        self,
        call: ToolCallRequest,
        session_id: str,
        turn_id: str,
        step_id: str,
        index: int,
        *,
        step_index: int,
        recovery: bool,
        in_flight: set[str],
        confirm_commit: bool,
        confirm_keys: set[str],
        agent_id: str,
    ) -> ToolResult:
        key = idempotency_key(session_id, turn_id, step_index, index, call)
        invocation = InvocationContext(
            session_id=session_id,
            turn_id=turn_id,
            step_id=step_id,
            step_index=step_index,
            call_index=index,
            agent_id=agent_id,
            idempotency_key=key,
            recovery=recovery,
            in_flight=key in in_flight,
            confirm_commit=confirm_commit or key in confirm_keys,
            confirm_keys=confirm_keys,
        )
        token = bind_invocation(invocation)
        try:
            return await self._invoke_bound(call, key, invocation)
        finally:
            reset_invocation(token)

    async def _invoke_bound(
        self,
        call: ToolCallRequest,
        key: str,
        invocation: InvocationContext,
    ) -> ToolResult:
        granted = self._is_granted(invocation)
        if granted:
            invocation.confirm_commit = True
        cached = await self._cache_lookup(key)
        if cached is not None:
            prepare_waiting_commit = cached.phase == "prepare" and invocation.confirm_commit
            if not prepare_waiting_commit:
                return cached.model_copy(update={"from_cache": True})

        try:
            tool = self.resolver.resolve(call.name, call.version)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(call_id=call.id, name=call.name, error=str(exc), idempotency_key=key)

        pending_prepare_commit = (
            cached is not None and cached.phase == "prepare" and invocation.confirm_commit
        )
        # HITL grant or two-phase confirm must run (or commit), not skip as a duplicate side effect.
        if (
            invocation.recovery
            and invocation.in_flight
            and not _is_idempotent(tool)
            and not granted
            and not pending_prepare_commit
        ):
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                version=getattr(tool, "version", ""),
                error="non-idempotent tool was in-flight; recovery will not re-execute it",
                idempotency_key=key,
                skipped_non_idempotent=True,
            )
            await self._cache_store(invocation, result, phase="skipped")
            return result

        try:
            _validate_params(call.arguments, tool.parameter_schema)
        except JSONSchemaValidationError as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                version=getattr(tool, "version", ""),
                error=f"parameter validation failed: {exc.message}",
                idempotency_key=key,
            )

        if self.approvals is not None:
            await self._enforce_approval(call, invocation)

        two_phase = bool(getattr(tool, "two_phase", False))
        phase = "commit" if (two_phase and invocation.confirm_commit) else ("prepare" if two_phase else "execute")

        attempts = 0
        last_error: str | None = None
        max_tries = 1 + max(0, self.config.max_retries)
        raw: Any = None
        succeeded = False
        issued = None
        issued_token = None
        if self.vault is not None:
            issued, issued_token = await self._issue_credentials(tool, invocation)

        try:
            while attempts < max_tries:
                attempts += 1

                async def core(payload: ToolCallRequest) -> Any:
                    return await self._call_tool(tool, payload, invocation, phase=phase)

                try:
                    raw = await asyncio.wait_for(
                        run_chain(
                            self.middlewares,
                            MiddlewareStage.BEFORE_TOOL,
                            call,
                            core,
                        ),
                        timeout=self.config.timeout_seconds,
                    )
                    succeeded = True
                    break
                except ApprovalRequired:
                    raise
                except PermissionError:
                    raise
                except TimeoutError:
                    last_error = f"tool timed out after {self.config.timeout_seconds}s"
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    if isinstance(exc, ToolInvokeError) and not exc.retryable:
                        break
                if attempts < max_tries:
                    await asyncio.sleep(self.config.retry_backoff_seconds * attempts)
        finally:
            if issued_token is not None:
                from agentsphere.identity.vault import reset_issued_credential

                reset_issued_credential(issued_token)
            del issued

        if not succeeded:
            result = ToolResult(
                call_id=call.id,
                name=call.name,
                version=getattr(tool, "version", ""),
                error=last_error or "tool failed",
                attempts=attempts,
                idempotency_key=key,
                phase=phase,
            )
            return result

        try:
            after = await run_chain(
                self.middlewares,
                MiddlewareStage.AFTER_TOOL,
                raw,
                _identity,
            )
        except ApprovalRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                call_id=call.id,
                name=call.name,
                version=getattr(tool, "version", ""),
                error=f"after_tool middleware failed: {exc}",
                attempts=attempts,
                idempotency_key=key,
                phase=phase,
            )

        text, truncated = _truncate(_stringify(after), self.config.max_result_chars)
        text = self._redact_text(text)
        result = ToolResult(
            call_id=call.id,
            name=call.name,
            version=getattr(tool, "version", ""),
            output=text,
            truncated=truncated,
            attempts=attempts,
            idempotency_key=key,
            phase=phase,
            untrusted=True,
        )
        await self._cache_store(invocation, result, phase=phase)
        await self._audit_tool(invocation, call, result)
        if two_phase and phase == "prepare" and not invocation.confirm_commit:
            raise ApprovalRequired(
                call,
                reason="two-phase tool requires confirmation before commit",
                session_id=invocation.session_id,
                turn_id=invocation.turn_id,
                idempotency_key=key,
            )
        return result

    async def _call_tool(
        self,
        tool: ExecutableTool,
        call: ToolCallRequest,
        invocation: InvocationContext,
        *,
        phase: str,
    ) -> Any:
        if getattr(tool, "sandboxed", False) and self.sandbox is not None:
            return await self._sandbox_invoke(tool, call, invocation)
        handler = tool.invoke
        spec = getattr(tool, "spec", None)
        if phase == "prepare" and spec is not None and spec.prepare_handler is not None:
            handler = spec.prepare_handler
        elif phase == "commit" and spec is not None and spec.commit_handler is not None:
            handler = spec.commit_handler
        return await _call_handler(handler, call.arguments)

    async def _sandbox_invoke(
        self,
        tool: ExecutableTool,
        call: ToolCallRequest,
        invocation: InvocationContext,
    ) -> Any:
        from agentsphere.sandbox.limits import SandboxSpec
        from agentsphere.sandbox.worker import SECRET_FD

        spec = getattr(tool, "spec", None)
        handler = getattr(tool, "sandbox_handler", None) or (spec.sandbox_handler if spec is not None else None)
        if not handler:
            raise ToolInvokeError("sandboxed tool is missing sandbox_handler (module:function)")
        job = json.dumps({"handler": handler, "arguments": call.arguments, "phase": "execute"})
        secrets: dict[str, str] = {}
        if self.vault is not None:
            from agentsphere.identity.vault import current_issued_credential

            issued = current_issued_credential()
            if issued is not None:
                secrets[issued.secret_name] = await self.vault.reveal(issued.token)
        secret_r, secret_w = os.pipe()
        os.write(secret_w, json.dumps(secrets).encode("utf-8"))
        os.close(secret_w)
        secrets.clear()
        extra_readable = [sys.prefix, str(Path(sys.executable).resolve().parent)]
        try:
            import agentsphere as _as

            extra_readable.append(str(Path(_as.__file__).resolve().parent.parent))
        except Exception:
            pass
        extra_readable.extend(p for p in sys.path if p and Path(p).exists())
        try:
            sandbox_spec = SandboxSpec(
                working_directory=self.workspace_path,
                restrict_filesystem=True,
                extra_readable_paths=extra_readable,
            )
            result = await self.sandbox.run(
                [sys.executable, "-m", "agentsphere.sandbox.worker"],
                sandbox_spec,
                stdin_data=job.encode("utf-8"),
                extra_fds={SECRET_FD: secret_r},
            )
        finally:
            try:
                os.close(secret_r)
            except OSError:
                pass
        if result.timed_out:
            raise ToolInvokeError(f"sandbox timed out after {sandbox_spec.limits.wall_time_seconds}s")
        payload = json.loads(result.stdout or "{}") if result.stdout else {}
        if not payload.get("ok", False):
            raise ToolInvokeError(str(payload.get("error") or result.stderr or "sandbox tool failed"))
        return payload.get("result")

    async def _issue_credentials(self, tool: ExecutableTool, invocation: InvocationContext):
        from agentsphere.identity.vault import bind_issued_credential

        names = list(getattr(tool, "required_credentials", []) or [])
        if not names:
            return None, None
        agent_id = invocation.agent_id or getattr(self.identity, "agent_id", "")
        issued = await self.vault.issue(names[0], agent_id=agent_id)
        return issued, bind_issued_credential(issued)

    def _is_granted(self, invocation: InvocationContext) -> bool:
        if invocation.confirm_commit:
            return True
        if invocation.confirm_keys and invocation.idempotency_key in invocation.confirm_keys:
            return True
        if self.approvals is None:
            return False
        record = self.approvals.decision_for(
            session_id=invocation.session_id,
            idempotency_key=invocation.idempotency_key,
        )
        return record is not None and record.status == "granted"

    def _redact_text(self, text: str) -> str:
        if not text:
            return text
        secrets: set[str] = set()
        if self.vault is not None:
            lookup = getattr(self.vault, "known_plaintext_secrets", None)
            if callable(lookup):
                secrets |= set(lookup() or ())
        if not secrets:
            return text
        from agentsphere.identity.redact import redact

        redacted = redact(text, extra_secrets=secrets)
        return redacted if isinstance(redacted, str) else text

    async def _enforce_approval(self, call: ToolCallRequest, invocation: InvocationContext) -> None:
        record = self.approvals.decision_for(
            session_id=invocation.session_id,
            idempotency_key=invocation.idempotency_key,
        )
        if record is None:
            return
        if record.status == "granted":
            invocation.confirm_commit = True
            return
        if record.status == "denied":
            raise PermissionError(f"human denied tool {call.name}")
        if record.status == "pending":
            raise ApprovalRequired(
                call,
                reason="waiting for human approval",
                approval_id=record.approval_id,
                session_id=invocation.session_id,
                turn_id=invocation.turn_id,
                idempotency_key=invocation.idempotency_key,
            )

    async def _cache_lookup(self, key: str) -> ToolResult | None:
        if self.result_cache is None:
            return None
        record = await self.result_cache.get_result(key)
        if record is None:
            return None
        payload = record.payload if hasattr(record, "payload") else record
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        result = ToolResult.model_validate(payload)
        result.from_cache = True
        if hasattr(record, "phase"):
            result.phase = record.phase
        return result

    async def _cache_store(self, invocation: InvocationContext, result: ToolResult, *, phase: str) -> None:
        if self.result_cache is None:
            return
        from agentsphere.runtime.cache import CachedResult

        await self.result_cache.put_result(
            CachedResult(
                idempotency_key=result.idempotency_key,
                session_id=invocation.session_id,
                turn_id=invocation.turn_id,
                step_index=invocation.step_index,
                name=result.name,
                payload=result.model_dump(),
                phase=phase,
            )
        )

    async def _audit_tool(
        self,
        invocation: InvocationContext,
        call: ToolCallRequest,
        result: ToolResult,
    ) -> None:
        if self.audit is None:
            return
        from agentsphere.identity.audit import AuditRecord
        from agentsphere.identity.redact import redact

        await self.audit.record(
            AuditRecord(
                agent_id=invocation.agent_id,
                action="tool.call",
                resource=call.name,
                decision="error" if result.error else "ok",
                session_id=invocation.session_id,
                turn_id=invocation.turn_id,
                details=redact(
                    {
                        "call_id": call.id,
                        "arguments": call.arguments,
                        "from_cache": result.from_cache,
                        "phase": result.phase,
                    }
                ),
            )
        )


async def _identity(payload: Any) -> Any:
    return payload


async def _call_handler(handler: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    """Run async handlers on the loop; run sync handlers in a worker thread so timeouts apply."""
    args = dict(arguments)
    if inspect.iscoroutinefunction(handler):
        return await handler(args)

    def run_sync() -> Any:
        return handler(args)

    result = await asyncio.to_thread(run_sync)
    if inspect.isawaitable(result):
        return await result
    return result


def _validate_params(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if not schema:
        return
    Draft202012Validator(schema).validate(dict(arguments))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False)


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "...[truncated]"
    keep = max(0, max_chars - len(marker))
    return text[:keep] + marker, True


def _is_idempotent(tool: ExecutableTool) -> bool:
    return bool(getattr(tool, "idempotent", False))


def idempotency_key(
    session_id: str,
    turn_id: str,
    step_index: int,
    index: int,
    call: ToolCallRequest,
) -> str:
    """Stable key: hash(session, turn, step). Recovery must reuse this, so step_index is used, not a random step_id."""
    material = json.dumps(
        {
            "session": session_id,
            "turn": turn_id,
            "step": step_index,
            "index": index,
            "name": call.name,
            "arguments": call.arguments,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# P0 name kept as a private alias so existing imports of the helper still work if any.
_idempotency_key = idempotency_key
