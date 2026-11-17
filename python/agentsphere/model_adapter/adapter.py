"""Model Adapter: unified messages, tool calling, streaming, params, capabilities, fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from agentsphere.model_adapter.provider import FallbackHook, ModelAdapterError, ModelProvider
from agentsphere.model_adapter.types import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
)


class ModelAdapter:
    """Facade over one primary provider and an ordered fallback list.

    The Kernel talks only to this type. Adding a vendor means implementing
    ``ModelProvider`` and passing it here — Kernel code does not change.
    """

    def __init__(
        self,
        primary: ModelProvider,
        fallbacks: Sequence[ModelProvider] | None = None,
        hooks: Sequence[FallbackHook] | None = None,
    ) -> None:
        self.primary = primary
        self.fallbacks = list(fallbacks or [])
        self.hooks = list(hooks or [])

    def capabilities(self) -> ModelCapabilities:
        """Union of capabilities across the chain (primary first, then fallbacks)."""
        caps = [self.primary.capabilities(), *[p.capabilities() for p in self.fallbacks]]
        return ModelCapabilities(
            supports_tools=any(c.supports_tools for c in caps),
            supports_parallel_tools=any(c.supports_parallel_tools for c in caps),
            supports_streaming=any(c.supports_streaming for c in caps),
            supports_thinking=any(c.supports_thinking for c in caps),
            context_window=max(c.context_window for c in caps),
            max_output_tokens=max(c.max_output_tokens for c in caps),
        )

    def detect_capabilities(self, provider: ModelProvider | None = None) -> ModelCapabilities:
        target = provider or self.primary
        return target.capabilities()

    def providers(self) -> list[ModelProvider]:
        return [self.primary, *self.fallbacks]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        last_error: BaseException | None = None
        for provider in self._candidates(request):
            try:
                compatible, missing = self._compatibility(provider, request)
                if not compatible:
                    replacement = self._run_mismatch_hooks(provider, request, missing)
                    if replacement is not None:
                        return await replacement.complete(request)
                    continue
                return await provider.complete(request)
            except Exception as exc:  # noqa: BLE001 — fallback is the error policy
                last_error = exc
                replacement = self._run_failure_hooks(provider, request, exc)
                if replacement is not None:
                    try:
                        return await replacement.complete(request)
                    except Exception as hook_exc:  # noqa: BLE001
                        last_error = hook_exc
                        continue
                continue
        if request.tools:
            degraded = request.model_copy(update={"tools": []})
            try:
                return await self.primary.complete(degraded)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise ModelAdapterError("all model providers failed or were incompatible", last_error)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        last_error: BaseException | None = None
        for provider in self._candidates(request):
            caps = provider.capabilities()
            if request.params.stream and not caps.supports_streaming:
                self._run_mismatch_hooks(provider, request, "streaming")
                continue
            try:
                async for chunk in provider.stream(request):
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._run_failure_hooks(provider, request, exc)
                continue
        raise ModelAdapterError("all streaming providers failed or were incompatible", last_error)

    def _candidates(self, request: ModelRequest) -> list[ModelProvider]:
        del request
        return self.providers()

    def _compatibility(self, provider: ModelProvider, request: ModelRequest) -> tuple[bool, str]:
        caps = provider.capabilities()
        if request.tools and not caps.supports_tools:
            return False, "tools"
        if (
            request.tools
            and len(request.tools) > 0
            and request.params.extra.get("parallel_tools")
            and not caps.supports_parallel_tools
        ):
            return False, "parallel_tools"
        if request.params.stream and not caps.supports_streaming:
            return False, "streaming"
        return True, ""

    def _run_failure_hooks(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        error: BaseException,
    ) -> ModelProvider | None:
        for hook in self.hooks:
            replacement = hook.on_failure(provider, request, error)
            if replacement is not None:
                return replacement
        return None

    def _run_mismatch_hooks(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        missing: str,
    ) -> ModelProvider | None:
        for hook in self.hooks:
            replacement = hook.on_capability_mismatch(provider, request, missing)
            if replacement is not None:
                return replacement
        return None
