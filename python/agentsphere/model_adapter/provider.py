"""ModelProvider protocol. New vendors implement this; the adapter stays stable."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from agentsphere.model_adapter.types import ModelCapabilities, ModelChunk, ModelRequest, ModelResponse


@runtime_checkable
class ModelProvider(Protocol):
    """A single inference backend (OpenAI, Anthropic, local, mock, ...)."""

    name: str

    def capabilities(self) -> ModelCapabilities: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]: ...


class FallbackHook(Protocol):
    """Called when a provider fails or cannot satisfy the request."""

    def on_failure(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        error: BaseException,
    ) -> ModelProvider | None:
        """Return a replacement provider, or None to continue the default chain."""

    def on_capability_mismatch(
        self,
        provider: ModelProvider,
        request: ModelRequest,
        missing: str,
    ) -> ModelProvider | None:
        """Return a replacement when `provider` lacks a required capability."""


class ModelAdapterError(RuntimeError):
    """Raised when every provider (including fallbacks) fails or cannot serve the request."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
