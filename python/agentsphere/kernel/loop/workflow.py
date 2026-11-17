"""Workflow: deterministic DAG / state-machine orchestration (no free-form ReAct loop)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from agentsphere.kernel.actions import CallModel, CallTool, Finish, ToolCallRequest
from agentsphere.kernel.context import TurnContext
from agentsphere.kernel.ids import new_id
from agentsphere.kernel.loop.strategy import register_strategy
from agentsphere.model_adapter.types import InferenceParams, Message, MessageRole, ToolDefinition


class WorkflowNode(BaseModel):
    id: str
    type: Literal["model", "tool", "end"]
    prompt: str | None = None
    system_prompt: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None


class WorkflowEdge(BaseModel):
    source: str
    target: str
    when: str | None = None
    """If set, the edge is taken when `when` is a substring of the source node's output."""


class WorkflowDefinition(BaseModel):
    start: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = Field(default_factory=list)

    def node(self, node_id: str) -> WorkflowNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"workflow node not found: {node_id}")

    def successors(self, node_id: str, output_text: str = "") -> list[str]:
        matched: list[str] = []
        unconditional: list[str] = []
        for edge in self.edges:
            if edge.source != node_id:
                continue
            if edge.when is None:
                unconditional.append(edge.target)
            elif edge.when in output_text:
                matched.append(edge.target)
        return matched or unconditional


class WorkflowStrategy:
    """Walk a DAG: each node is a model call, a tool call, or an end state."""

    name = "workflow"

    def __init__(
        self,
        definition: WorkflowDefinition,
        params: InferenceParams | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> None:
        self.definition = definition
        self.params = params or InferenceParams()
        self.tools = list(tools or [])

    def step(self, ctx: TurnContext) -> CallModel | CallTool | Finish:
        state = ctx.session_state
        if not state.workflow_cursor:
            state.workflow_cursor = self.definition.start

        # Bound DAG walks so a cycle cannot livelock the Kernel.
        for _ in range(len(self.definition.nodes) + 1):
            node = self.definition.node(state.workflow_cursor)

            if node.id in state.workflow_node_outputs:
                output_text = _output_text(state.workflow_node_outputs[node.id])
                nxt = self.definition.successors(node.id, output_text)
                if not nxt:
                    return Finish(output=output_text, reason="workflow_complete")
                state.workflow_cursor = nxt[0]
                continue

            if node.type == "end":
                return Finish(output=node.output or _last_output(state), reason="workflow_end")

            if node.type == "model":
                messages = self._model_messages(ctx, node)
                return CallModel(messages=messages, tools=list(self.tools), params=self.params)

            if node.type == "tool":
                if not node.tool_name:
                    raise ValueError(f"workflow tool node '{node.id}' is missing tool_name")
                arguments = _render_arguments(node.arguments, state.workflow_node_outputs, ctx.user_input)
                return CallTool(
                    calls=[
                        ToolCallRequest(
                            id=new_id("call"),
                            name=node.tool_name,
                            arguments=arguments,
                            version=node.tool_version,
                        )
                    ]
                )

            raise ValueError(f"unknown workflow node type: {node.type}")

        return Finish(output=_last_output(state), reason="workflow_cycle")

    def _model_messages(self, ctx: TurnContext, node: WorkflowNode) -> list[Message]:
        if ctx.assembled is not None and ctx.assembled.messages:
            messages = list(ctx.assembled.messages)
            if node.prompt:
                messages.append(Message(role=MessageRole.USER, content=node.prompt))
            return messages
        messages: list[Message] = []
        system = node.system_prompt or ctx.session_state.system_prompt
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        prompt = node.prompt or ctx.user_input
        messages.append(Message(role=MessageRole.USER, content=prompt))
        return messages


def _output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value and value["content"] is not None:
            return str(value["content"])
        if "output" in value and value["output"] is not None:
            return str(value["output"])
    return str(value)


def _last_output(state: Any) -> str:
    if not state.workflow_node_outputs:
        return ""
    last_key = next(reversed(state.workflow_node_outputs))
    return _output_text(state.workflow_node_outputs[last_key])


def _render_arguments(
    template: dict[str, Any],
    outputs: dict[str, Any],
    user_input: str,
) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for key, value in template.items():
        if isinstance(value, str):
            rendered[key] = (
                value.replace("{{user_input}}", user_input).replace(
                    "{{last}}", _output_text(next(reversed(outputs.values()), ""))
                )
            )
        else:
            rendered[key] = value
    return rendered


register_strategy(WorkflowStrategy.name, WorkflowStrategy)
