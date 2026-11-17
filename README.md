# AgentSphere

Enterprise agent production platform. This repository is a **monorepo**.

P0 delivers the skeleton from the design: Harness Kernel (ReAct + Workflow),
Model Adapter, basic Runtime (Session / Turn / Event), Tool Registry v1, and
basic Trace. Later phases (control plane, eval/gate, sandbox, memory, multi-agent)
are intentionally not implemented.

## Layout

```
python/agentsphere/          # Python packages (Kernel + Runtime)
  kernel/                    # Harness: loop, context, dispatcher, middleware, events
    loop/                    # ReAct + Workflow strategies; extension registry
  model_adapter/             # Unified model API + mock provider (no network)
  runtime/                   # Session/Turn/Event store, replay, RuntimeProvider
  tool_registry/             # Tool Registry v1 (schema, SemVer, owner, risk, idempotency)
  trace/                     # Session → Turn → Step → Model/Tool Call (OpenTelemetry)
control-plane/               # Reserved for future Go services (not in P0)
tests/                       # Unit + integration tests (offline)
.github/workflows/ci.yml     # GitHub Actions: install deps, run pytest
```

The Kernel is essentially pure: `(session_state, user_input) → events`. It does
not hold durable in-process session state. Runtime appends events to an
in-memory or SQLite store and rebuilds `SessionState` by replaying the log.

## Requirements

- Python 3.11+
- No paid API keys and no external network services are required for tests.

## Setup and tests

```bash
python -m pip install -e ".[dev]"
pytest
```

CI runs the same commands on `main` and on pull requests.

## Public API sketch (matches the design)

```python
from agentsphere import (
    AgentHarness,          # AgentHarness.run_turn(session_state, user_input) → AsyncIterator[Event]
    ReActStrategy,         # LoopStrategy.step(ctx) → CallModel | CallTool | Finish | Handoff
    WorkflowStrategy,
    ModelAdapter,
    MockModelProvider,
    SessionRuntime,
    ToolRegistry,
    TraceRecorder,
)
```

Planner-Executor and Multi-Agent are registered as **unimplemented** strategy
names; add them later with `register_strategy()`.
