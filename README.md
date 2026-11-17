# AgentSphere

Enterprise agent production platform. This repository is a **monorepo**.

P0 delivered the skeleton: Harness Kernel (ReAct + Workflow), Model Adapter,
basic Runtime (Session / Turn / Event), Tool Registry v1, and basic Trace.

P1 (this slice) adds Managed Runtime recovery and the security execution
substrate: checkpoints, worker leases, idempotent tool execution, pause /
resume / cancel, HITL, Sandbox, Agent Identity, Credential Vault, policy, and
audit. Control Plane (Artifact/version/release), Evaluation, Memory/RAG, MCP
Gateway, Planner-Executor, Multi-Agent, BYO hosting, cost governance, and
Anchor Tenant integration are still out of scope.

## Layout

```
python/agentsphere/          # Python packages (Kernel + Runtime + security)
  kernel/                    # Harness: loop, context, dispatcher, middleware, events
    loop/                    # ReAct + Workflow strategies; extension registry
  model_adapter/             # Unified model API + mock provider (no network)
  runtime/                   # Session/Turn/Event, Checkpoint, Worker lease, recovery
  tool_registry/             # Tool Registry v1 (schema, SemVer, owner, risk, idempotency)
  trace/                     # Session → Turn → Step → Model/Tool Call (OpenTelemetry)
  sandbox/                   # Sandbox protocol + isolated-subprocess backend
  identity/                  # Agent Identity, Credential Vault, policy, audit
control-plane/               # Reserved for future Go services
tests/                       # Unit + integration tests (offline)
.github/workflows/ci.yml     # GitHub Actions: install deps, run pytest
```

The Kernel is essentially pure: `(session_state, user_input) → events`. It does
not hold durable in-process session state. Runtime appends events only after they
are durable; a step is complete only then. Recovery loads the latest
`Checkpoint`, replays remaining events, and continues under a worker `lease`.

Pluggable production backends (not required in CI): Kafka event log,
Firecracker/gVisor/Kata sandbox, HashiCorp Vault/SPIFFE, OPA/Cedar. Local
SQLite / in-memory / subprocess implementations sit behind the same interfaces.

## Requirements

- Python 3.11+
- No paid API keys and no external network services are required for tests.
- No Firecracker, Kafka, Redis, or external Vault.

## Setup and tests

```bash
python -m pip install -e ".[dev]"
pytest
```

P0 and P1 tests run together. Useful subsets:

```bash
pytest tests/test_runtime_replay.py tests/test_harness.py          # P0
pytest tests/test_checkpoint_and_idempotency.py tests/test_lease.py \
       tests/test_pause_resume_cancel.py tests/test_sandbox.py \
       tests/test_vault_identity_audit.py                          # P1
```

CI runs `pytest` on `main` and on pull requests, and prints sandbox capability
probe output (user/net namespaces, Landlock). Sandbox tests that need those
capabilities skip with a clear reason when they are genuinely absent.

### Sandbox backend notes (ubuntu-latest)

`SubprocessSandbox` applies `resource.setrlimit` and a wall-clock timeout always.
Default-deny outbound network uses a Linux network namespace (`unshare` /
`CLONE_NEWNET`, often together with a user namespace). Filesystem hiding uses
Landlock when the kernel supports it (`restrict_filesystem=True`). Neither
Firecracker nor gVisor is required.

## Public API sketch (matches the design)

```python
from agentsphere import (
    AgentHarness,          # AgentHarness.run_turn(session_state, user_input) → AsyncIterator[Event]
    ReActStrategy,         # LoopStrategy.step(ctx) → CallModel | CallTool | Finish | Handoff
    WorkflowStrategy,
    ModelAdapter,
    MockModelProvider,
    SessionRuntime,        # run_turn / resume / pause_session / cancel_session / acquire_lease
    ToolRegistry,
    TraceRecorder,
    AgentIdentity,         # workload identity, not a developer account
    InMemoryVault,         # Credential Vault (short-lived tokens; secrets never enter model context)
    StaticPolicyEngine,    # default-deny policy; high-risk → HITL
    SubprocessSandbox,     # Sandbox: rlimits, default-deny net, scoped filesystem
)
```

Planner-Executor and Multi-Agent are registered as **unimplemented** strategy
names; add them later with `register_strategy()`.
