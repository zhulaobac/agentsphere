# AgentSphere

Enterprise agent production platform. This repository is a **monorepo**.

P0 delivered the skeleton: Harness Kernel (ReAct + Workflow), Model Adapter,
basic Runtime (Session / Turn / Event), Tool Registry v1, and basic Trace.

P1 first slice added Managed Runtime recovery and the security execution
substrate: checkpoints, worker leases, idempotent tool execution, pause /
resume / cancel, HITL, Sandbox, Agent Identity, Credential Vault, policy, and
audit.

P1 second slice added the Python Control Plane (unified Agent Artifact,
versioned assets, lifecycle, canary/full release, rollback, tenant scoping,
Managed Runtime Provider) and the Evaluation foundation (datasets, pluggable
evaluators, Experiment, regression, Release Gate).

P1 third slice onboards the first Anchor Tenant: **product-team**
(display name 产品团队). Its first workflow is requirements research and
competitive analysis (需求调研和竞品分析). A later second tenant should
implement the same `TenantPack` interface. Memory/RAG, MCP Gateway,
Planner-Executor, Multi-Agent, BYO hosting, cost governance, online sampling,
the evolver, and a second production Anchor Tenant remain out of scope.

This tree also adds an OpenAI-compatible Chat Completions `ModelProvider` behind
the existing `ModelAdapter`. It is **off by default**. CI and official tests stay
offline (scripted HTTP transport or a loopback fake server). Do not put API keys
in env, git, events, traces, or model context; load them from the Credential
Vault at call time.

## Layout

```
python/agentsphere/          # Python packages (Kernel + Runtime + security + P1 CP/Eval)
  kernel/                    # Harness: loop, context, dispatcher, middleware, events
    loop/                    # ReAct + Workflow strategies; extension registry
  model_adapter/             # Unified model API + mock + OpenAI-compatible provider (live off)
  runtime/                   # Session/Turn/Event, Checkpoint, Worker lease, recovery
  tool_registry/             # Tool Registry v1 (schema, SemVer, owner, risk, idempotency)
  trace/                     # Session → Turn → Step → Model/Tool Call (OpenTelemetry)
  sandbox/                   # Sandbox protocol + isolated-subprocess backend
  identity/                  # Agent Identity, Credential Vault, policy, audit
  controlplane/              # Artifact Registry, lifecycle, release, Runtime Provider
  evaluation/                # Datasets, evaluators, Experiment, regression, Gate
  tenants/                   # Anchor Tenant packs (product-team first; reusable TenantPack)
control-plane/               # Reserved for future Go services (same contracts)
tests/                       # Unit + integration tests (offline)
.github/workflows/ci.yml     # GitHub Actions: pytest + fixture Release Gate
```

The Kernel is essentially pure: `(session_state, user_input) → events`. It does
not hold durable in-process session state. Runtime appends events only after they
are durable; a step is complete only then. Recovery loads the latest
`Checkpoint`, replays remaining events, and continues under a worker `lease`.

Pluggable production backends (not required in CI): Kafka event log,
Firecracker/gVisor/Kata sandbox, HashiCorp Vault/SPIFFE, OPA/Cedar, Kubernetes
deploy, LLM-as-Judge providers. Local SQLite / in-memory / subprocess /
scripted-evaluator implementations sit behind the same interfaces.

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
       tests/test_vault_identity_audit.py                          # P1 recovery/sandbox/vault
pytest tests/test_controlplane_artifacts.py tests/test_lifecycle.py \
       tests/test_controlplane_auth.py tests/test_release.py \
       tests/test_evaluation.py tests/test_experiment.py \
       tests/test_gate.py tests/test_ci_gate.py                          # P1 CP + Eval
pytest tests/test_product_team_tenant.py tests/test_product_team_isolation.py
                                                                         # P1 first Anchor Tenant
pytest tests/test_openai_compat_provider.py                       # OpenAI-compatible provider (offline)
python -m agentsphere.evaluation.ci_gate                          # fixture Release Gate
python -m agentsphere.tenants.product_team.ci                     # product-team canary → Full
```

CI runs `pytest` then `python -m agentsphere.evaluation.ci_gate` on `main` and
on pull requests, and prints sandbox capability probe output (user/net
namespaces, Landlock). Sandbox tests that need those capabilities skip with a
clear reason when they are genuinely absent.

### OpenAI-compatible model provider (optional, default off)

`ModelAdapter` still talks only to `ModelProvider`. Official tests use
`MockModelProvider` or `OpenAICompatibleProvider` with `ScriptedHttpTransport`
(no sockets) / a loopback fake server. They never call a paid vendor.

To point a **local** smoke run at OpenAI or any Chat Completions-compatible
vendor (Chinese or otherwise), keep the API key in the Credential Vault and set:

| Setting | Example |
|---|---|
| `AGENTSPHERE_MODEL_PROVIDER` | `openai-compat` (unset or `mock` in CI) |
| `AGENTSPHERE_OPENAI_BASE_URL` | `https://api.openai.com/v1` or another vendor's compatible `/v1` |
| `AGENTSPHERE_OPENAI_MODEL` | vendor model id |
| `AGENTSPHERE_OPENAI_SECRET_NAME` | vault secret **name** (not the key) |

```python
from agentsphere.identity import AgentIdentity, InMemoryVault
from agentsphere.model_adapter import ModelAdapter, resolve_model_provider

identity = AgentIdentity(name="smoke-agent")
vault = InMemoryVault()
await vault.put("openai_api_key", api_key, agent_id=identity.agent_id)

# allow_network=True is required for a public vendor. Never set this in CI.
provider = resolve_model_provider(
    vault=vault,
    agent_id=identity.agent_id,
    allow_network=True,
)
adapter = ModelAdapter(provider)
```

A missing key, empty key, missing base URL/model, or a live provider without an
injected transport / `allow_network=True` **fails closed**. Do not commit keys,
`.env` files, or sample tokens that look real. `StdlibHttpTransport` refuses
non-loopback hosts unless `allow_public_network=True`.

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
    MockModelProvider,     # default / CI
    OpenAICompatibleProvider,  # Chat Completions; vault key + injected HTTP; default off
    SessionRuntime,        # run_turn / resume / pause_session / cancel_session / acquire_lease
    ToolRegistry,
    TraceRecorder,
    AgentIdentity,         # workload identity, not a developer account
    InMemoryVault,         # Credential Vault (short-lived tokens; secrets never enter model context)
    StaticPolicyEngine,    # default-deny policy; high-risk → HITL
    SubprocessSandbox,     # Sandbox: rlimits, default-deny net, scoped filesystem
    ControlPlane,          # Artifact Registry, lifecycle, canary/full release, rollback
    AgentArtifact,         # apiVersion agentsphere/v1, kind Agent
    ManagedRuntimeProvider,# Control Plane → existing Managed Runtime (not a second runtime)
    ExperimentRunner,      # (agent version, dataset version, evaluator version, env)
    ReleaseGate,           # mandatory check before Approved → Released
    ProductTeamPack,       # first Anchor Tenant (product-team / 产品团队)
    TenantRegistry,        # tenant records; principals only see their own tenant
)
```

Planner-Executor and Multi-Agent are registered as **unimplemented** strategy
names; add them later with `register_strategy()`.
