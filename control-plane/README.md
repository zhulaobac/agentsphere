# Control Plane (not in P0 / this P1 slice)

This directory is reserved for future **Go** control-plane services:

- Artifact Registry, identity/owner, versioning, permissions
- Release pipeline, quota/cost, tenant management
- Interfaces that talk to Runtime via `RuntimeProvider` (see `python/agentsphere/runtime/interfaces.py`)

P1 implements recovery, sandbox, vault, agent identity, policy, and audit in
Python under `python/agentsphere/`. Do not place Python packages here; keep this
tree as a sibling of `python/` so Go modules can be added without mixing into
the Kernel or Runtime.
