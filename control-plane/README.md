# Control Plane (Go services — reserved)

This directory is reserved for future **Go** control-plane services:

- Artifact Registry, identity/owner, versioning, permissions
- Release pipeline, quota/cost, tenant management
- Interfaces that talk to Runtime via `RuntimeProvider` (see `python/agentsphere/runtime/interfaces.py`)

The P1 Python Control Plane (Artifact / version / release, tenant scoping,
Managed Runtime Provider) lives under `python/agentsphere/controlplane/`. The
evaluation foundation (datasets, evaluators, Experiment, Gate) lives under
`python/agentsphere/evaluation/`. Do not place Python packages here; keep this
tree as a sibling of `python/` so a later Go module can implement the same
contracts without mixing into the Kernel or Runtime.
