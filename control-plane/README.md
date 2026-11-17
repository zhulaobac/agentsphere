# Control Plane (not in P0)

This directory is reserved for future **Go** control-plane services:

- Artifact Registry, identity/owner, versioning, permissions
- Release pipeline, quota/cost, tenant management, audit
- Interfaces that talk to Runtime via `RuntimeProvider` (see `python/agentsphere/runtime/interfaces.py`)

P0 implements only the Python Harness Kernel and a basic Runtime. Do not place
Python packages here; keep this tree as a sibling of `python/` so Go modules
can be added without mixing into the Kernel.
