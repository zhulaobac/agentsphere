"""Anchor Tenant packs: reusable onboarding on top of Control Plane + Evaluation.

A later second tenant should implement :class:`TenantPack` the same way as
``product_team``. Isolation is tenant-id scoped; do not share artifacts,
experiments, gates, or releases across tenants.
"""

from agentsphere.tenants.pack import OnboardedTenant, TenantPack
from agentsphere.tenants.registry import TenantRegistry
from agentsphere.tenants.types import TenantRecord

__all__ = [
    "OnboardedTenant",
    "TenantPack",
    "TenantRecord",
    "TenantRegistry",
]
