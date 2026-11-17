"""In-process tenant registry with the same isolation rules as Control Plane assets."""

from __future__ import annotations

from agentsphere.controlplane.errors import AuthorizationError, DuplicateAssetError
from agentsphere.controlplane.types import Principal
from agentsphere.tenants.types import TenantRecord


class TenantNotFoundError(KeyError):
    pass


class TenantRegistry:
    """Register / get / list tenants. Principals only see their own tenant_id."""

    def __init__(self) -> None:
        self._tenants: dict[str, TenantRecord] = {}

    def register(self, record: TenantRecord, principal: Principal) -> TenantRecord:
        if principal.tenant != record.tenant_id:
            raise AuthorizationError(
                f"tenant {principal.tenant!r} cannot register tenant {record.tenant_id!r}"
            )
        if principal.owner != record.owner and principal.name != record.owner:
            raise AuthorizationError(
                f"{principal.name!r} cannot register tenant {record.tenant_id!r} owned by {record.owner!r}"
            )
        existing = self._tenants.get(record.tenant_id)
        if existing is not None:
            raise DuplicateAssetError(f"tenant {record.tenant_id!r} is already registered")
        stored = record.model_copy(deep=True)
        self._tenants[stored.tenant_id] = stored
        return stored.model_copy(deep=True)

    def get(self, tenant_id: str, principal: Principal) -> TenantRecord:
        if principal.tenant != tenant_id:
            raise AuthorizationError(
                f"tenant {principal.tenant!r} cannot read tenant {tenant_id!r}"
            )
        record = self._tenants.get(tenant_id)
        if record is None:
            raise TenantNotFoundError(tenant_id)
        return record.model_copy(deep=True)

    def list(self, principal: Principal) -> list[TenantRecord]:
        item = self._tenants.get(principal.tenant)
        if item is None:
            return []
        return [item.model_copy(deep=True)]

    def contains(self, tenant_id: str) -> bool:
        return tenant_id in self._tenants
