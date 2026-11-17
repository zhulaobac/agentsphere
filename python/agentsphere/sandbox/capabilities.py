"""Probe OS capabilities used by the subprocess sandbox. Tests skip only when a capability is absent."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxCapabilities:
    unshare: bool
    user_namespace: bool
    net_namespace: bool
    mount_namespace: bool
    landlock: bool
    setrlimit: bool
    platform: str
    reason: str = ""

    @property
    def network_isolation(self) -> bool:
        return self.net_namespace

    @property
    def filesystem_isolation(self) -> bool:
        return self.landlock or (self.user_namespace and self.mount_namespace)


def probe() -> SandboxCapabilities:
    if sys.platform != "linux":
        return SandboxCapabilities(
            unshare=False,
            user_namespace=False,
            net_namespace=False,
            mount_namespace=False,
            landlock=False,
            setrlimit=hasattr(os, "setrlimit") if False else _has_resource(),
            platform=sys.platform,
            reason=f"sandbox isolation is implemented for Linux; current platform is {sys.platform}",
        )
    return SandboxCapabilities(
        unshare=_has_unshare(),
        user_namespace=_flag_available("CLONE_NEWUSER"),
        net_namespace=_flag_available("CLONE_NEWNET"),
        mount_namespace=_flag_available("CLONE_NEWNS"),
        landlock=_landlock_available(),
        setrlimit=_has_resource(),
        platform=sys.platform,
    )


def _has_resource() -> bool:
    try:
        import resource  # noqa: F401

        return True
    except Exception:
        return False


def _has_unshare() -> bool:
    return hasattr(os, "unshare") or sys.platform == "linux"


def _flag_available(name: str) -> bool:
    if hasattr(os, name):
        return True
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return hasattr(os, "unshare") or hasattr(libc, "unshare")
    except OSError:
        return False


def _landlock_available() -> bool:
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        create = getattr(libc, "syscall", None)
        if create is None:
            return False
        # SYS_landlock_create_ruleset is 444 on x86_64 / arm64 Linux.
        nr = 444
        attr = _LandlockRulesetAttr()
        attr.handled_access_fs = (1 << 15) - 1
        flags = 0
        result = libc.syscall(nr, ctypes.byref(attr), ctypes.sizeof(attr), flags)
        if result >= 0:
            os.close(result)
            return True
        errno = ctypes.get_errno()
        # ENOSYS (38) / EOPNOTSUPP (95) => kernel or config lacks Landlock.
        return errno not in {38, 95}
    except Exception:
        return False


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]
