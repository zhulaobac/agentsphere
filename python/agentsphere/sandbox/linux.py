"""Linux isolation helpers: rlimits, user/net/mount namespaces, Landlock filesystem policy."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Iterable
from pathlib import Path

from agentsphere.sandbox.limits import ResourceLimits, SandboxSpec

CLONE_NEWUSER = getattr(os, "CLONE_NEWUSER", 0x10000000)
CLONE_NEWNET = getattr(os, "CLONE_NEWNET", 0x40000000)
CLONE_NEWNS = getattr(os, "CLONE_NEWNS", 0x00020000)
CLONE_NEWPID = getattr(os, "CLONE_NEWPID", 0x20000000)

PR_SET_NO_NEW_PRIVS = 38

LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_RULE_PATH_BENEATH = 1

ACCESS_FS_EXECUTE = 1 << 0
ACCESS_FS_WRITE_FILE = 1 << 1
ACCESS_FS_READ_FILE = 1 << 2
ACCESS_FS_READ_DIR = 1 << 3
ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_MAKE_CHAR = 1 << 6
ACCESS_FS_MAKE_DIR = 1 << 7
ACCESS_FS_MAKE_REG = 1 << 8
ACCESS_FS_MAKE_SOCK = 1 << 9
ACCESS_FS_MAKE_FIFO = 1 << 10
ACCESS_FS_MAKE_BLOCK = 1 << 11
ACCESS_FS_MAKE_SYM = 1 << 12
ACCESS_FS_REFER = 1 << 13
ACCESS_FS_TRUNCATE = 1 << 14

FS_READ = ACCESS_FS_EXECUTE | ACCESS_FS_READ_FILE | ACCESS_FS_READ_DIR | ACCESS_FS_REFER
FS_WRITE = (
    ACCESS_FS_WRITE_FILE
    | ACCESS_FS_REMOVE_DIR
    | ACCESS_FS_REMOVE_FILE
    | ACCESS_FS_MAKE_CHAR
    | ACCESS_FS_MAKE_DIR
    | ACCESS_FS_MAKE_REG
    | ACCESS_FS_MAKE_SOCK
    | ACCESS_FS_MAKE_FIFO
    | ACCESS_FS_MAKE_BLOCK
    | ACCESS_FS_MAKE_SYM
    | ACCESS_FS_TRUNCATE
)
FS_ALL = FS_READ | FS_WRITE


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def apply_rlimits(limits: ResourceLimits) -> None:
    try:
        import resource
    except ImportError:
        return
    if limits.cpu_seconds is not None:
        cpu = max(1, int(limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if limits.memory_bytes is not None:
        mem = max(1024 * 1024, int(limits.memory_bytes))
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            attr = getattr(resource, name, None)
            if attr is not None:
                try:
                    resource.setrlimit(attr, (mem, mem))
                except (ValueError, OSError):
                    continue
    if limits.max_processes is not None:
        nproc = max(1, int(limits.max_processes))
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError, AttributeError):
            pass
    if limits.max_open_files is not None:
        nofile = max(8, int(limits.max_open_files))
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
        except (ValueError, OSError):
            pass


def unshare_flags(flags: int) -> bool:
    if hasattr(os, "unshare"):
        try:
            os.unshare(flags)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result = libc.unshare(ctypes.c_int(flags))
        if result == 0:
            return True
        return False
    except Exception:
        return False


def become_mapped_root() -> bool:
    """Write uid/gid maps after CLONE_NEWUSER so the child appears as root inside the namespace."""
    try:
        Path("/proc/self/setgroups").write_text("deny")
    except OSError:
        pass
    uid = os.getuid()
    gid = os.getgid()
    try:
        Path("/proc/self/uid_map").write_text(f"0 {uid} 1")
        Path("/proc/self/gid_map").write_text(f"0 {gid} 1")
        return True
    except OSError:
        return False


def set_no_new_privs() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    except Exception:
        pass


def apply_landlock(writable: Iterable[Path], readable: Iterable[Path]) -> bool:
    """Restrict filesystem access to the given paths. Returns False if Landlock is unavailable."""
    if sys.platform != "linux":
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.syscall.restype = ctypes.c_long
        attr = _RulesetAttr(FS_ALL)
        ruleset_fd = libc.syscall(LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
        if ruleset_fd < 0:
            return False
        try:
            seen: set[str] = set()
            for path in writable:
                _add_path(libc, int(ruleset_fd), Path(path), FS_ALL, seen)
            for path in readable:
                _add_path(libc, int(ruleset_fd), Path(path), FS_READ, seen)
            set_no_new_privs()
            result = libc.syscall(LANDLOCK_RESTRICT_SELF, ctypes.c_int(int(ruleset_fd)), 0)
            return result == 0
        finally:
            os.close(int(ruleset_fd))
    except Exception:
        return False


def _add_path(libc: ctypes.CDLL, ruleset_fd: int, path: Path, access: int, seen: set[str]) -> None:
    resolved = str(Path(path).resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    fd = os.open(resolved, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathBeneath(access, fd)
        libc.syscall(
            LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            0,
        )
    finally:
        os.close(fd)


def default_readable_paths() -> list[Path]:
    paths: list[Path] = []
    for raw in (
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/etc/ssl",
        "/etc/nsswitch.conf",
        "/etc/passwd",
        "/etc/group",
        "/dev/null",
        "/dev/zero",
        "/dev/urandom",
        "/proc/self",
        sys.prefix,
        sys.exec_prefix,
        sys.executable,
    ):
        path = Path(raw)
        if path.exists():
            paths.append(path)
    try:
        paths.append(Path(sys.executable).resolve().parent)
    except OSError:
        pass
    return paths


def isolation_preexec(spec: SandboxSpec, *, workspace: Path | None) -> tuple[bool, bool]:
    """
    Called in the child after fork, before exec. Returns (network_isolated, fs_isolated).
    Must not raise: a failure falls back to whatever isolation succeeded.
    """
    apply_rlimits(spec.limits)
    network_isolated = False
    fs_isolated = False
    if spec.network.default_deny:
        network_isolated = unshare_flags(CLONE_NEWNET)
        if not network_isolated:
            # Try user+net together; some kernels require a user ns to create a net ns.
            if unshare_flags(CLONE_NEWUSER | CLONE_NEWNET):
                become_mapped_root()
                network_isolated = True
    writable = []
    if workspace is not None:
        writable.append(workspace)
    for mount in spec.mounts:
        if not mount.read_only:
            writable.append(Path(mount.host_path))
    readable = list(default_readable_paths())
    readable.extend(Path(p) for p in spec.extra_readable_paths)
    for mount in spec.mounts:
        if mount.read_only:
            readable.append(Path(mount.host_path))
    if workspace is not None:
        readable.append(workspace)
    fs_isolated = False
    if spec.restrict_filesystem:
        fs_isolated = apply_landlock(writable, readable)
    return network_isolated, fs_isolated
