"""Session workspace and snapshot references. Production object storage can replace the local backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from agentsphere.kernel.ids import new_id


class WorkspaceSnapshotStore(Protocol):
    def snapshot(self, session_id: str, workspace_root: Path) -> str: ...

    def restore(self, session_id: str, snapshot_ref: str, workspace_root: Path) -> None: ...

    def path_for(self, session_id: str) -> Path: ...


class LocalWorkspaceStore:
    """Copies the workspace directory into a snapshots folder. Fine for tests and local/dev."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        path = self.root / "workspaces" / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def snapshot(self, session_id: str, workspace_root: Path) -> str:
        ref = new_id("snap")
        dest = self.root / "snapshots" / session_id / ref
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        if Path(workspace_root).exists():
            shutil.copytree(workspace_root, dest, dirs_exist_ok=True)
        else:
            dest.mkdir(parents=True, exist_ok=True)
        return ref

    def restore(self, session_id: str, snapshot_ref: str, workspace_root: Path) -> None:
        src = self.root / "snapshots" / session_id / snapshot_ref
        dest = Path(workspace_root)
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists():
            shutil.copytree(src, dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
