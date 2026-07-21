"""Exclusive single-GPU run lock.

Only ONE CUDA training process may touch the 4090 at a time. The lock is a small
JSON file holding the owning pid, tmux session, run/stage, and a heartbeat. A lock
whose pid is no longer alive is *stale* and may be reclaimed (power loss / crash).
The lock never kills a process — it only refuses to hand out a second one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal
    return True


@dataclass
class LockInfo:
    pid: int
    run_id: str
    stage: str
    tmux_session: Optional[str]
    acquired_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {"pid": self.pid, "run_id": self.run_id, "stage": self.stage,
                "tmux_session": self.tmux_session, "acquired_at_utc": self.acquired_at_utc}


class GpuLockError(RuntimeError):
    pass


class GpuLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> Optional[LockInfo]:
        if not self.path.exists():
            return None
        try:
            d = json.loads(self.path.read_text())
            return LockInfo(**d)
        except Exception:
            return None  # corrupt lock treated as absent (will be overwritten on acquire)

    def is_held_by_live_process(self) -> bool:
        info = self.read()
        return info is not None and _pid_alive(info.pid)

    def acquire(self, run_id: str, stage: str, tmux_session: Optional[str],
                now_utc: str, pid: Optional[int] = None) -> LockInfo:
        """Acquire the lock or raise if a *live* process already holds it.

        A stale lock (dead owner) is reclaimed. Write is atomic (tmp -> rename).
        """
        existing = self.read()
        if existing is not None and _pid_alive(existing.pid):
            if existing.pid == (pid or os.getpid()):
                return existing  # re-entrant for same process
            raise GpuLockError(
                f"GPU lock held by live pid {existing.pid} "
                f"({existing.run_id}/{existing.stage}); refusing second GPU job")
        info = LockInfo(pid=pid or os.getpid(), run_id=run_id, stage=stage,
                        tmux_session=tmux_session, acquired_at_utc=now_utc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info.to_dict(), indent=2))
        os.replace(tmp, self.path)
        return info

    def release(self, pid: Optional[int] = None) -> bool:
        """Release only if we own it (or it is stale). Returns True if removed."""
        info = self.read()
        if info is None:
            return False
        me = pid or os.getpid()
        if info.pid == me or not _pid_alive(info.pid):
            self.path.unlink(missing_ok=True)
            return True
        return False
