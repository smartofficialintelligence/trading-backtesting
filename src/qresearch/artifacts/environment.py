"""Capture of the execution environment for reproducibility (DEVELOPMENT_PLAN.md sec. 7)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from pydantic import Field

import qresearch
from qresearch.config import FrozenModel
from qresearch.time import UtcDatetime, now_utc

_PACKAGES = ("polars", "duckdb", "pydantic", "pyarrow", "numpy")


class EnvironmentRecord(FrozenModel):
    captured_at: UtcDatetime
    python: str
    platform: str
    qresearch_version: str
    packages: dict[str, str]
    lock_sha256: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_diff_sha256: str | None = None
    seed: int = 0
    thread_settings: dict[str, str] = Field(default_factory=dict)

    @property
    def code_revision(self) -> str | None:
        if self.git_commit is None:
            return None
        if self.git_dirty and self.git_diff_sha256:
            return f"{self.git_commit}-dirty-{self.git_diff_sha256[:12]}"
        return self.git_commit


def repository_root() -> Path:
    """The checkout containing the installed package (editable install), or its parent."""
    return Path(qresearch.__file__).resolve().parents[2]


def capture_environment(*, seed: int = 0, root: Path | None = None) -> EnvironmentRecord:
    root = root or repository_root()
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    lock = root / "uv.lock"
    lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    commit, dirty, diff_sha = _git_state(root)
    threads = {
        k: v
        for k, v in os.environ.items()
        if k in {"POLARS_MAX_THREADS", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"}
    }
    return EnvironmentRecord(
        captured_at=now_utc(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        qresearch_version=qresearch.__version__,
        packages=packages,
        lock_sha256=lock_sha,
        git_commit=commit,
        git_dirty=dirty,
        git_diff_sha256=diff_sha,
        seed=seed,
        thread_settings=threads,
    )


def _git_state(root: Path) -> tuple[str | None, bool | None, str | None]:
    if not (root / ".git").exists():
        return None, None, None
    try:
        commit = _git(root, "rev-parse", "HEAD")
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None, None, None
    dirty = bool(status.strip())
    diff_sha = None
    if dirty:
        diff = _git(root, "diff", "HEAD")
        diff_sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    return commit, dirty, diff_sha


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
