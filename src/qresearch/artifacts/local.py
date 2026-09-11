"""Local, append-only run store with atomic finalisation.

Layout::

    <root>/<run_id>/            complete runs only
    <root>/.tmp/<run_id>-<n>/   in progress
    <root>/failed/<run_id>-<t>/ failed runs, with status.json carrying the error
    <root>/conflicts/<run_id>-<t>/  a rerun that produced different economics

A run directory appears under ``<root>`` only by a rename from ``.tmp``, so a reader
never sees a partially written run. Rerunning an identical spec reuses the existing run
if its economic digest matches; a different digest is a reproducibility failure and is
kept aside rather than overwriting (ARCHITECTURE.md sec. 2, "immutable inputs and
append-only results").
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qresearch.artifacts.contracts import RunResult, RunSpec, RunStatus
from qresearch.artifacts.environment import EnvironmentRecord
from qresearch.config import FrozenModel
from qresearch.features.transforms import FittedState
from qresearch.ids import RunId, content_hash_full
from qresearch.time import now_utc


class ReproducibilityError(RuntimeError):
    """The same spec produced different economics on a rerun."""


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: RunId
    directory: Path
    spec: RunSpec
    started_at: _dt.datetime


class LocalArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # -- reading ---------------------------------------------------------------------

    def path_for(self, run_id: RunId | str) -> Path:
        return self.root / str(run_id)

    def exists(self, run_id: RunId | str) -> bool:
        return (self.path_for(run_id) / "status.json").exists()

    def list_runs(self) -> tuple[RunId, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            sorted(
                RunId(p.name)
                for p in self.root.iterdir()
                if p.is_dir() and (p / "status.json").exists()
            )
        )

    def load_spec(self, run_id: RunId | str) -> RunSpec:
        return RunSpec.model_validate_json(_read(self.path_for(run_id) / "run_spec.json"))

    def load_result(self, run_id: RunId | str) -> RunResult:
        return RunResult.model_validate_json(_read(self.path_for(run_id) / "result.json"))

    def load_environment(self, run_id: RunId | str) -> EnvironmentRecord:
        return EnvironmentRecord.model_validate_json(
            _read(self.path_for(run_id) / "environment.json")
        )

    def load_frame(self, run_id: RunId | str, name: str) -> pl.DataFrame:
        return pl.read_parquet(self.path_for(run_id) / f"{name}.parquet")

    def load_fitted_states(self, run_id: RunId | str) -> list[FittedState]:
        directory = self.path_for(run_id) / "fitted"
        if not directory.exists():
            return []
        return [FittedState.model_validate_json(_read(p)) for p in sorted(directory.glob("*.json"))]

    # -- writing ---------------------------------------------------------------------

    def begin(self, spec: RunSpec, environment: EnvironmentRecord) -> RunHandle:
        started = now_utc()
        tmp_root = self.root / ".tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        directory = tmp_root / f"{spec.run_id}-{started.strftime('%Y%m%dT%H%M%S%f')}-{os.getpid()}"
        directory.mkdir()
        _write(directory / "run_spec.json", spec)
        _write(directory / "environment.json", environment)
        _write_json(
            directory / "status.json",
            {"status": RunStatus.RUNNING.value, "started_at": started.isoformat()},
        )
        return RunHandle(run_id=spec.run_id, directory=directory, spec=spec, started_at=started)

    def finalize(
        self,
        handle: RunHandle,
        result: RunResult,
        frames: Mapping[str, pl.DataFrame],
        fitted_states: list[FittedState] = (),  # type: ignore[assignment]
    ) -> RunResult:
        """Persist everything and publish the run atomically. Returns the stored result."""
        for name, frame in frames.items():
            frame.write_parquet(handle.directory / f"{name}.parquet", compression="zstd")
        if fitted_states:
            fitted = handle.directory / "fitted"
            fitted.mkdir(exist_ok=True)
            for state in fitted_states:
                _write(fitted / f"fold{state.fold:03d}_{state.transform}.json", state)
        digest = economic_digest(frames)
        files = tuple(
            sorted(
                p.relative_to(handle.directory).as_posix()
                for p in handle.directory.rglob("*")
                if p.is_file()
            )
        )
        stored = result.model_copy(
            update={
                "status": RunStatus.COMPLETE,
                "finished_at": now_utc(),
                "economic_digest": digest,
                "artifact_files": (*files, "result.json", "metrics.json", "status.json"),
            }
        )
        _write(handle.directory / "result.json", stored)
        _write_json(
            handle.directory / "metrics.json",
            {
                "folds": [f.model_dump(mode="json") for f in stored.folds],
                "aggregate": {k: v.model_dump(mode="json") for k, v in stored.aggregate.items()},
            },
        )
        _write_json(
            handle.directory / "status.json",
            {
                "status": RunStatus.COMPLETE.value,
                "started_at": handle.started_at.isoformat(),
                "finished_at": stored.finished_at.isoformat() if stored.finished_at else None,
                "economic_digest": digest,
            },
        )

        target = self.path_for(handle.run_id)
        if target.exists():
            existing = self.load_result(handle.run_id)
            if existing.economic_digest == digest:
                shutil.rmtree(handle.directory)
                return existing
            conflicts = self.root / "conflicts"
            conflicts.mkdir(exist_ok=True)
            kept = conflicts / f"{handle.run_id}-{now_utc().strftime('%Y%m%dT%H%M%S%f')}"
            handle.directory.rename(kept)
            raise ReproducibilityError(
                f"run {handle.run_id} already exists with economic digest "
                f"{(existing.economic_digest or '')[:16]}... but this execution produced "
                f"{digest[:16]}...; the same spec gave different results. The new run is "
                f"kept at {kept} for inspection and the original is untouched."
            )
        handle.directory.rename(target)
        return stored

    def fail(self, handle: RunHandle, error: BaseException) -> RunResult:
        finished = now_utc()
        result = RunResult(
            run_id=handle.run_id,
            status=RunStatus.FAILED,
            started_at=handle.started_at,
            finished_at=finished,
            error=f"{type(error).__name__}: {error}",
        )
        _write(handle.directory / "result.json", result)
        _write_json(
            handle.directory / "status.json",
            {
                "status": RunStatus.FAILED.value,
                "started_at": handle.started_at.isoformat(),
                "finished_at": finished.isoformat(),
                "error": result.error,
            },
        )
        failed = self.root / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        handle.directory.rename(failed / f"{handle.run_id}-{finished.strftime('%Y%m%dT%H%M%S%f')}")
        return result


def economic_digest(frames: Mapping[str, pl.DataFrame]) -> str:
    """Hash of the canonical economic content: fills and the equity curve.

    Independent of Parquet writer details and of bookkeeping columns, so two executions
    of one spec compare on what happened to the money.
    """
    parts: dict[str, list[dict[str, object]]] = {}
    fills = frames.get("fills")
    if fills is not None and not fills.is_empty():
        fill_keys: list[str] = [
            c
            for c in (
                "fold",
                "role",
                "instrument_id",
                "fill_at",
                "side",
                "quantity",
                "price",
                "fee",
            )
            if c in fills.columns
        ]
        parts["fills"] = fills.select(fill_keys).sort(fill_keys).to_dicts()
    curve = frames.get("equity_curve")
    if curve is not None and not curve.is_empty():
        curve_keys: list[str] = [
            c for c in ("fold", "role", "at", "equity", "cash") if c in curve.columns
        ]
        parts["equity_curve"] = curve.select(curve_keys).sort(curve_keys).to_dicts()
    return content_hash_full(parts)


def _write(path: Path, model: FrozenModel) -> None:
    path.write_text(
        json.dumps(json.loads(model.model_dump_json()), indent=2, sort_keys=True), encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")
