"""FastAPI app for the local research UI.

Read-only over the run store. Two rules hold the design together:

* **Nothing the UI can do is inaccessible from the CLI.** Today that is trivially true
  because the app only reads; when it grows a backtest launcher it must build the same
  ``BacktestConfig`` the YAML path builds, so a run started in the browser stays
  reproducible with ``qresearch backtest run``. A UI-only code path would quietly void
  the reproducibility guarantees the rest of the platform is built on.
* **Local by default.** It binds to loopback and has no authentication, because it exposes
  a filesystem and will eventually execute strategy code. Serving it publicly is not a
  supported configuration.

The static ``report.html`` in each run directory remains the archival artifact; this is
the exploratory view over many runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from qresearch.artifacts.local import LocalArtifactStore
from qresearch.artifacts.report import build_report_for_run
from qresearch.logging import get_logger
from qresearch.research.experiments import fold_table, run_table
from qresearch.ui.pages import compare_page, not_found, runs_page

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger(__name__)


class UIUnavailableError(RuntimeError):
    """Raised when the optional UI dependencies are not installed."""


_MISSING = "the UI needs the optional 'ui' extra: uv sync --extra ui"


def create_app(runs_root: Path | str = "runs", data_root: Path | str = "data") -> FastAPI:
    """Build the app for one run store."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by the extra
        raise UIUnavailableError(_MISSING) from error

    store = LocalArtifactStore(runs_root)
    app = FastAPI(title="qresearch", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.store = store
    app.state.data_root = Path(data_root)

    def _table(run_ids: list[str] | None = None) -> pl.DataFrame:
        return run_table(store, run_ids)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(runs_page(_table()))

    @app.get("/compare", response_class=HTMLResponse)
    def compare(runs: str = "", role: str = "test") -> HTMLResponse:
        wanted = [r for r in runs.split(",") if r]
        known = [r for r in wanted if store.exists(r)]
        if not known:
            return HTMLResponse(compare_page(pl.DataFrame(), {}, role), status_code=200)
        curves: dict[str, pl.DataFrame] = {}
        for run_id in known:
            try:
                curves[run_id] = store.load_frame(run_id, "equity_curve")
            except (FileNotFoundError, OSError):
                continue
        return HTMLResponse(compare_page(_table(known), curves, role))

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(run_id: str) -> HTMLResponse:
        if not store.exists(run_id):
            return HTMLResponse(not_found(f"no run {run_id!r} under {store.root}"), status_code=404)
        # The per-run view is the archival report itself: one renderer, one layout, so the
        # browser and the stored file can never drift apart.
        report = store.path_for(run_id) / "report.html"
        if not report.exists():
            report = build_report_for_run(store, run_id)
        return HTMLResponse(report.read_text(encoding="utf-8"))

    @app.get("/api/runs")
    def api_runs() -> JSONResponse:
        return JSONResponse(_table().write_json())

    @app.get("/api/runs/{run_id}/folds")
    def api_folds(run_id: str) -> Response:
        if not store.exists(run_id):
            return JSONResponse({"error": "unknown run"}, status_code=404)
        return Response(fold_table(store, [run_id]).write_json(), media_type="application/json")

    @app.get("/api/runs/{run_id}/ledger/{name}")
    def api_ledger(run_id: str, name: str, limit: int = 1000) -> Response:
        """A run's Parquet ledger as JSON, for ad-hoc inspection."""
        allowed = {"fills", "trades", "orders", "positions", "equity_curve", "warnings", "folds"}
        if name not in allowed:
            return JSONResponse(
                {"error": f"unknown ledger; one of {sorted(allowed)}"}, status_code=400
            )
        if not store.exists(run_id):
            return JSONResponse({"error": "unknown run"}, status_code=404)
        try:
            frame = store.load_frame(run_id, name)
        except (FileNotFoundError, OSError):
            return JSONResponse({"error": f"run has no {name} ledger"}, status_code=404)
        return Response(frame.head(limit).write_json(), media_type="application/json")

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "runs": len(store.list_runs()), "root": str(store.root)}
        )

    return app


def serve(
    runs_root: Path | str = "runs",
    data_root: Path | str = "data",
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the app with uvicorn. Loopback by default; see the module docstring."""
    try:
        import uvicorn
    except ModuleNotFoundError as error:  # pragma: no cover
        raise UIUnavailableError(_MISSING) from error
    application = create_app(runs_root, data_root)
    log.info(
        "serving the UI", extra={"fields": {"host": host, "port": port, "runs": str(runs_root)}}
    )
    uvicorn.run(application, host=host, port=port, log_level="warning")
