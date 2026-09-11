"""The CLI is thin; these tests check wiring and exit codes, not business logic."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from qresearch.cli import app

runner = CliRunner()


def _demo(tmp_path: Path) -> str:
    result = runner.invoke(app, ["data", "demo", "--root", str(tmp_path / "data")])
    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if ln.startswith("dataset "))
    return line.split()[1]


def test_demo_then_inspect_then_verify(tmp_path: Path) -> None:
    dataset_id = _demo(tmp_path)
    root = str(tmp_path / "data")

    inspect = runner.invoke(app, ["data", "inspect", dataset_id, "--root", root])
    assert inspect.exit_code == 0, inspect.output
    assert "timestamp label  bar_start" in inspect.output
    assert "missing_bars" in inspect.output

    verify = runner.invoke(app, ["data", "verify", dataset_id, "--root", root])
    assert verify.exit_code == 0, verify.output


def test_head_requires_an_aware_as_of(tmp_path: Path) -> None:
    dataset_id = _demo(tmp_path)
    root = str(tmp_path / "data")
    naive = runner.invoke(
        app, ["data", "head", dataset_id, "--root", root, "--as-of", "2024-03-04T00:33:00"]
    )
    assert naive.exit_code == 2
    assert "naive" in naive.output

    aware = runner.invoke(
        app, ["data", "head", dataset_id, "--root", root, "--as-of", "2024-03-04T00:33:00Z"]
    )
    assert aware.exit_code == 0, aware.output


def test_unknown_dataset_exits_nonzero(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "inspect", "ds_nope", "--root", str(tmp_path)])
    assert result.exit_code == 2
    assert "no manifest" in result.output


def test_declarative_ingest_matches_the_demo(tmp_path: Path) -> None:
    """The YAML path and the demo path must resolve to the same dataset id."""
    dataset_id = _demo(tmp_path)
    root = tmp_path / "data"

    config = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/examples/ingest_synthetic_crypto.yaml").read_text()
    )
    config["source_uri"] = str(root / "source" / "synthetic_crypto_1m.csv")
    config_path = tmp_path / "ingest.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(app, ["data", "ingest", "-c", str(config_path), "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert f"reused {dataset_id}" in result.output


def test_a_misspelled_config_key_is_rejected(tmp_path: Path) -> None:
    _demo(tmp_path)
    config = yaml.safe_load(
        (Path(__file__).parents[2] / "configs/examples/ingest_synthetic_crypto.yaml").read_text()
    )
    config["policy"]["publication_latnecy"] = config["policy"].pop("publication_latency")
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = runner.invoke(
        app, ["data", "ingest", "-c", str(config_path), "--root", str(tmp_path / "data")]
    )
    assert result.exit_code != 0
    assert "publication_latnecy" in str(result.exception or result.output)


# -- Stage 5 workflow ----------------------------------------------------------------------------


def _run_id_from(
    output: str, scenario: str = "base", rule: str = "next_open_after_eligibility"
) -> str:
    prefix = f"== scenario {scenario} / fill rule {rule}:"
    line = next(ln for ln in output.splitlines() if ln.startswith(prefix))
    return line[len(prefix) :].strip()


def _write_config(tmp_path: Path, example: str, dataset_id: str) -> Path:
    text = (Path(__file__).parents[2] / "configs/examples" / example).read_text()
    path = tmp_path / example
    path.write_text(text.replace("REPLACE_WITH_DATASET_ID", dataset_id))
    return path


def test_crypto_workflow_from_demo_to_reproduce(tmp_path: Path) -> None:
    root, runs = str(tmp_path / "data"), str(tmp_path / "runs")
    dataset_id = _demo(tmp_path)
    config = _write_config(tmp_path, "crypto_momentum.yaml", dataset_id)

    built = runner.invoke(
        app,
        ["features", "build", "-c", str(config), "-o", str(tmp_path / "f.parquet"), "--root", root],
    )
    assert built.exit_code == 0, built.output
    import polars as pl

    features = pl.read_parquet(tmp_path / "f.parquet")
    assert {"ret_1", "vol_10", "rvol_10", "available_at"} <= set(features.columns)
    assert (tmp_path / "f.json").exists()

    ran = runner.invoke(app, ["backtest", "run", "-c", str(config), "--root", root, "--runs", runs])
    assert ran.exit_code == 0, ran.output
    for scenario in ("base", "free", "stressed"):
        for rule in ("next_open_after_eligibility", "open_of_current_bar"):
            assert f"== scenario {scenario} / fill rule {rule}:" in ran.output
    assert "fill rule        next_open_after_eligibility" in ran.output
    assert "fill rule        open_of_current_bar" in ran.output
    assert "optimistic_fill_rule" in ran.output
    assert "[test      ]" in ran.output and "[validation]" in ran.output
    assert "annualisation    24x7" in ran.output
    assert "docs/leakage_checklist.md" in ran.output
    base, free = _run_id_from(ran.output), _run_id_from(ran.output, "free")

    listed = runner.invoke(app, ["runs", "list", "--runs", runs])
    assert listed.exit_code == 0 and base in listed.output

    shown = runner.invoke(app, ["runs", "show", base, "--runs", runs])
    assert (
        shown.exit_code == 0
        and "cost scenario    base" in shown.output
        and "economic digest" in shown.output
    )

    compared = runner.invoke(app, ["runs", "compare", base, free, "--runs", runs, "--role", "test"])
    assert compared.exit_code == 0 and "cost_scenario" in compared.output

    indexed = runner.invoke(
        app, ["runs", "index", "--runs", runs, "--database", str(tmp_path / "idx.duckdb")]
    )
    assert indexed.exit_code == 0 and "runs, run_folds" in indexed.output

    reproduced = runner.invoke(app, ["runs", "reproduce", base, "--root", root, "--runs", runs])
    assert reproduced.exit_code == 0, reproduced.output
    assert "exact" in reproduced.output

    again = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "-c",
            str(config),
            "--root",
            root,
            "--runs",
            runs,
            "--scenario",
            "base",
        ],
    )
    assert again.exit_code == 0 and _run_id_from(again.output) == base, (
        "identical spec reuses the run"
    )


def test_equity_workflow_uses_sessions_and_the_xnys_calendar(tmp_path: Path) -> None:
    root, runs = str(tmp_path / "data"), str(tmp_path / "runs")
    demo = runner.invoke(app, ["data", "demo", "--root", root, "--market", "equity"])
    assert demo.exit_code == 0, demo.output
    dataset_id = next(ln for ln in demo.output.splitlines() if ln.startswith("dataset ")).split()[1]
    assert "outside_session" not in demo.output, "the synthetic equity sample has session bars only"

    inspected = runner.invoke(app, ["data", "inspect", dataset_id, "--root", root])
    assert "calendar XNYS:1" in inspected.output and "equity @ XNYS" in inspected.output

    validated = runner.invoke(app, ["data", "validate", dataset_id, "--root", root])
    assert validated.exit_code == 0, validated.output

    config = _write_config(tmp_path, "equities_mean_reversion.yaml", dataset_id)
    ran = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "-c",
            str(config),
            "--root",
            root,
            "--runs",
            runs,
            "--scenario",
            "base",
        ],
    )
    assert ran.exit_code == 0, ran.output
    assert "annualisation    XNYS" in ran.output
    run_id = _run_id_from(ran.output)
    reproduced = runner.invoke(app, ["runs", "reproduce", run_id, "--root", root, "--runs", runs])
    assert reproduced.exit_code == 0, reproduced.output


def test_unknown_scenario_and_missing_run_exit_nonzero(tmp_path: Path) -> None:
    root, runs = str(tmp_path / "data"), str(tmp_path / "runs")
    dataset_id = _demo(tmp_path)
    config = _write_config(tmp_path, "crypto_momentum.yaml", dataset_id)
    bad = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "-c",
            str(config),
            "--root",
            root,
            "--runs",
            runs,
            "--scenario",
            "wild",
        ],
    )
    assert bad.exit_code == 2 and "unknown cost scenario" in bad.output
    missing = runner.invoke(app, ["runs", "show", "run_nope", "--runs", runs])
    assert missing.exit_code == 2
