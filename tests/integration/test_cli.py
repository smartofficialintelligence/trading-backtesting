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
    config["source_uri"] = str(root / "source" / "synthetic_1m.csv")
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
