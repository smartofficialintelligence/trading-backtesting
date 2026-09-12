"""Load the UI in a real browser.

This exists because the rest of the suite could not catch the failure it is named for: a
syntax error in ``launch.js`` meant the entire launcher script failed to parse, so nothing
on the page worked — and every test still passed, because
``client.get("/static/launch.js").status_code == 200`` only proves a file was served, not
that it is valid JavaScript.

Serving is not working. These tests execute the page.

Needs the ``browser`` extra and an installed chromium:
``uv sync --extra ui --extra browser && uv run playwright install chromium``
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="needs the 'browser' extra")
pytest.importorskip("fastapi", reason="needs the 'ui' extra")

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

pytestmark = pytest.mark.browser

STATIC = Path(__file__).parents[2] / "src" / "qresearch" / "ui" / "static"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    try:
        with sync_playwright() as pw:
            try:
                instance = pw.chromium.launch()
            except PlaywrightError as error:
                pytest.skip(f"chromium not installed: {error}")
            yield instance
            instance.close()
    except PlaywrightError as error:  # pragma: no cover
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small ingested dataset, built once for the whole module."""
    import datetime as dt

    from qresearch.application.ingest import ingest_bars
    from qresearch.data.adapters.base import ColumnMapping, IngestRequest
    from qresearch.data.adapters.csv_parquet import LocalFileBarAdapter
    from qresearch.data.catalog import DatasetCatalog
    from qresearch.data.contracts import AssetClass, PriceAdjustment
    from qresearch.data.manifests import DuplicatePolicy, NormalizationPolicy, TimestampLabel
    from qresearch.data.synthetic import (
        SOURCE_NAME,
        SyntheticSpec,
        instruments_for,
        write_source_csv,
    )

    root = tmp_path_factory.mktemp("data")
    spec = SyntheticSpec()
    source = write_source_csv(root / "source" / "bars.csv", spec)
    request = IngestRequest(
        uri=str(source),
        bar_size="1m",
        policy=NormalizationPolicy(
            timestamp_label=TimestampLabel.BAR_START,
            publication_latency=dt.timedelta(seconds=2),
            price_adjustment=PriceAdjustment.NOT_APPLICABLE,
            duplicates=DuplicatePolicy.KEEP_HIGHEST_REVISION,
            volume_unit="base_asset",
        ),
        instruments=instruments_for(spec),
        source_name=SOURCE_NAME,
        mapping=ColumnMapping(
            trade_count="trades", available_at="available_at", revision="revision"
        ),
    )
    ingest_bars(
        request,
        catalog=DatasetCatalog(root),
        adapter=LocalFileBarAdapter(),
        asset_class=AssetClass.CRYPTO,
        venue="SYNTH",
        calendar_id="24x7:1",
        normalization_version="1",
        created_by="browser-test",
    )
    return root


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory, dataset_root: Path) -> Iterator[str]:
    """A real uvicorn process, not a TestClient: the browser needs a socket."""
    runs = tmp_path_factory.mktemp("runs")
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qresearch.cli",
            "ui",
            "--runs",
            str(runs),
            "--root",
            str(dataset_root),
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"server exited: {process.stdout.read() if process.stdout else ''}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        process.kill()
        raise AssertionError("server did not start")
    yield base
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()
        process.wait(timeout=5)
    finally:
        # Closing the pipe matters: an unclosed one leaks a descriptor and pytest reports
        # it as an unraisable exception at teardown.
        if process.stdout is not None:
            process.stdout.close()


class PageErrors:
    """JavaScript faults and failed requests, kept apart.

    A page error is always a defect. A failed *request* may be the behaviour under test --
    submitting an invalid config is supposed to return 422 -- so the two cannot share a
    single "no errors" assertion without making the negative tests lie.
    """

    def __init__(self) -> None:
        self.js: list[str] = []
        self.requests: list[str] = []

    def attach(self, page: object) -> None:
        page.on("pageerror", lambda e: self.js.append(str(e)))  # type: ignore[attr-defined]
        page.on("console", self._console)  # type: ignore[attr-defined]

    def _console(self, message: object) -> None:
        if message.type != "error":  # type: ignore[attr-defined]
            return
        text = message.text  # type: ignore[attr-defined]
        (self.requests if "Failed to load resource" in text else self.js).append(f"console: {text}")


def open_page(browser: object, url: str) -> tuple[object, PageErrors]:
    page = browser.new_page()  # type: ignore[attr-defined]
    errors = PageErrors()
    errors.attach(page)
    page.goto(url, wait_until="networkidle")
    return page, errors


# -- the regression this file exists for ------------------------------------------------


@pytest.mark.parametrize("name", sorted(p.name for p in STATIC.glob("*.js")))
def test_every_script_parses(name: str) -> None:
    """A script that does not parse takes the whole page with it, silently."""
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available for a syntax check")
    result = subprocess.run([node, "--check", str(STATIC / name)], capture_output=True, text=True)
    assert result.returncode == 0, f"{name} is not valid JavaScript:\n{result.stderr}"


@pytest.mark.parametrize("path", ["/", "/new", "/jobs", "/compare"])
def test_pages_load_without_javascript_errors(browser: object, server: str, path: str) -> None:
    page, errors = open_page(browser, server + path)
    page.wait_for_timeout(500)
    assert errors.js == [], f"{path} raised: {errors.js}"
    assert errors.requests == [], f"{path} failed to load: {errors.requests}"
    page.close()


# -- the launcher actually works -----------------------------------------------------------


def test_the_launcher_populates_itself_from_the_api(browser: object, server: str) -> None:
    """Empty dropdowns are what a broken script looks like from the outside."""
    page, errors = open_page(browser, server + "/new")
    page.wait_for_timeout(800)
    assert errors.js == [] and errors.requests == []
    assert page.locator("#dataset option").count() >= 1, "no datasets in the picker"
    assert page.locator("#instruments option").count() >= 1, "no instruments"
    assert page.locator("#features .chip").count() >= 1, "no feature row"
    assert page.locator("#strategy .chip").count() == 1, "no strategy row"
    assert page.locator(".scen").count() >= 1 and page.locator(".fr").count() == 2
    assert page.locator("#plan [data-p]").count() >= 5
    assert "bars" in page.locator("#dsinfo").inner_text()
    page.close()


def test_choosing_a_component_rerenders_its_parameters(browser: object, server: str) -> None:
    page, errors = open_page(browser, server + "/new")
    page.wait_for_timeout(800)
    page.click("#addfeature")
    page.wait_for_timeout(200)
    assert page.locator("#features .chip").count() == 2
    row = page.locator("#features .chip").nth(1)
    row.locator(".kind").select_option("rolling_volatility")
    page.wait_for_timeout(200)
    names = {
        row.locator("[data-p]").nth(i).get_attribute("data-p")
        for i in range(row.locator("[data-p]").count())
    }
    assert "window" in names, f"parameters did not re-render: {names}"
    assert errors.js == []
    page.close()


def test_preview_renders_a_fold_ribbon(browser: object, server: str) -> None:
    page, errors = open_page(browser, server + "/new")
    page.wait_for_timeout(800)
    for name, value in (
        ("train", "PT1H"),
        ("validation", ""),
        ("test", "PT30M"),
        ("purge", "PT1M"),
        ("label_horizon", "PT1M"),
        ("warmup", ""),
    ):
        page.fill(f'#plan [data-p="{name}"]', value)
    page.click("#preview")
    page.wait_for_timeout(2500)
    assert "simulations" in page.locator("#status").inner_text(), page.locator(
        "#status"
    ).inner_text()
    assert page.locator("#previewout svg").count() == 1, "no fold ribbon"
    assert page.locator("#yaml").text_content(), "the config preview is empty"
    assert errors.js == [] and errors.requests == []
    page.close()


def test_an_impossible_plan_is_reported_in_the_page(browser: object, server: str) -> None:
    page, errors = open_page(browser, server + "/new")
    page.wait_for_timeout(800)
    page.fill('#plan [data-p="train"]', "P30D")
    page.click("#preview")
    page.wait_for_timeout(2000)
    status = page.locator("#status").inner_text().lower()
    assert "no fold fits" in status or "invalid" in status, f"silent failure: {status!r}"
    assert errors.js == [], "rejecting a bad plan must not throw"
    assert len(errors.requests) == 1, "the 422 is the behaviour under test"
    page.close()
