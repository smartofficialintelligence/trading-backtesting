"""Structured logging carries the run id through the context."""

from __future__ import annotations

import io
import json
import logging

from qresearch.logging import configure, current_run_id, get_logger, run_context


def test_json_lines_carry_the_run_id() -> None:
    stream = io.StringIO()
    configure(logging.INFO, json_lines=True, stream=stream)
    log = get_logger("test")
    log.info("outside")
    with run_context("run_abc"):
        log.info("inside", extra={"fields": {"fold": 2}})
        assert current_run_id() == "run_abc"
    assert current_run_id() is None
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [ln["run_id"] for ln in lines] == ["-", "run_abc"]
    assert lines[1]["fold"] == 2 and lines[1]["logger"] == "qresearch.test"


def test_human_format_and_idempotent_configure() -> None:
    stream = io.StringIO()
    configure(logging.INFO, stream=stream)
    configure(logging.INFO, stream=stream)
    with run_context("run_x"):
        get_logger("qresearch.other").warning("careful")
    assert stream.getvalue().count("careful") == 1, "one handler, not two"
    assert "run=run_x" in stream.getvalue()
    configure(logging.WARNING)
