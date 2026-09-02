"""Unit tests for the login-node Slurm job monitor."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cursor/skills/dcu-slurm-submit/scripts"))
from start_job_monitor import (  # noqa: E402
    SENTINEL,
    build_continue_payload,
    classify_completion,
    format_sentinel,
)


def test_classify_running_without_terminal_state(tmp_path):
    result = tmp_path / "result.json"
    done = tmp_path / "DONE"
    assert classify_completion("RUNNING", {"passed": True}, done, result) == "running"
    assert classify_completion("PENDING", None, done, result) == "running"


def test_classify_incomplete_when_result_or_done_missing(tmp_path):
    result = tmp_path / "result.json"
    done = tmp_path / "DONE"
    assert classify_completion("COMPLETED", None, done, result) == "infrastructure-incomplete"
    result.write_text("{}\n")
    assert classify_completion("COMPLETED", {}, done, result) == "infrastructure-incomplete"


def test_classify_incomplete_when_done_is_stale(tmp_path):
    result = tmp_path / "result.json"
    done = tmp_path / "DONE"
    now = time.time()
    result.write_text(json.dumps({"passed": True}) + "\n")
    done.write_text("ok\n")
    import os

    os.utime(done, (now - 10, now - 10))
    os.utime(result, (now, now))
    assert classify_completion("COMPLETED", {"passed": True}, done, result) == "infrastructure-incomplete"


def test_classify_passed_and_failed_complete_results(tmp_path):
    result = tmp_path / "result.json"
    done = tmp_path / "DONE"
    result.write_text("{}\n")
    done.write_text("ok\n")
    now = time.time()
    import os

    os.utime(result, (now - 2, now - 2))
    os.utime(done, (now, now))
    assert classify_completion("COMPLETED", {"passed": True}, done, result) == "completed-passed"
    assert classify_completion("COMPLETED", {"passed": False}, done, result) == "completed-failed"
    assert classify_completion("FAILED", {"passed": False}, done, result) == "completed-failed"


def test_format_sentinel_is_one_line_and_parseable():
    payload = build_continue_payload(
        job_id="119799911",
        slurm_state="COMPLETED",
        exit_code="0:0",
        ledger_state="completed-passed",
        result={"passed": True, "returncode": 0},
        continuation_prompt="regenerate the PDF",
        result_directory="/tmp/out",
    )
    line = format_sentinel(payload)
    assert line.startswith(SENTINEL + " ")
    assert "\n" not in line
    parsed = json.loads(line[len(SENTINEL) + 1 :])
    assert parsed["job_id"] == "119799911"
    assert parsed["passed"] is True
    assert parsed["continuation_prompt"] == "regenerate the PDF"
