#!/usr/bin/env python3
"""Login-node monitor for one Slurm test job.

Polls squeue/sacct, updates job-state.json, writes CONTINUE.json, and prints
one SLURM_MONITOR_DONE line so a Cursor session can resume after the compute
job finishes. Do not run this on a DCU node.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
STOP_STATES = {"cancelled", "stopped"}
SENTINEL = "SLURM_MONITOR_DONE"
POLL_SENTINEL = "SLURM_MONITOR_POLL"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def scheduler_state(job_id: str):
    queue = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T|%R"],
        text=True,
        capture_output=True,
        check=False,
    )
    queued = queue.stdout.strip().splitlines()
    if queued:
        state, _, reason = queued[0].partition("|")
        return state.strip().upper(), None, reason.strip() or None
    accounting = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            str(job_id),
            "--format=State,ExitCode",
            "--parsable2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    rows = [line.split("|") for line in accounting.stdout.splitlines() if line.strip()]
    if not rows:
        return "UNKNOWN", None, None
    state = rows[0][0].split()[0].split("+")[0].upper()
    exit_code = rows[0][1] if len(rows[0]) > 1 else None
    return state, exit_code, None


def classify_completion(slurm_state, result, done_path: Path, result_path: Path) -> str:
    """Return ledger state for the current scheduler/result snapshot."""
    state = (slurm_state or "UNKNOWN").upper()
    if state not in TERMINAL_STATES:
        return "running"
    if not result_path.is_file() or not done_path.is_file() or result is None:
        return "infrastructure-incomplete"
    if done_path.stat().st_mtime + 1.0 < result_path.stat().st_mtime:
        return "infrastructure-incomplete"
    if state != "COMPLETED":
        return "completed-failed"
    if result.get("passed"):
        return "completed-passed"
    return "completed-failed"


def build_continue_payload(
    *,
    job_id: str,
    slurm_state: str,
    exit_code,
    ledger_state: str,
    result,
    continuation_prompt: str,
    result_directory: str,
) -> dict:
    return {
        "job_id": str(job_id),
        "slurm_state": slurm_state,
        "exit_code": exit_code,
        "ledger_state": ledger_state,
        "passed": bool((result or {}).get("passed")),
        "result": result,
        "result_directory": result_directory,
        "continuation_prompt": continuation_prompt,
        "completed_at": now(),
    }


def format_sentinel(payload: dict) -> str:
    compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{SENTINEL} {compact}"


def default_continuation_prompt(ledger: dict) -> str:
    logical = ledger.get("logical_test_id") or "the submitted test"
    reason = ledger.get("submission_reason") or "the declared test goal"
    return (
        f"Slurm job {ledger.get('job_id')} for {logical} is terminal. "
        f"Read {ledger.get('result_directory')}/CONTINUE.json, result.json, "
        f"and job-state.json. Continue the approved next step for: {reason}. "
        "Do not regenerate perambulators, do not scancel, and do not loosen "
        "the powered-gate. If the result is incomplete, report that instead "
        "of repairing code."
    )


def update_ledger(path: Path, **fields) -> dict:
    ledger = load_json(path) or {}
    ledger.update(fields)
    ledger["last_polled_at"] = now()
    atomic_json(path, ledger)
    return ledger


def emit(line: str) -> None:
    print(line, flush=True)


def run_monitor(args) -> int:
    result_dir = Path(args.result_dir)
    ledger_path = Path(args.ledger) if args.ledger else result_dir / "job-state.json"
    result_path = result_dir / "result.json"
    done_path = result_dir / "DONE"
    continue_path = result_dir / "CONTINUE.json"
    log_path = result_dir / "monitor.log"

    ledger = load_json(ledger_path) or {}
    ledger.setdefault("job_id", str(args.job_id))
    ledger.setdefault("result_directory", str(result_dir))
    ledger["monitor_pid"] = os.getpid()
    ledger["monitor_started_at"] = now()
    if ledger.get("state") not in STOP_STATES:
        ledger["state"] = "monitoring"
    atomic_json(ledger_path, ledger)
    prompt = args.continuation_prompt or ledger.get("continuation_prompt")
    if not prompt:
        prompt = default_continuation_prompt(ledger)
        ledger["continuation_prompt"] = prompt
        atomic_json(ledger_path, ledger)

    delay = max(10, int(args.poll_seconds))
    deadline = None
    if args.max_runtime_seconds and int(args.max_runtime_seconds) > 0:
        deadline = time.time() + int(args.max_runtime_seconds)

    while True:
        ledger = load_json(ledger_path) or ledger
        if ledger.get("state") in STOP_STATES or not ledger.get("automatic_continuation", True):
            emit(f"{POLL_SENTINEL} stopped")
            return 0

        slurm_state, exit_code, reason = scheduler_state(args.job_id)
        result = load_json(result_path)
        ledger_state = classify_completion(slurm_state, result, done_path, result_path)
        update_ledger(
            ledger_path,
            slurm_state=slurm_state,
            exit_code=exit_code,
            hold_reason=reason,
            state=ledger_state if ledger_state != "running" else "monitoring",
            monitor_pid=os.getpid(),
        )
        with log_path.open("a") as log:
            log.write(f"{now()} state={slurm_state} exit={exit_code} ledger={ledger_state} reason={reason}\n")

        if ledger_state != "running":
            payload = build_continue_payload(
                job_id=str(args.job_id),
                slurm_state=slurm_state,
                exit_code=exit_code,
                ledger_state=ledger_state,
                result=result,
                continuation_prompt=prompt,
                result_directory=str(result_dir),
            )
            atomic_json(continue_path, payload)
            emit(format_sentinel(payload))
            return 0

        if args.once:
            emit(
                f"{POLL_SENTINEL} "
                + json.dumps(
                    {
                        "job_id": str(args.job_id),
                        "slurm_state": slurm_state,
                        "ledger_state": ledger_state,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0

        if deadline is not None and time.time() >= deadline:
            emit(f"{POLL_SENTINEL} timeout")
            return 2

        time.sleep(delay)
        delay = min(int(args.max_poll_seconds), max(delay, int(delay * 1.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor one Slurm job from the login node and wake Cursor.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--continuation-prompt", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-poll-seconds", type=int, default=300)
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Classify the current snapshot and exit without sleeping.",
    )
    return run_monitor(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
