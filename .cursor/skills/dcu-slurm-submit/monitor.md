# Login-node job monitor

## Outputs

| File | Meaning |
|---|---|
| `$RESULT_DIR/monitor_job.sh` | Wrapper generated at `sbatch` time |
| `$RESULT_DIR/job-state.json` | Ledger; `state` becomes `monitoring` then a terminal class |
| `$RESULT_DIR/monitor.log` | One poll line per check |
| `$RESULT_DIR/CONTINUE.json` | Written once, when Slurm is terminal and the result gate is decided |

## Wake line

The monitor prints exactly one completion line:

```text
SLURM_MONITOR_DONE {"job_id":"...","ledger_state":"completed-passed",...}
```

Cursor must attach `notify_on_output.pattern` = `^SLURM_MONITOR_DONE`.
While the job is still running, `--once` prints `SLURM_MONITOR_POLL` instead.

## `CONTINUE.json`

```json
{
  "job_id": "119799911",
  "slurm_state": "COMPLETED",
  "exit_code": "0:0",
  "ledger_state": "completed-passed",
  "passed": true,
  "result": {"passed": true, "returncode": 0},
  "result_directory": "/public/home/siyangchen/BASE/STDIN_...",
  "continuation_prompt": "approved next step",
  "completed_at": "2026-08-24T..."
}
```

`ledger_state` is one of: `running`, `completed-passed`, `completed-failed`,
`infrastructure-incomplete`. On the last three, the agent reads this file
and executes `continuation_prompt` through the skill's completion gate.
