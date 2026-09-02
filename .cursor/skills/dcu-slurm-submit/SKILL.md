---
name: dcu-slurm-submit
description: >-
  Safely submit and monitor EasyDistillation / qedinf test jobs on Sugon DCU
  via Slurm (partition kshdnormal, gres=dcu:1), then resume bounded analysis or
  repair only after terminal, complete test results. Use for GPU/CuPy/HIP tests,
  localized blending verification, sbatch, result monitors, controlled retries,
  or login-node hipErrorNoDevice / no ROCm-capable device errors.
---

# DCU Slurm Submit (Sugon)

## Critical environment fact

- **Login / interactive nodes have no DCU.** Local `python` + CuPy fails with
  `hipErrorNoDevice` / `hipErrorInvalidDevice` / `C-3000 module is NOT loaded`.
- **DCU is available on compute nodes** of partition `kshdnormal` (`--gres=dcu:1`).
- Any GPU / CuPy / `PropagatorWithCurrent` highmode / current contraction test
  **must** be submitted with `sbatch`, not run on the login node.

## PyQUDA MPI ABI requirement

The installed PyQUDA/QUDA extension links HPC-X OpenMPI `libmpi.so.40`, while the
base conda environment's default `mpi4py` and `mpiexec` use MPICH
`libmpi.so.12`. Mixing them makes mpi4py initialize MPICH while QUDA observes its
own OpenMPI as uninitialized and aborts at `communicator_mpi.cpp:46`.

All EasyDistillation PyQUDA jobs must therefore use the project helper
`test/dcu_smoke_mpi.py` or an equivalent environment with:

```bash
MPI4PY_MPIABI=openmpi
PYTHONPATH=/public/home/siyangchen/.local/mpi4py-hpcx-wheel-py311:$PYTHONPATH
PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:$LD_LIBRARY_PATH
/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin/mpiexec -n <grid-product> python ...
```

The MPI rank count must equal the four-dimensional grid product. Do not invoke
nested `srun` from a Python wrapper, and do not use the conda MPICH `mpiexec` for
PyQUDA. The validated initialization probe is `test/pyquda_hpcx_init_probe.py`.

## Canonical tools (repo)

| Tool | Path |
|------|------|
| Job generator | `EasyDistillation/generate_slurm_job.sh` |
| Output query | `EasyDistillation/query_job_output.sh` |
| Historical hand template | `/public/home/siyangchen/qedinf/dcu.slurm` |
| Example generated job | `EasyDistillation/test_calc_calc_disp_consistency.slurm` |

Working directory for submit helpers: `EasyDistillation/`.

## Defaults (from submission history)

| Option | Default |
|--------|---------|
| Partition | `kshdnormal` |
| GRES | `dcu:1` |
| Nodes | `1` |
| ntasks-per-node | `1` |
| cpus-per-task | `8` |
| Time | `72:00:00` |
| Comment | `BASE` |
| Output root | `$HOME/BASE` (override with `BASE=...`) |
| Output dir | `$BASE/STDIN_MMDD_HHMMSS/` |
| Python env | `conda activate my_env` (via `~/.bashrc` modules) |

Modules typically loaded from bashrc on this account:

```bash
module use /public/share/ybyang/modules
module purge
module load compiler/cmake/3.23.3
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
module load compiler/dtk/25.04
module load quda/250425
conda activate my_env
```

Job scripts should inherit login env via bashrc, or explicitly source the same
modules inside the job if non-interactive shells skip bashrc.

## Submission safety policy

Before every GPU test submission, classify it as a **test**, never a production
run, and enforce this pre-approved envelope unless the user explicitly approves
a different one:

| Resource | Pre-approved limit |
|----------|--------------------|
| Template | `generate_slurm_job.sh` output or the historical template listed above |
| Partition | `kshdnormal` only |
| Account | `ybyang` only |
| GRES | at most `dcu:1` per job |
| Nodes | `1` |
| CPUs | at most `8` CPUs/task |
| Wall time | at most `72:00:00` |
| Automatic concurrency | at most one runnable DCU test item at a time |
| Automatic attempts | `MAX_AUTO_ATTEMPTS=2` total submissions per logical test item |

A user-approved dependency chain may be submitted as one logical test item, but
automation must not create parallel runnable branches. Never increase GPUs,
wall time, concurrency, partition, account, or attempt budget to work around a
queue or test failure. Ask the user first.

Before `sbatch`, require:

1. an absolute test entry-script path and a dedicated result directory;
2. valid `$OUT/.portal/job_portal.var` and `job_interface.var` files;
3. a declared submission reason and logical test-item ID;
4. the current `git rev-parse HEAD` value and whether the worktree is dirty;
5. a result contract in which the job atomically writes `result.json`, then
   creates `DONE` only after every required artifact has been flushed.

Do not submit if any item is missing. Do not submit production generation,
physics-production contractions, deployment, or destructive cleanup through
this automatic workflow.

## Standard submit workflow

Match historical usage:

```bash
cd /public/home/siyangchen/qedinf/EasyDistillation

# Generate under $HOME/BASE/STDIN_MMDD_HHMMSS/ and submit
sh generate_slurm_job.sh /absolute/path/to/script.py -s

# Or generate only, then sbatch the printed path
sh generate_slurm_job.sh /absolute/path/to/script.py
sbatch /public/home/siyangchen/BASE/STDIN_*/<job_name>.slurm
```

Useful flags:

```bash
sh generate_slurm_job.sh script.py -j my_job -t 24:00:00 -g dcu:1 -s
sh generate_slurm_job.sh script.py -o /public/home/siyangchen/BASE/STDIN_custom -s
```

Immediately after `sbatch` succeeds, create or update a durable ledger at
`$RESULT_DIR/job-state.json` **and start the login-node monitor program**.
Do not end the turn after submit without a running monitor. Record at least:

```json
{
  "logical_test_id": "localized-psp-smoke",
  "job_id": "123456789",
  "git_commit": "<git rev-parse HEAD>",
  "worktree_dirty": true,
  "result_directory": "/absolute/path/to/results",
  "submission_reason": "validate full-time PSP generation",
  "attempt": 1,
  "max_attempts": 2,
  "state": "submitted",
  "monitor_worker_id": null,
  "development_worker_id": null,
  "automatic_continuation": true
}
```

Write ledger updates atomically (`temporary file` then `mv`). Never overwrite or
delete prior `result.json`, logs, or attempt directories; each retry gets a new
attempt subdirectory and job ID.

### Portal vars requirement

Generated scripts `source $MIDFILE_DIR/job_portal.var` and `job_interface.var`.
If `.portal/` is empty, copy a template from a previous successful job, e.g.:

```bash
OUT=/public/home/siyangchen/BASE/STDIN_<stamp>
mkdir -p "$OUT/.portal"
cp /public/home/siyangchen/BASE/STDIN_1222_204826/.portal/job_portal.var "$OUT/.portal/"
cp /public/home/siyangchen/BASE/STDIN_1222_204826/.portal/job_interface.var "$OUT/.portal/"
# Then edit GAP_WORK_DIR / GAP_STD_* paths inside job_portal.var to match $OUT
```

Minimal fields historically required in `job_portal.var`:

- `GAP_WORK_DIR`, `GAP_STD_OUT_FILE`, `GAP_STD_ERR_FILE`
- `GAP_QUEUE=kshdnormal`, `GAP_NDCU=1`, `GAP_WALL_TIME=72:00:00`
- `GAP_NNODE=1`, `GAP_PPN=1`, `GAP_JOB_NAME=...`

See [portal-template.md](portal-template.md) for a copy-paste stub.

## Query job output

Historical pattern after submit:

```bash
cd /public/home/siyangchen/qedinf/EasyDistillation
./query_job_output.sh -j <JOBID>
./query_job_output.sh -j <JOBID> -o /public/home/siyangchen/BASE/STDIN_<stamp>
./query_job_output.sh -j <JOBID> -t out -l 100
./query_job_output.sh -j <JOBID> -t err
```

Also:

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,NodeList -X
```

## Post-submit monitor program (required)

`sbatch` only starts the compute job. After the ledger exists, produce and
start exactly one **login-node monitor program** so work continues when that
job finishes. A Cursor subagent by itself is not enough: it dies with the
turn. The monitor is a real process whose stdout can wake this session.

Canonical program:

`EasyDistillation/.cursor/skills/dcu-slurm-submit/scripts/start_job_monitor.py`

Schema and wake line: [monitor.md](monitor.md)

`generate_slurm_job.sh ... -s` writes `$RESULT_DIR/monitor_job.sh` bound to
the new Job ID. If you submitted by hand, write the same wrapper or call the
Python program directly.

Start it from the login node, not the DCU:

```bash
python3 -u /public/home/siyangchen/qedinf/EasyDistillation/.cursor/skills/dcu-slurm-submit/scripts/start_job_monitor.py \
  --job-id <JOBID> \
  --result-dir <RESULT_DIR> \
  --continuation-prompt "<approved next step after this job>"
```

Or `bash $RESULT_DIR/monitor_job.sh --continuation-prompt "..."`.

Required launch method in Cursor:

1. Start the monitor with `block_until_ms: 0` so the shell stays attached.
2. Set `notify_on_output.pattern` to `^SLURM_MONITOR_DONE`.
3. Smoke-check the output file once (must show `monitor_pid` / poll or startup).
4. Record `monitor_pid` and the shell id in `job-state.json`.
5. Put `continuation_prompt` in the ledger. That prompt is the work to do
   after the compute job, not a new research question.

The monitor is read-only except for `job-state.json`, `CONTINUE.json`, and
`monitor.log`. It must:

1. poll `squeue` while pending/running (60s, backoff to 5 minutes);
2. use `sacct -X` for the terminal Slurm state and exit code;
3. require `result.json` then `DONE`, and reject a `DONE` older than the
   result file;
4. write `$RESULT_DIR/CONTINUE.json` and print one line
   `SLURM_MONITOR_DONE {...}`;
5. exit after that line. Do not repair code, scancel, or sbatch.

On `SLURM_MONITOR_DONE`, read `CONTINUE.json` and execute
`continuation_prompt` through the completion gate below. If the session died
before the sentinel, the next user turn still resumes from `CONTINUE.json`.

Do not hold a blocking shell poll in the parent agent. If the monitor cannot
be started, set `state: monitoring-required`, print the exact command, and
do not claim that work will continue automatically.

### Completion gate

Start a restricted development worker only when **both** conditions hold:

- Slurm is terminal (`COMPLETED`, `FAILED`, `CANCELLED`, `TIMEOUT`,
  `OUT_OF_MEMORY`, `NODE_FAIL`, or another documented terminal state); and
- `result.json` parses and validates, and `DONE` exists after all expected
  artifacts are present.

A complete passing result may trigger a worker to summarize evidence and
continue the already-approved next development step. A complete failing result
may trigger a worker to inspect logs, identify root cause, make a scoped repair,
and run CPU checks. Terminal Slurm state with missing/incomplete results is an
infrastructure/incomplete-result failure: do not start a code-repair worker.
Within budget, the orchestrator may retry the exact pre-approved test without
increasing resources; otherwise ask the user.

### Restricted development worker

The continuation worker receives only the logical test goal, ledger,
`result.json`, relevant logs, retry budget, and an explicit file/scope allowlist.
It may analyze, make a focused repair, and run CPU checks. It may request one
more identical DCU test attempt only when `attempt < max_attempts` and resource
limits remain satisfied. Every retry must create a new ledger attempt record and
new result directory.

It must not automatically:

- commit, push, merge, deploy, or alter branches;
- delete or overwrite results/logs;
- submit production jobs or expand the test scope;
- call `scancel`;
- exceed the resource, concurrency, or retry budget.

After budget exhaustion, resource-limit pressure, ambiguous results, or a
request for broader changes, set `state: awaiting-user` and ask the user.

Healthy DCU startup in stdout looks like:

```text
Using backend: cupy
[GPU MEM][init] used=...GB free=...GB total=15.984GB
```

## Interruption and stop semantics

- Stop local workers with `/subagents-stop` or
  `subagent({ action: "stop", id })` when that tool is available.
- Stopping a local monitor/development worker does **not** cancel its Slurm job.
- Execute `scancel <job-id>` only after an explicit user request naming or
  unambiguously identifying the cluster job.
- To stop automatic continuation without cancelling the cluster job, atomically
  set ledger `state` to `stopped` and `automatic_continuation` to `false`, then
  stop local workers. A user-requested cluster cancellation sets `state` to
  `cancelled` after `sacct` confirms it.
- `stopped` and `cancelled` are absorbing automation states: monitors may record
  final scheduler facts but must never launch development workers or resubmit.
- A plain user message such as “stop” means stop the local automatic chain; it
  does not authorize `scancel` unless the user explicitly asks to cancel the
  cluster job.

## Agent rules for GPU / localized-blending tests

1. **Do not** treat login-node CuPy failures as “no DCU on this machine”.
2. For any test marked `gpu` / needing CuPy / highmode / current contraction:
   - write or reuse a Python entry script under `test/` or `tests/`;
   - make it emit atomic `result.json` followed by `DONE`;
   - submit via `generate_slurm_job.sh ... -s` inside the safety envelope;
   - record the job ledger and start `start_job_monitor.py` / `monitor_job.sh`;
   - continue development only after `SLURM_MONITOR_DONE` and the completion gate.
3. CPU-only theory checks (`test_sampling_weight`, pure NumPy formula tests) may
   still run on the login node.
4. Prefer absolute paths to Python scripts when calling `generate_slurm_job.sh`.

## Minimal SBATCH header (reference)

From historical `dcu.slurm` / generated jobs:

```bash
#SBATCH -J <job_name>
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --time 72:00:00
#SBATCH --comment=BASE
#SBATCH -o /public/home/siyangchen/BASE/STDIN_<stamp>/std.out.%j
#SBATCH -e /public/home/siyangchen/BASE/STDIN_<stamp>/std.err.%j
#SBATCH --cpus-per-task=8
```

App section historically ends with:

```bash
python -u /absolute/path/to/script.py
```

## Checklist

```
- [ ] Confirm this is a test requiring DCU, not CPU-only or production work
- [ ] Verify template, partition/account, dcu:1, wall-time, concurrency limits
- [ ] Declare logical test ID, reason, result directory, and max attempts
- [ ] Ensure entry script atomically writes result.json then DONE
- [ ] cd EasyDistillation and verify .portal files
- [ ] generate_slurm_job.sh <abs_script.py> -s
- [ ] Record job ID, Git commit/dirty state, result directory, reason, attempt
- [ ] Write/start `$RESULT_DIR/monitor_job.sh` (or `start_job_monitor.py`)
- [ ] Attach notify_on_output to `^SLURM_MONITOR_DONE` and record monitor_pid
- [ ] On DONE sentinel, read CONTINUE.json and continue the approved next step
- [ ] Launch restricted development only after both completion gates pass
- [ ] Never auto commit/push/deploy/delete/scancel/submit production
- [ ] Stop or escalate when attempts/resources/scope exceed the approved budget
```
