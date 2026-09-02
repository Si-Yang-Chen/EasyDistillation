# Minimal `.portal` stubs for generated DCU jobs

Place these under `$OUT/.portal/` where `$OUT` is the job output directory
(e.g. `$HOME/BASE/STDIN_MMDD_HHMMSS`). Replace every `REPLACE_OUT` and
`REPLACE_NAME` before submit.

## `job_portal.var`

```bash
GAP_NNODE=1
GAP_JOB_NAME=REPLACE_NAME
GAP_NODE_TYPE=nodeNum
GAP_STD_ERR_FILE=REPLACE_OUT/std.err.%j
GAP_SUBMIT_TYPE=cmd
GAP_CMD_FILE=/tmp/STD_CMD_dummy.sh
GAP_SCHEDULER_OPT_WEB=/tmp/SCHEDULER_dummy.var
GAP_GUI=0
GAP_NGPU=
GAP_CLUSTER_ID=20035
advance=true
GAP_WORK_DIR=REPLACE_OUT
GAP_MULTI_SUB=1
GAP_STD_OUT_FILE=REPLACE_OUT/std.out.%j
GAP_JOB_MEM=
GAP_NDCU=1
GAP_QUEUE=kshdnormal
GAP_SCHED_TYPE=SLURM
GAP_PPN=1
GAP_WALL_TIME=72:00:00
```

## `job_interface.var`

Prefer copying a known-good file from a previous job:

```bash
cp /public/home/siyangchen/BASE/STDIN_1222_204826/.portal/job_interface.var \
   REPLACE_OUT/.portal/job_interface.var
```

That file maps `GAP_*` into `WORK_DIR`, `QUEUE`, `NDCU`, etc. used by the
Sugon BASE Slurm template.

## One-liner bootstrap

```bash
OUT=/public/home/siyangchen/BASE/STDIN_$(date +%m%d_%H%M%S)
NAME=my_dcu_job
mkdir -p "$OUT/.portal"
cp /public/home/siyangchen/BASE/STDIN_1222_204826/.portal/job_interface.var "$OUT/.portal/"
sed -e "s|REPLACE_OUT|$OUT|g" -e "s|REPLACE_NAME|$NAME|g" \
  /public/home/siyangchen/qedinf/EasyDistillation/.cursor/skills/dcu-slurm-submit/job_portal.var.template \
  > "$OUT/.portal/job_portal.var"
```

If `job_portal.var.template` is missing, paste the `job_portal.var` block above
manually.
