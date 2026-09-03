# Wilson Current 重启指南

## 先读

1. `AGENTS.md`：强制开发、Kunshan 和多 Agent 规则；
2. `TASKBOARD.md`：当前阶段、动态状态和 blockers；
3. `PLAN.md`：长期计划；
4. `INVENTORY.md`：交付边界；
5. `docs/archive/2026-09-03-kunshan-permanence.md`：完整永久化、上线和恢复规则；
6. `docs/current-api.md`：修改 Current/API/schema/轴时阅读。

## 最小本地开发流程

```bash
cd C:/Users/Lenovo/Project/lattice-flow-restart
python -m pytest -p no:cacheprovider -q \
  tests/test_current.py \
  tests/test_current_consumption.py \
  tests/test_temporal_current_elemental.py \
  tests/test_current_v2v_persistence.py \
  tests/test_current_v2v_contraction.py
python -m ruff format --check \
  lattice/current_elemental.py \
  lattice/generator/elemental.py \
  lattice/generator/sparsened_point.py \
  lattice/insertion/current.py \
  lattice/insertion/gauge_link.py \
  lattice/quark_diagram.py \
  tests/test_current_v2v_contraction.py \
  experiments/directed-current-v2v/contract_existing_vsv_pair.py
python -m ruff check --no-cache \
  lattice/current_elemental.py \
  lattice/generator/elemental.py \
  lattice/generator/sparsened_point.py \
  lattice/insertion/current.py \
  lattice/insertion/gauge_link.py \
  lattice/quark_diagram.py \
  tests/test_current_v2v_contraction.py \
  experiments/directed-current-v2v/contract_existing_vsv_pair.py
git diff --check
```

合成自由场 precheck：

```bash
PYTHONDONTWRITEBYTECODE=1 python test/current_conservation_cpu_precheck.py
```

日常开发只维护源码、对应测试和 `TASKBOARD.md`。不要为了普通 CPU 开发重建 handover manifest、检查 Kunshan 数据或运行 Slurm。

## 当前实现入口

- terms、endpoint、spin 和 Wilson coefficients：`lattice/insertion/current.py`；
- 八方向 raw：`lattice/generator/elemental.py::calc_directed_current_raw`；
- artifact 和单/双 current V2V contraction：`lattice/current_elemental.py`；
- pair smoke：`experiments/directed-current-v2v/contract_existing_vsv_pair.py`；
- readiness/物理合同：`experiments/conserved-current-validation/`。

双流 kernel 是 ordered、connected、unflavored、unsigned raw trace；不自动加入 Wick sign、flavor weight、normalization、取实部或 source averaging。

## 需要 Kunshan 时

遵守 `AGENTS.md`，并在上传或运行前重新核对：

```bash
git fetch --prune origin
git status --short --branch
git rev-parse HEAD
git rev-parse origin/master
ssh -o BatchMode=yes kunshan 'cd "$HOME/qedinf/EasyDistillation" && git status --short --branch && git rev-parse HEAD && squeue -h -u "$USER" -o "%i|%T|%j|%R"'
```

若其他 agent 变化与本项目路径、API、schema、数据格式、runtime 和消费公式均无交集，记录 ref 后原样接受，不做逐行审查；若触及共享依赖，必须兼容审查和测试。任何部署都进入新的隔离源码/结果目录，不能覆盖 shared checkout。

真实数据路径、shape、dtype、轴和 hash 见 `docs/kunshan-easydistillation-data-map.md`；单流/双流真实 smoke 证据见对应 Kunshan 文档。

## 永久化和交接

准备用户交付时再执行：

```bash
python tools/build_handover_manifest.py check-files --root . --files-from docs/delivery-files.list
python tools/build_handover_manifest.py verify docs/handover-manifest.json --root .
```

commit、push、fresh clone、source manifest、retention、Slurm monitor、失败恢复和禁止操作全部归档于 `docs/archive/2026-09-03-kunshan-permanence.md`。不要在本文件复制完整永久化流程。

## 当前真实证据

- cfg10000 单流 artifact smoke：`docs/kunshan-current-vsv-smoke.md`；
- cfg10000 双流 `J4×J4` smoke：`docs/kunshan-current-vsv-pair-smoke.md`；
- 8-cfg 输入地图：`docs/kunshan-easydistillation-data-map.md`；
- rho charge/WT readiness：`docs/kunshan-current-measurement-readiness.md`。

这些证据属于不同层级；artifact smoke 不等于 ensemble 或物理结论。
