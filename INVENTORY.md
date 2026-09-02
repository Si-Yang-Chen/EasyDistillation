# Wilson Current 交付内容清单

## 用途

本清单用于三件事：

1. 确定用户可接收的源码和文档边界；
2. 为未来使用 `docs/delivery-files.list` 精确 staging 提供审核起点；
3. 防止把其他 agent、用户原有文件、缓存、大型数据或临时运行产物误纳入本项目提交。

当前清单是**候选交付清单**，不是 staged/committed 证明。最终状态必须由 `git diff --cached --name-status` 和 remote commit 验证。

## A. 本项目核心交付候选

### A1. Current API 与生成器

| 路径 | 状态/用途 |
| --- | --- |
| `lattice/insertion/current.py` | CurrentTerm、Current schemas、ConservedVectorCurrent、端点、spin 和验证 |
| `lattice/insertion/gauge_link.py` | directed eight-one-link basis，与 legacy spatial GaugeLink 并存 |
| `lattice/generator/elemental.py` | 四方向 gauge 保留与 `calc_directed_current_raw()` |
| `lattice/current_elemental.py` | artifact persistence、单 Current VSV bridge、双 Current V2V kernel |
| `lattice/quark_diagram.py` | equal-time adapter 和共享 consumption seam；与 localized-blending 有交叉，提交前必须兼容审查 |
| `lattice/insertion/__init__.py` | Current API exports；共享导出面 |
| `lattice/__init__.py` | package exports；共享导出面 |
| `lattice/correlator/conserved_charge.py` | formula-neutral scalar projection/declared ratio helpers |
| `lattice/correlator/__init__.py` | correlator helper exports |

### A2. 本地测试

| 路径 | 覆盖范围 |
| --- | --- |
| `tests/test_current.py` | terms、schema、边界、验证 |
| `tests/test_current_consumption.py` | spin-aware consumption |
| `tests/test_temporal_current_elemental.py` | temporal directed raw generation |
| `tests/test_current_v2v_persistence.py` | content-addressed artifact 和 source hashes |
| `tests/test_current_v2v_contraction.py` | 单流和双流 V2V term/pair contraction |
| `tests/test_conserved_charge_v2v.py` | formula-neutral projection/ratio |
| `tests/test_slurm_job_monitor.py` | 持久 monitor completion/sentinel 回归；因 AGENTS 强制依赖站点流程而纳入 |
| `test/current_conservation_cpu_precheck.py` | synthetic deterministic CPU precheck |
| `tests/test_handover_manifest.py` | handover manifest build/verify、安全路径和篡改回归 |

`tests/test_quark_diagram_contraction.py` 是已有/共享测试；只有实际修改时才纳入本项目 staging。`tests/test_slurm_job_monitor.py` 已因本项目强制依赖通用 monitor 而列入 `docs/delivery-files.list`；legacy localized monitor 仍排除。

### A3. 实验工具

| 路径 | 用途/边界 |
| --- | --- |
| `experiments/directed-current-v2v/build_vsv_timeslice_manifest.py` | VSV family hash manifest |
| `experiments/directed-current-v2v/generate_current_artifact_dcu.py` | Kunshan DCU directed-current artifact generator |
| `experiments/directed-current-v2v/contract_existing_vsv.py` | strict 单 Current existing-VSV smoke |
| `experiments/directed-current-v2v/contract_existing_vsv_pair.py` | 双流 pair-smoke CLI；绑定 source manifest 与 execution record |
| `experiments/directed-current-v2v/test_contract_existing_vsv_cpu.py` | CLI subprocess regression |
| `experiments/directed-current-v2v/README.md` | artifact workflow说明 |
| `experiments/conserved-current-validation/README.md` | validation package 边界说明 |
| `experiments/conserved-current-validation/analyze_conservation.py` | formula-neutral ensemble analysis primitives |
| `experiments/conserved-current-validation/audit_measurement_readiness.py` | contract/inventory readiness audit |
| `experiments/conserved-current-validation/build_measurement_inventory.py` | declarative inventory builder |
| `experiments/conserved-current-validation/manifest.template.json` | audited observable producer template |
| `experiments/conserved-current-validation/measurement-contract.template.json` | physics contract template |
| `experiments/conserved-current-validation/measurement-inventory-source.template.json` | inventory source template |
| `experiments/conserved-current-validation/produce_real_observables.py` | audited contractions packager |
| `experiments/conserved-current-validation/run_real_gauge_validation.py` | gated real-observable analyzer |
| `experiments/conserved-current-validation/submit_validation.slurm.template` | site-neutral Slurm template |
| `experiments/conserved-current-validation/test_experiment_cpu.py` | analyzer CPU tests |
| `experiments/conserved-current-validation/test_measurement_inventory_cpu.py` | inventory CPU tests |
| `experiments/conserved-current-validation/test_measurement_readiness_cpu.py` | readiness CPU tests |
| `experiments/conserved-current-validation/test_producer_cpu.py` | producer CPU tests |

pair-smoke CLI 已实现并通过合成 subprocess tests；真实 Kunshan smoke 完成后在此记录 schema/evidence。

### A4. 用户文档、永久化与上线工具

| 路径 | 作用 |
| --- | --- |
| `tools/build_handover_manifest.py` | 对显式交付 allowlist 生成/验证 SHA-256 manifest；拒绝目录、重复和越界项 |
| `docs/delivery-files.list` | canonical 逐文件 staging/source-manifest allowlist；不得包含目录 |
| `docs/handover-files.list` | 用户交接文档子集；用于阅读审计，不替代完整 delivery list |
| `docs/data-retention-decision.json` | 大型数据第二副本、single-copy 风险和删除策略的 hard-gate 决策 |
| `.cursor/skills/dcu-slurm-submit/SKILL.md` | AGENTS 强制引用的 Kunshan DCU 流程 |
| `.cursor/skills/dcu-slurm-submit/monitor.md` | monitor 结果协议 |
| `.cursor/skills/dcu-slurm-submit/portal-template.md` | portal 配置说明 |
| `.cursor/skills/dcu-slurm-submit/job_portal.var.template` | portal 变量模板 |
| `.cursor/skills/dcu-slurm-submit/scripts/start_job_monitor.py` | 通用持久 Slurm monitor；不含 legacy localized 专用 monitor |
| `AGENTS.md` | 项目强制执行规则、多 Agent 与 Kunshan 规则 |
| `HANDOVER.md` | 当前状态单一入口 |
| `RESTART.md` | 可执行恢复 runbook |
| `INVENTORY.md` | 本交付边界 |
| `TASKBOARD.md` | 动态任务状态 |
| `docs/permanence-and-deployment.md` | 永久化、集成、上线、结果和恢复门禁 |
| `docs/current-api.md` | Current API 和公式契约 |
| `docs/temporal-gauge-link-elementals.md` | directed raw basis 与 temporal link实现 |
| `docs/kunshan-current-vsv-smoke.md` | cfg10000 真实单流 smoke evidence |
| `docs/kunshan-easydistillation-data-map.md` | Kunshan 数据路径、shape、axis 和兼容性 |
| `docs/kunshan-current-measurement-readiness.md` | charge/H–J–H readiness v4 |
| `docs/conserved-current-measurement-decisions.md` | 物理授权 checklist |
| `docs/ledger/decisions.md` | ownership 与策略决策 |
| `docs/ledger/facts.md` | 稳定事实台账 |
| `DOCUMENTATION_INDEX.md` | 项目级文档索引；本轮会补充 Current 入口，但不重写其他模块索引 |

canonical manifest 有意不列入 `docs/delivery-files.list`，因为它必须在完整 source/doc commit 已 push 后生成，并作为第二个 commit 单独发布；否则会形成自引用且无法证明 remote recoverability。

## B. 共享/其他 Agent 文件

下列类型不能因为出现在本地或 Kunshan checkout 中就自动归入本项目提交：

- `experiments/localized-blending/`：由 orchestrator-localized-blending 管理；
- `lattice/localized_sampling.py`、`lattice/temporal_slab_io.py`、`lattice/point_propagator_io.py`、`lattice/propagator_estimates.py`：来自 localized-blending 分支，若作为依赖接入，应保留来源 commit并做共享兼容测试；
- `lattice/preset.py`、`lattice/generator/perambulator.py`：可能由 localized-blending 修改；本项目未修改时不应制造无关 diff；
- stochastic 模块和 `origin/feature-stochastic`：无交集时原样接受，不逐行审查，也不归为本项目实现；
- `.cursor/skills/dcu-slurm-submit/scripts/monitor_legacy_localized_chain.py`：localized 历史专用 monitor，不纳入本项目交付；
- 未列入 `docs/delivery-files.list` 的其他 `.cursor` 工具：不自动归入本项目。

对无关变化的“接受”是基线集成决定，不代表把它列入本项目成果或对其代码质量背书。

## C. 本地用户/历史文件：默认排除

当前工作区存在许多未跟踪的历史或用户文件，除非用户逐项指定，否则不得 staging：

- `chic1_single.py`；
- `contraction.py`、`contraction_cpu.py`；
- `single_hadron_charm.py`；
- `stochastic_*.py`；
- `test_correlator.py`、`test_final_verification.py`、`test_metadata_correct.py`；
- `NUL`；
- 未经 ownership 审核的 `final-results/` 内容；
- IDE、assistant session、临时输出和本地环境文件。

这些文件的存在不是删除授权。不得清理、覆盖或重命名它们。

## D. 永不进入普通 Git 提交

- gauge `.lime`；
- LapH eigenvector arrays；
- VSV/PSV/VSP/PSP propagators；
- current/meson elementals；
- production correlator arrays；
- 大型 HDF5/NPY/NPZ；
- Slurm stdout/stderr 全量日志；
- Python bytecode、`.pytest_cache`、`.ruff_cache`、build/dist；
- 临时 stage、partial、lock、monitor process state。

大型数据保留在 Kunshan；Git 只保存小型 manifest、hash、路径、axis 和结果摘要。若用户批准 Git LFS 或外部 artifact store，应另立 retention/restore 决策，不默认改变本规则。

## E. Kunshan 权威外部对象

### E1. 真实输入根

- 数据根：`/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72`；
- gauge root：`/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original`；
- localized VSV/PSV/PSP、overlap、candidate C2 绝对路径见 `docs/kunshan-easydistillation-data-map.md`。

这些文件不随本地源码交付；其路径和 hashes 是交付证据的一部分。

### E2. 当前 canonical smoke 证据

- 工作根：`/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612`；
- directed-current generation accepted attempt：Job `120571967`；
- single-current VSV smoke canonical result：`results/vsv-current-smoke-cfg10000-v3`；
- measurement readiness canonical report：`audit/measurement-readiness-v4.json`。

旧 attempts/v1/v2 保留为 lineage，不覆盖、不删除，也不作为 canonical completion。

### E3. 历史不兼容/已删除结果

`05.correlator.localized.production.ne128.np64.src18` 的旧 8-cfg localized correlators 在 2026-08-22 implementation fix 前生成并已清理。历史 result manifests 可作 provenance，不可作当前可消费 arrays。

## F. 提交前精确核验

获得用户 staging 授权后：

```bash
python tools/build_handover_manifest.py check-files \
  --root C:/Users/Lenovo/Project/lattice-flow-restart \
  --files-from C:/Users/Lenovo/Project/lattice-flow-restart/docs/delivery-files.list

git add --pathspec-from-file=docs/delivery-files.list
git diff --cached --name-status
git diff --cached --check
git diff --cached
```

必须确认：

- staged 文件全部属于 A 类或经用户单独批准；
- C/D 类为零；
- 共享文件保留其他 agent 语义；
- 没有大型数据；
- 新文件不是只存在于 working tree 而遗漏于 staged tree；
- 测试是在 staged tree 对应内容上通过。

## G. 当前永久化状态

| 项目 | 状态 |
| --- | --- |
| 本地 Git 副本 | 有，活动目录已确认 |
| remote 绑定 | 有，`origin` 指向 EasyDistillation GitHub |
| 核心源码 staging | 未授权、未执行 |
| commit/push | 未执行 |
| Kunshan 单流 evidence | 已完成并记录 |
| 双流 library kernel | 本地聚焦通过，未完整交付 |
| pair CLI / real smoke | 未完成 |
| 8-cfg Wilson `J4×J4` result | 未生成 |
| 大型数据第二副本 | hard gate 为 pending：未确认备份；在 decision 改为 `approved` 或用户签署 `not-required` 前 release-ready 必须失败 |
| 用户文档整理 | workspace 入口/runbook/inventory/permanence/taskboard 完成；manifest+monitor focused `19 passed`；canonical transitional manifest 提供 exact-byte 验证 |
