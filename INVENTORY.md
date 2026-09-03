# Wilson Current 交付清单说明

本文件只说明交付边界；逐文件事实以 [`docs/delivery-files.list`](docs/delivery-files.list) 为准，用户阅读子集以 [`docs/handover-files.list`](docs/handover-files.list) 为准。

## 交付范围

交付内容分为：

- Wilson Current API、八方向 directed raw 和单/双 current contraction；
- directed-current artifact、existing-VSV/pair-smoke 和 conserved-current validation 工具；
- 对应 CPU/subprocess/monitor 测试；
- `README.md`、`AGENTS.md`、`HANDOVER.md`、`RESTART.md`、`PLAN.md`、`TASKBOARD.md` 和本文件；
- Current API、Kunshan 数据/结果证据、多 Agent/永久化归档和 retention decision；
- `.cursor/skills/dcu-slurm-submit/` 中 AGENTS 强制依赖的通用 DCU skill、portal 模板和 monitor，但不包含 legacy localized 专用 monitor。

## 不属于本项目交付

- `experiments/localized-blending/` 及其 owner 管理内容；
- 其他 agent 的 stochastic、localized 或私有实验代码；
- 用户原有未跟踪脚本：`chic1_single.py`、`contraction*.py`、`single_hadron_charm.py`、`stochastic_*.py`、`test_correlator.py`、`test_final_verification.py`、`test_metadata_correct.py`；
- `NUL`、session、IDE、缓存、临时日志和测试输出；
- `.cursor/skills/dcu-slurm-submit/scripts/monitor_legacy_localized_chain.py`；
- 没有逐项列入 `docs/delivery-files.list` 的文件。

这些文件默认保留在工作区，不因本清单而删除、覆盖或清理。

## 永不进入普通 Git

大型 gauge、LapH eigenvector、VSV/PSV/VSP/PSP、elemental、NPY/NPZ/HDF5 correlator、完整 Slurm 日志和运行缓存不进入普通 Git。Git 保存小型 manifest、绝对路径、configuration、shape、dtype、轴语义、SHA-256、producer/Job ID 和结果摘要。

大型数据是否有 secondary copy 由 [`docs/data-retention-decision.json`](docs/data-retention-decision.json) 决定。`pending` 时不得宣告永久 release；任何删除操作仍需 owner/用户明确授权。

## 当前关键外部对象

Kunshan ensemble root：

```text
/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72
```

真实输入路径、8 个配置、manifest hash、shape、dtype、轴和兼容性结论见 [`docs/kunshan-easydistillation-data-map.md`](docs/kunshan-easydistillation-data-map.md)。

单流与双流 smoke 证据：

- [`docs/kunshan-current-vsv-smoke.md`](docs/kunshan-current-vsv-smoke.md)；
- [`docs/kunshan-current-vsv-pair-smoke.md`](docs/kunshan-current-vsv-pair-smoke.md)。

## staging 规则

只在用户授权且准备发布时执行：

```bash
python tools/build_handover_manifest.py check-files --root . --files-from docs/delivery-files.list
git add --pathspec-from-file=docs/delivery-files.list
git diff --cached --name-status
git diff --cached --check
```

`check-files` 拒绝目录、重复、绝对路径、越界路径、不存在文件和被 Git 忽略的文件。禁止 `git add .`、目录 pathspec、强制 push 和覆盖其他 agent。

完整的 Kunshan preflight、隔离部署、source manifest、execution record、原子结果、monitor、fresh clone 和 release gate 见 [`docs/archive/2026-09-03-kunshan-permanence.md`](docs/archive/2026-09-03-kunshan-permanence.md)。

## 当前状态

- 本地专用分支和分组 commits 已形成，push 已获持续授权；
- Kunshan shared checkout 与其他 agent 状态必须每次上线前重新核对；
- cfg10000 `Ne=1` 双 `J4×J4` artifact smoke 已完成；8-cfg `J4×J4` ensemble 尚未生成；
- H–J–H charge/WT 仍等待物理合同；
- delivery manifest 是 exact-byte 检查，不替代 Git remote 发布或大型数据备份。
