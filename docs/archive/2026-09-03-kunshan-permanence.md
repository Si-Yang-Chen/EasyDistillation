# Kunshan 与永久化规则归档

> 归档日期：2026-09-03
>
> 当前开发者只需阅读 `AGENTS.md`、`README.md`、`TASKBOARD.md` 和 `PLAN.md`。本文件保留上线、归档和恢复时需要的详细规则。

## 规则来源

- `AGENTS.md` 是强制执行入口；本文件是被其指向的详细说明。
- `TASKBOARD.md` 记录当前阶段的动态状态，不在本文件复制 HEAD、Job ID 或测试数字。
- `docs/ledger/decisions.md` 记录长期 ownership 和协作决策。
- `docs/ledger/facts.md` 只保留稳定技术事实。

## 多 Agent 集成

上线 Kunshan 前必须重新执行 `git fetch --prune`，并核对：

- 本地 HEAD、目标 remote ref；
- Kunshan shared checkout 的 HEAD、branch、dirty/untracked；
- `~/qedinf/experiments/*` 的 owner 和实际依赖；
- 当前用户作业与相关 Slurm 作业。

变化按路径 ownership、实际依赖、公共 API/import-export、schema/数据格式、运行环境和消费公式分类：

- 所有方面均无交集：记录来源 ref/commit 后原样接受，不逐行审查，也不为其补跑专项测试；这不构成本项目对其正确性的背书。
- 任一方面有交集：保留双方语义，做兼容审查、聚焦测试和必要的完整测试。

禁止对共享 checkout 执行 `reset --hard`、`git clean`、强制 checkout、`rsync --delete` 或无差别覆盖。项目部署必须进入新的隔离源码/结果目录。

当前重点共享路径包括：`lattice/__init__.py`、`lattice/generator/*`、`lattice/insertion/*`、`lattice/quark_diagram.py`、`lattice/preset.py`、propagator/slab readers、Current schemas 和 Kunshan launcher。

## Kunshan 数据

Kunshan 是大型数据和 DCU 计算环境。源码仓库只保存 manifest、路径、shape、dtype、轴语义、hash、作业身份和小型结果摘要，不保存 gauge/eigenvector/propagator/elemental/大型 correlator 本体。

每个数据 family 的 manifest 必须记录：

- ensemble、configuration、Ne/Np、boundary 和 source set；
- 绝对路径或严格 pattern；
- 文件覆盖数、shape、dtype、完整轴顺序；
- source-relative/source-sink/absolute 时间语义；
- 文件或 family manifest SHA-256；
- producer code、Job ID 和 compatibility verdict。

文件存在不代表兼容。旧 schema、旧轴、旧公式或 superseded 结果不得静默复用。

## 作业与结果

- 登录节点只做 CPU 只读检查、manifest/hash 检查和轻量 NumPy 工作；GPU/CuPy/PyQUDA/QUDA 必须通过批准的 Slurm 计算节点流程。
- 每个 attempt 使用新的结果目录，记录实际部署 snapshot、source/environment/input manifest、资源参数、Job ID、stdout/stderr 和 monitor。
- 计算在 stage 目录完成；结果和 manifest flush/fsync 后原子发布；最后才创建 `DONE`。
- monitor 必须确认 Slurm terminal state、全部 result artifacts、result manifest 和 `DONE`，之后才宣告成功。
- 结果分类必须区分 synthetic CPU、real-artifact smoke、real-gauge measurement、statistical/physics conclusion。

## Source snapshot

部署 snapshot 必须绑定：

- Git commit；
- tracked/untracked dirty 状态；
- 已接受的外部 refs；
- source manifest 及其 SHA-256；
- launcher、Python/NumPy/CuPy/PyQUDA/MPI 环境；
- 实际运行脚本 hash。

如果使用 dirty snapshot，必须保存确定性 patch/source manifest，直到源码永久形成 commit。pair smoke 还必须让 execution record 的 Git 状态和 source manifest 完全一致。

## 交付与 release gate

`docs/delivery-files.list` 是逐文件交付 allowlist，禁止目录项。使用 `tools/build_handover_manifest.py check-files` 检查后，再使用：

```bash
git add --pathspec-from-file=docs/delivery-files.list
git diff --cached --name-status
git diff --cached --check
```

`docs/handover-manifest.json` 是 exact-byte manifest。普通 `verify` 证明清单内文件未改变；`verify --require-release-ready` 还要求：

- 清单文件和 manifest 已 Git-clean；
- source/doc commit 与 manifest commit 已 push 到 remote-tracking ref；
- fresh clone 可复原并验证；
- retention decision 已解决。

大型数据 retention 必须二选一：

- `approved`：记录 secondary copy/snapshot、保留期、恢复负责人、删除策略，以及带检查人/时间/路径/hash 的 passed `restore_check`；
- `not-required`：由用户明确接受 single-copy 风险，同样记录 authority/time、保留期、恢复负责人和删除策略。

pending retention 不得通过 release-ready。任何 retention 状态都不授权自动删除其他 agent 或既有生产数据。

## 科学边界

- Wilson `J4×J4` raw pair trace 与 flavor、Wick sign、归一化和取实部定义分离。
- current-current 二点不能代替 rho charge normalization 所需的 H–J–H C3。
- artifact smoke 不能代替 ensemble、Ward--Takahashi 或 charge-normalization 结论。
- 物理测量继续由 `docs/kunshan-current-measurement-readiness.md` 和 `docs/conserved-current-measurement-decisions.md` 管理。
