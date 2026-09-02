# EasyDistillation Wilson Current 工作交接

## 交接入口

本文件是当前工作的首要入口。维护者应按以下顺序阅读：

1. `AGENTS.md`：项目、Kunshan 和多 Agent 协作的强制规则；
2. `HANDOVER.md`：当前状态与下一步；
3. `RESTART.md`：安全恢复和验证命令；
4. `INVENTORY.md`：交付边界与文件清单；
5. `docs/permanence-and-deployment.md`：源码、部署、数据和结果永久化门禁。

## 交接文档完整性

canonical transitional manifest 位于 `docs/handover-manifest.json`。在当前 workspace 可运行：

```bash
python tools/build_handover_manifest.py verify \
  docs/handover-manifest.json \
  --root C:/Users/Lenovo/Project/lattice-flow-restart
```

该 manifest 校验完整 delivery allowlist 的 exact bytes，并记录生成时 Git HEAD、dirty 状态和已接受的外部 refs。普通 verify 只证明 bytes；`--require-release-ready` 还要求 Git-clean、remote ref 包含 manifest commit且 retention decision 已解析。当前普通 verify 应通过，但 `release_ready=false`：文档可供当前工作区或连同 manifest 一起复制的 snapshot 恢复，尚不是 fresh-clone 可恢复的永久发布。

## 仓库与运行环境

- 活动本地 Git 副本：`C:/Users/Lenovo/Project/lattice-flow-restart`。
- Git remote：`git@github.com:Si-Yang-Chen/EasyDistillation.git`。
- 当前本地 branch/HEAD/remote refs、dirty 状态与最新 Kunshan 作业见 `TASKBOARD.md` 的时间戳 snapshot；部署前必须重新 fetch/核验，不能把文档中的观察值当永久最新状态。
- Kunshan 是真实数据和 DCU 计算环境；登录节点只能做轻量只读检查，GPU/CuPy/PyQUDA/QUDA 必须通过 Slurm。
- 当前新增/修改尚未 staging、commit 或 push；不得把本地工作区视为已永久化版本。

## 两条工作线

### A. Wilson `J4×J4` Current×Current

这是当前直接推进的主线：构造两个 Wilson 点分裂时间分量守恒流之间的 connected V2V ordered trace，并在真实 Kunshan 数据上完成可审计 smoke 和 ensemble production。

已经实现：

- 八方向 directed-current raw basis：`+x,+y,+z,-x,-y,-z,+t,-t`；
- Wilson forward/backward terms：`-1/2(r-gamma_mu)` 与 `+1/2(r+gamma_mu)`；
- temporal bar/field/link endpoints 和 periodic/open boundary；
- directed-current artifact 的内容寻址 NPY、原子 manifest、输入/data hashes；
- 单 current 与已加载 VSV 的逐 term 跨时 contraction；
- 双 current V2V kernel `contract_directed_current_pair_v2v()`：对 temporal `2×2=4` term pairs 逐项闭合。

双流 kernel 当前严格定义为 **ordered、connected、unflavored、unsigned raw trace**：

- 不隐含 Wick/fermion sign；
- 不隐含 flavor/electric-charge factor；
- 不做体积或 source normalization；
- 不隐式 conjugate、取实部、source average 或 fit。

双流 kernel 与 pair-smoke CLI 已通过聚焦/subprocess CPU 测试并用显式指标循环交叉核验 einsum；真实 cfg10000 双流 smoke 和加入新 kernel 后的完整相关套件见 `TASKBOARD.md` 当前验收状态。

### B. rho charge normalization / Ward--Takahashi

这是独立的物理测量线。它需要 hadron–current–hadron C3、批准的 source/sink projector、flavor weights、C2/C3/ratio、contact 和时间策略。

现有 meson–current 二点或 Current×Current 二点不能充当 H–J–H C3。readiness v4 有意保持 `ready=false`，直到物理负责人批准 measurement contract 并产生真正 C3。

## 已完成的真实 Kunshan 证据

### cfg10000 directed-current / existing-VSV smoke

- 复用 `Ne=1` 的 72 个 source-time VSV 文件；磁盘第二时间轴明确为 `source-relative`；
- Slurm Job `120571967` 生成匹配 directed-current artifact；
- temporal `J4`，`source=0, sink=8, current=4` 的单 current 逐 term contraction 通过；
- Current artifact identity：`bb322b7d704cc3e7b551c12e22ca625de8300e529831fe94980b586f28ae280f`；
- result SHA-256：`f31f3975dbb4493d6eb0bb1c7bbcdf9ceaa7e2f30ab529d7048a3b6404880ac8`。

完整证据见 `docs/kunshan-current-vsv-smoke.md`。这证明 artifact、轴、端点和消费协议，不证明物理归一化或 WT 恒等式。

### 8-cfg EasyDistillation 数据地图

目标配置为 `10000,13000,14000,15000,16000,17000,18000,19000`。已确认存在：

- localized VSV：`576/576` rank slabs；
- localized PSV：`2304/2304` rank slabs；
- localized PSP：`576/576` rank slabs；
- overlap matrices：`8/8`；
- gauge、LapH eigenvectors、sparse points 和 candidate C2 families。

详见 `docs/kunshan-easydistillation-data-map.md`。旧 `03.current_elemental_all` 使用六个空间 GaugeLink 与 legacy `GammaName.A0` 定义，不是新八方向 Wilson temporal `J4`，不能静默复用为 directed-current artifact。历史 8-cfg localized current-current correlators 在 2026-08-22 修复前生成，已清理并被权威记录标为需要重算。

## 多 Agent 集成状态

ownership 见 `docs/ledger/decisions.md`：

- `experiments/nonlocal-current/` 属于本项目；
- `experiments/localized-blending/` 属于另一 owner；
- 路径/API/schema/runtime 均不相交的其他 agent 更改只记录 commit 并原样接受，不逐行审查；
- 触及共享代码时必须兼容审查，不能回退另一方。

本轮 remote/Kunshan 具体 ref 观察已移至 `TASKBOARD.md`。稳定判定是：无交集变化直接接受并记录；共享 `lattice/*` 变化必须兼容合并与测试。部署不得覆盖 Kunshan 共享 checkout；使用新的隔离源码和结果目录，并绑定实际 snapshot identity。

## 当前验收状态

| 层次 | 状态 | 证据/边界 |
| --- | --- | --- |
| Current API 与 directed raw | 已实现 | 本地测试；八方向、term schema、端点和边界契约 |
| 单 current VSV bridge | 已实现并真实 smoke | cfg10000 canonical v3 evidence |
| 双 current V2V kernel | 已实现并真实 smoke | pair kernel focused `6 passed`; pair CLI subprocess tests passed; cfg10000 real pair smoke passed |
| 8-cfg raw input readiness | 输入完整 | VSV/PSV/PSP/overlap 精确计数；operator compatibility 分开判断 |
| Wilson `J4×J4` ensemble | 未生成 | 需 pair CLI、真实 smoke、production contract 和 Slurm jobs |
| H–J–H charge/WT | intentionally not ready | 缺批准物理合同与 C3 |
| 源码永久化 | 未完成 | 大量 scoped 文件仍 untracked/modified；未 commit/push |

最后一次在 pair kernel 之前的完整相关测试为 `166 passed, 1 skipped`；加入 pair kernel 后仅聚焦 `6 passed` 已执行。移交时必须保留这一区分，不能声称新 kernel 已通过 `166` 套件。

## 下一步顺序

1. ~~完成 strict pair-smoke CLI 和 subprocess tests~~（已完成）；
2. 跑完整相关 pytest、Ruff、format 和 diff checks，并完成本地分组 commit；
3. ~~以 cfg10000 `Ne=1` existing VSV + directed-current artifact 做真实双 `J4` smoke~~（已通过，见 `docs/kunshan-current-vsv-pair-smoke.md`）；
4. 审计 full-size `J4×J4` 所需 endpoint/source coverage，禁止仅凭 18-source VSV shape 假设完整；
5. 与 localized-blending 共享代码做兼容合并，建立新的隔离 Kunshan snapshot；
6. 准备 8-cfg production input manifest、资源估算、result contract、Slurm ledger 和 monitor；
7. 用户授权后按 `INVENTORY.md` 精确 staging，review、commit、push；
8. H–J–H 工作线在物理合同获批后另行推进。

## 不能做的事

- 不把 legacy spatial current 当作 Wilson temporal `J4`；
- 不把 current-current 或 meson-current 二点当作 H–J–H C3；
- 不把本地/合成测试冒充真实物理结论；
- 不覆盖或清理其他 agent 的 checkout、dirty 文件和结果；
- 不因本地无数据而停工；
- 未获授权不 staging、commit 或 push。
