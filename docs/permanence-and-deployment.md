# Wilson Current 工作流永久化与 Kunshan 上线规范

## 目的

本规范定义如何把当前 Wilson 点分裂守恒流工作从本地 dirty 工作区和 Kunshan 临时执行目录，转化为可长期复现、可由其他维护者接管的源码、数据索引、执行快照和结果证据。

“永久化”不等于把大型格点数据复制进 Git。它要求每一层都有稳定身份、至少一个权威位置、可验证的内容哈希，以及从结果回溯到源码和输入的完整链路。

## 永久化对象

| 层次 | 权威对象 | 最低永久化要求 | 当前状态 |
| --- | --- | --- | --- |
| 源码 | Git commit 和 remote ref | 干净 commit、review、测试、push；共享模块兼容记录 | 尚未 commit/push |
| 部署源码 | Kunshan 隔离 checkout/snapshot | 绑定 Git SHA；dirty 时另存 patch/source manifest SHA-256 | cfg10000 smoke 有历史隔离快照；新 pair 工作尚未部署 |
| 大型输入 | Kunshan gauge/eigenvector/VSV/PSV/PSP 等 | 绝对路径、cfg、shape、dtype、轴、逐文件或 manifest SHA-256、兼容性结论 | 目标 8 cfg 的 localized inputs 已定位并做完整性检查 |
| 作业 | Slurm attempt | 资源、命令、环境、Job ID、源码身份、输入 manifest、持久 monitor | cfg10000 directed-current generation 已有完整记录；8-cfg pair 尚未提交 |
| 结果 | 新的不可覆盖结果目录 | 原子 artifact/result、manifest、最后写 `DONE`、结果哈希、attempt lineage | 单流 cfg10000 smoke 已完成；双流结果尚未生成 |
| 用户文档 | 本仓库交接文档集 | 当前状态、重启命令、inventory、证据索引、未决事项与验收记录 | 动态状态见 `TASKBOARD.md` |

## 1. 源码永久化

### 1.1 选定共同基线

上线或提交前必须重新执行：

```bash
git fetch --prune origin
git status --short --branch
git rev-parse HEAD
git rev-parse origin/master
git log --oneline --decorate --graph -20 --all
```

同时只读核验 Kunshan：

```bash
ssh -o BatchMode=yes kunshan '
  cd "$HOME/qedinf/EasyDistillation" &&
  git status --short --branch &&
  git rev-parse HEAD &&
  git for-each-ref --format="%(refname:short) %(objectname:short)" refs/heads refs/remotes
'
```

该命令用于人工 preflight；任何嵌入生产脚本的版本仍必须先在隔离环境验证 shell quoting 和输出格式。

### 1.2 多 Agent 变化分类

外部变化按以下顺序分类：

1. 比较变更路径；
2. 比较 import/export 和公共 API；
3. 比较 schema、文件格式和轴约定；
4. 比较运行环境、launcher 和数据消费公式。

如果四项均与本项目无交集：

- 记录 ref/commit；
- 原样接受进入共同基线；
- 不逐行审查、不为其补跑专项测试；
- 不对该模块的正确性作本项目背书；
- 不得为了缩小本项目 diff 而回退该变化。

如果任一项有交集，则属于共享依赖，必须做兼容审查。当前共享重点包括：

- `lattice/__init__.py`；
- `lattice/generator/elemental.py`；
- `lattice/generator/perambulator.py`；
- `lattice/insertion/`；
- `lattice/preset.py`；
- `lattice/quark_diagram.py`；
- temporal-slab/propagator readers；
- Current schema、axis 和 contraction formula；
- Kunshan launcher、MPI ABI 和结果协议。

### 1.3 精确 staging

未经用户明确授权，不执行 `git add`、commit 或 push。获得授权后也不得使用 `git add .` 或目录 pathspec。canonical 逐文件清单位于 `docs/delivery-files.list`；先验证，再 staging：

```bash
ROOT=C:/Users/Lenovo/Project/lattice-flow-restart
python tools/build_handover_manifest.py check-files \
  --root "$ROOT" \
  --files-from "$ROOT/docs/delivery-files.list"
git add --pathspec-from-file=docs/delivery-files.list
git diff --cached --name-status
git diff --cached --check
git diff --cached
```

`check-files` 拒绝目录、重复、绝对、越界和不存在项。提交前仍必须人工核验 staged 清单，排除用户原有脚本、缓存、临时日志、`NUL`、大型数据和未经审查的其他模块文件。若新增交付文件，先逐行修改 `docs/delivery-files.list`，重新执行检查；不得临时扩大为目录。

### 1.4 建议提交序列

建议拆为可独立审查的提交：

1. Current term/schema、directed basis 和 generator；
2. artifact persistence、单流/双流 VSV contraction 和测试；
3. existing-VSV/pair-smoke CLI 与 Kunshan 执行协议；
4. readiness/measurement-neutral utilities；
5. Kunshan evidence、数据地图和交接文档。

每个提交都应保留其他 agent 已接受的共同基线，不通过重写历史隐藏其变化。

## 2. 部署源码永久化

### 2.1 禁止共享 checkout 覆盖

不得向 `~/qedinf/EasyDistillation` 直接无差别复制本地树，也不得在其上执行：

- `git reset --hard`；
- `git clean`；
- 强制 checkout；
- 会删除未知文件的 `rsync --delete`；
- 覆盖其他 owner 的 `experiments/*`。

新工作应部署到新的隔离目录。目录名必须包含逻辑测试 ID 或 UTC 时间戳；实际绝对路径在创建后写入 execution record，不能提前猜测。

### 2.2 部署快照身份

每个部署快照至少包含：

- `git_commit`；
- `git_dirty`；
- 接受的外部 refs/commits；
- tracked diff SHA-256；
- untracked allowlist 和每个文件 SHA-256；
- 综合 source-manifest identity；
- Python、NumPy、CuPy、PyQUDA/QUDA 和 MPI 环境摘要；
- launcher/entry script SHA-256。

在永久 Git commit 形成前，dirty patch/source manifest 是临时身份；形成 commit 后，应以 commit 为主身份，并记录它与历史 snapshot 的对应关系。

## 3. 数据永久化

大型输入继续保存在 Kunshan。Git 中提交小型机器可读 manifest 和说明文档，而不是数据本体。

每个数据 family 的 manifest 至少记录：

- ensemble 和配置号；
- 绝对路径或严格 pattern；
- 文件数与 coverage；
- shape、dtype 和完整轴顺序；
- source/sink 时间是 absolute、source-relative 或其他明确约定；
- boundary、Ne/Np、source set；
- 文件 SHA-256 或受哈希保护的 family manifest；
- gauge/eigenvector/points 来源；
- producer commit/script/Job ID；
- 与当前消费公式的 compatibility verdict。

只有文件存在不足以判定可复用。旧 `current_elemental_all`、旧 current-current 结果等必须保留“不兼容/已 supersede/已删除”的事实记录，避免未来维护者静默误用。

当前 decision 文件为 `docs/data-retention-decision.json`。在其状态为 `pending` 时，永久交付和 production release 都必须失败。用户必须选择并签署以下一种状态：

- `approved`：记录已验证的 secondary copy/snapshot、retention period、restore owner、删除策略，并提供 `restore_check`：`status=passed`、检查人/时间、仓库内 evidence path 和 evidence SHA-256；evidence 文件必须列入 `docs/delivery-files.list`；
- `not-required`：由用户明确接受 single-copy 风险，记录 authority/time、retention period、restore owner（即使只有 Kunshan）和删除策略。

Git 中的 manifest 提供完整性和定位能力，但不替代大型数据备份。任何状态都不得授权本项目自动删除已有输入或结果。

## 4. 作业与结果永久化

### 4.1 每次 attempt 必须独立

每次 Slurm submission 使用新目录，不覆盖失败或旧结果。目录至少包含：

- `.portal/job_portal.var` 与 `job_interface.var`；
- 提交脚本；
- `job-state.json`；
- stdout/stderr；
- producer 输出；
- `result.json`；
- `DONE`；
- `CONTINUE.json` 与 monitor log。

### 4.2 发布顺序

1. 在 stage 目录计算；
2. flush/fsync 数据；
3. 写并校验机器可读 result/manifest；
4. 原子 rename 到最终目录；
5. 最后创建 `DONE`；
6. monitor 验证 Slurm terminal state、result 和 `DONE` 后才宣告 attempt 完成。

结果 manifest 必须明确分类：

- synthetic CPU evidence；
- Kunshan artifact smoke；
- real-gauge correlator measurement；
- statistical/physics conclusion。

较低层通过不能冒充较高层结论。

## 5. 用户可移交文档集

交付入口和职责如下：

| 文档 | 读者问题 |
| --- | --- |
| `HANDOVER.md` | 项目现在在哪里，完成了什么，下一步是什么？ |
| `RESTART.md` | 新维护者如何安全恢复工作并运行验证？ |
| `INVENTORY.md` | 哪些文件属于交付，哪些数据在 Kunshan，哪些必须排除？ |
| `TASKBOARD.md` | 当前任务状态、依赖和验收证据是什么？ |
| `docs/permanence-and-deployment.md` | 如何 commit、集成其他 agent、部署、归档和恢复？ |
| `docs/current-api.md` | Current API、term、轴和公式契约是什么？ |
| `docs/kunshan-current-vsv-smoke.md` | cfg10000 真实 artifact smoke 的证据是什么？ |
| `docs/kunshan-easydistillation-data-map.md` | 真实数据在哪里、是否兼容？ |
| `docs/kunshan-current-measurement-readiness.md` | H–J–H/charge 测量为什么仍未 ready？ |
| `docs/conserved-current-measurement-decisions.md` | 物理负责人必须批准哪些定义？ |
| `tools/build_handover_manifest.py` | 如何验证这组交接文档的 exact bytes？ |
| `docs/handover-manifest.json` | 哪些交接文件及 SHA-256 构成当前 canonical 文档包？ |

文档不得包含会快速失效而无证据来源的声明。动态状态写入 `TASKBOARD.md`；稳定协议写入 API/permanence 文档；真实路径和 hash 写入 Kunshan evidence 文档。

### 5.1 交接 manifest 生命周期

canonical manifest 覆盖 `docs/delivery-files.list` 中的完整交付源码/测试/文档集合；`docs/handover-files.list` 是面向用户阅读的子集。两份 list 都只允许逐文件条目，不递归扫描目录，manifest 不包含其自身，也不替代 Git commit 或大型数据 manifest。

第一阶段在源码/文档提交并 push 后构建 manifest：

```bash
ROOT=C:/Users/Lenovo/Project/lattice-flow-restart
PUBLISHED_REF=origin/wilson-current
python tools/build_handover_manifest.py check-files \
  --root "$ROOT" \
  --files-from "$ROOT/docs/delivery-files.list"
python tools/build_handover_manifest.py build \
  --root "$ROOT" \
  --output "$ROOT/docs/handover-manifest.next.json" \
  --files-from "$ROOT/docs/delivery-files.list" \
  --retention-decision docs/data-retention-decision.json \
  --accepted-git-ref origin/master \
  --published-git-ref "$PUBLISHED_REF"
```

`PUBLISHED_REF` 必须是已 fetch 的 remote-tracking ref，不得用仅存在于本机的 branch 冒充已发布。存在其他已判定无关且接受的 ref 时，逐项增加 `--accepted-git-ref`；共享、待兼容的 ref 不列为 accepted。

构建后：

1. 检查完整 file list、Git state、retention decision、accepted refs 和 identity；
2. 用普通 `verify` 校验 `.next.json` 的 exact bytes；
3. 将 `.next.json` 原子改名为 `docs/handover-manifest.json`；
4. 单独 commit manifest 并再次 push；
5. 从 fresh clone 运行：

```bash
python tools/build_handover_manifest.py verify \
  docs/handover-manifest.json \
  --root "$PWD" \
  --require-release-ready
```

只有最后命令通过才是永久可恢复交付。dirty transitional manifest 可以证明 exact bytes，但其 `release_ready` 必须为 false，不能被称为已永久发布。

## 6. 最终交付门禁

### 源码门禁

- [ ] 重新 fetch remote，并记录共同基线；
- [ ] 分类其他 agent 变化；无关变化原样接受，共享变化完成兼容审查；
- [ ] `docs/delivery-files.list` 通过 `check-files`，且 staging 无目录 pathspec、用户文件、缓存或大型数据；
- [ ] `git diff --cached --check` 通过；
- [ ] 聚焦测试、全套相关测试、Ruff 和 format 通过；
- [ ] reviewer 无 P0/P1；
- [ ] 用户授权 commit/push；
- [ ] retention decision 为 `approved`（hashed restore-check evidence 已列入 delivery manifest）或经用户签署的 `not-required`；
- [ ] source/doc commit 和后续 manifest commit 都已 push；
- [ ] fresh clone 的 `verify --require-release-ready` 通过。

### Kunshan 门禁

- [ ] 部署到新隔离目录；
- [ ] source/environment/input manifests 完整；
- [ ] 登录节点仅执行轻量只读检查；
- [ ] DCU 工作通过批准的 Slurm 生产/测试流程；
- [ ] 每个 attempt 有新结果目录和持久 monitor；
- [ ] artifact、manifest、`DONE` 哈希复核；
- [ ] 小型证据同步回 Git 文档或交付归档；
- [ ] retention decision 已解析：验证 secondary copy/restore，或记录用户接受 single-copy 风险；
- [ ] 未满足 retention gate 时不得宣告 production release 完成。

### 科学口径门禁

- [ ] `J4×J4` 原始 ordered trace 与 flavor/Wick/归一化定义分离；
- [ ] 任何物理 normalization 绑定批准文档及 SHA-256；
- [ ] current-current 二点不冒充 H–J–H C3；
- [ ] artifact smoke 不冒充 ensemble/WT/charge 结论。

## 7. 当前未完成的永久化事项

1. 双 `J4` pair-smoke CLI、subprocess tests 和真实 cfg10000 smoke 尚未完成；
2. 新 pair kernel 加入后的完整相关测试套件尚未重跑；
3. 本地核心文件仍有 modified/untracked 内容；永久 release 依赖用户授权后的精确 staging、commit 和 push；
4. 尚无本项目专用 remote branch、source/doc commit 或 manifest commit；
5. Kunshan 新 pair 工作尚无最终隔离部署 snapshot；
6. 8-cfg production 尚未建立资源批准、job ledger 和结果目录；
7. `docs/data-retention-decision.json` 仍为 `pending`，release-ready hard gate 必须失败；
8. H–J–H charge-normalization 的物理合同仍未批准。
