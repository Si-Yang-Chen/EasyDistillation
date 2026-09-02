# Wilson Current 工作重启操作手册

## 0. 开始前

必须先阅读：

1. `AGENTS.md`；
2. `HANDOVER.md`；
3. `docs/permanence-and-deployment.md`；
4. `docs/ledger/decisions.md`；
5. 若修改 `openspec/`，再完整阅读 `openspec/AGENTS.md`。

当前活动本地仓库应为：

```bash
cd C:/Users/Lenovo/Project/lattice-flow-restart
git rev-parse --show-toplevel
git remote -v
git status --short --branch
```

预期 remote 名为 `origin`，地址为 `git@github.com:Si-Yang-Chen/EasyDistillation.git`。不要只凭目录名判断仓库身份。

## 1. Remote 与多 Agent preflight

每次开始集成或 Kunshan 上线前重新运行：

```bash
git fetch --prune origin
git rev-parse HEAD
git rev-parse origin/master
git log --oneline --decorate --graph -20 --all
git status --short --branch
```

列出候选 ref 相对共同基线的文件差异：

```bash
BASE_REF=origin/master
CANDIDATE_REF=origin/feature-name
MERGE_BASE=$(git merge-base "$BASE_REF" "$CANDIDATE_REF")
git diff --name-status "$MERGE_BASE...$CANDIDATE_REF"
```

执行前将 `CANDIDATE_REF` 赋值为实际 ref；变量为空或 `git merge-base` 失败时停止，不拼接未经验证的 ref 字符串。

分类规则：

- 与本项目路径、import/export、API、schema、数据格式和 runtime 均不相交：记录 commit 后直接接受，无需逐行审查；
- 触及 `lattice/__init__.py`、generator/elemental、insertion、quark_diagram、preset/propagator/slab I/O 或 Current schema：共享依赖，必须做兼容审查和测试；
- 任何情况下不回退、覆盖或清理另一 agent 的变化。

Kunshan 只读 preflight：

```bash
ssh -o BatchMode=yes kunshan '
  cd "$HOME/qedinf/EasyDistillation" &&
  git remote -v &&
  git status --short --branch &&
  git rev-parse HEAD &&
  git for-each-ref --format="%(refname:short) %(objectname:short)" refs/heads refs/remotes &&
  squeue -u "$USER"
'
```

同时检查本项目与其他 owner 的实验目录，不修改它们：

```bash
ssh -o BatchMode=yes kunshan '
  for d in \
    "$HOME/qedinf/experiments/nonlocal-current/scripts" \
    "$HOME/qedinf/experiments/localized-blending/scripts";
  do
    echo "=== $d"
    test -d "$d" && find "$d" -maxdepth 1 -type f -printf "%f\n" | sort
  done
'
```

## 2. 本地环境

若已有依赖完整的 Python 环境，直接使用。需要新环境时：

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

`pip install` 可能访问网络；不要声称执行过未实际执行的安装。

## 3. 当前最小验证

### 3.1 双 Current kernel 聚焦测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_current_v2v_contraction.py
```

当前已观察结果：`6 passed`。该结果只覆盖库级 pair kernel，不覆盖尚未完成的 pair CLI。

### 3.2 Current 核心相关套件

pair CLI 完成后应运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_current.py \
  tests/test_current_consumption.py \
  tests/test_temporal_current_elemental.py \
  tests/test_current_v2v_persistence.py \
  tests/test_current_v2v_contraction.py \
  tests/test_conserved_charge_v2v.py \
  tests/test_quark_diagram_contraction.py \
  experiments/directed-current-v2v/test_contract_existing_vsv_cpu.py \
  experiments/conserved-current-validation/test_measurement_readiness_cpu.py \
  experiments/conserved-current-validation/test_measurement_inventory_cpu.py \
  experiments/conserved-current-validation/test_producer_cpu.py \
  experiments/conserved-current-validation/test_experiment_cpu.py
```

pair kernel 加入前该集合最近结果是 `166 passed, 1 skipped`；必须重跑后才能把该数字归于当前实现。

### 3.3 预检和静态检查

```bash
PYTHONDONTWRITEBYTECODE=1 python test/current_conservation_cpu_precheck.py
python -m ruff format --check lattice tests experiments
python -m ruff check --no-cache lattice tests experiments
git diff --check
```

如果全目录 Ruff 暴露历史无关问题，不修复无关模块；改为对本项目 allowlist 执行并在交接中记录范围。

### 3.4 交接文档 manifest 工具

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q \
  tests/test_handover_manifest.py
```

当前结果：manifest 子集 `14 passed`；与 persistent monitor tests 合计 `19 passed`。覆盖显式 file-list、approved hashed restore evidence、not-required single-copy 决策、ignored/directory/local-branch 拒绝、篡改拒绝、pending retention 拒绝，以及临时 bare remote 中“两阶段 commit/push + fresh clone release verify”。

验证交接文档 exact bytes（允许报告 release blockers）：

```bash
python tools/build_handover_manifest.py verify \
  docs/handover-manifest.json \
  --root C:/Users/Lenovo/Project/lattice-flow-restart
```

验证永久发布门禁：

```bash
python tools/build_handover_manifest.py verify \
  docs/handover-manifest.json \
  --root C:/Users/Lenovo/Project/lattice-flow-restart \
  --require-release-ready
```

普通 verify 只证明清单内文件的 exact bytes；只有带 `--require-release-ready` 的命令通过，才证明文件与 manifest 均已 Git-clean、retention decision 已解析、且记录的 remote-tracking ref 包含 manifest commit。任何清单文件变化后，旧 manifest 应验证失败；不要手改 hash 或 identity。

## 4. 当前 API 位置

- Current term、schema、端点与 spin：`lattice/insertion/current.py`；
- directed one-link raw generation：`lattice/generator/elemental.py::calc_directed_current_raw`；
- artifact persistence、单流和双流 V2V contraction：`lattice/current_elemental.py`；
- equal-time compatibility adapter：`lattice/quark_diagram.py::CurrentVertexAdapter`；
- strict existing-VSV CLI：`experiments/directed-current-v2v/contract_existing_vsv.py`；
- 双流 pair-smoke CLI：`experiments/directed-current-v2v/contract_existing_vsv_pair.py`，强制绑定 handover/source manifest、execution record 与全部依赖文件 bytes；

双流函数 `contract_directed_current_pair_v2v()` 只返回 ordered/unflavored/unsigned connected trace。任何 Wick sign、flavor factor、real-part selection 或 normalization 必须由单独、明确的 observable definition 提供。

## 5. Kunshan 数据只读核验

完整路径见 `docs/kunshan-easydistillation-data-map.md`。目标 8 cfg：

```text
10000 13000 14000 15000 16000 17000 18000 19000
```

轻量完整性检查可以在登录节点执行；不要加载整个大型数组：

```bash
ssh -o BatchMode=yes kunshan 'python3 - <<'"'"'PY'"'"'
from pathlib import Path
B = Path("/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72")
cfgs = (10000,13000,14000,15000,16000,17000,18000,19000)
checks = {
    "VSV": (B / "04.perambulator.localized.nev128_to_nev128.fulltime.src18.np64", range(0,72,4)),
    "PSV": (B / "04.perambulator.localized.nev128_to_np64.fulltime.src72", range(72)),
    "PSP": (B / "04.perambulator.localized.np64_to_np64.fulltime.src18", range(0,72,4)),
}
for name, (root, times) in checks.items():
    wanted = [root / f"{cfg}.t{t:03d}.rank{rank:04d}.npy"
              for cfg in cfgs for t in times for rank in range(4)]
    missing = [str(path) for path in wanted if not path.is_file()]
    print(name, "expected", len(wanted), "present", len(wanted)-len(missing), "missing", len(missing))
PY'
```

已有 count 只是当前事实；消费前还应验证 manifest hash、shape、dtype、轴和 source set。

## 6. cfg10000 真实 pair smoke 恢复点

已有可复用输入：

- directed-current artifact：`/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/results/current-ne1-cfg10000-attempt-02/current-artifact`；
- VSV family manifest：`/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/10000.vsv-timeslices.manifest.json`；
- VSV pattern：数据目录中的 `10000.t{source_time:03d}.rank0000.npy`；
- execution record：`/public/home/siyangchen/BASE/lattice-flow-current-vsv-smoke/20260901-062612/audit/10000.execution-record-login.json`。

在 pair CLI 部署运行前，必须提供与部署 snapshot 一致的 source manifest（`tools/build_handover_manifest.py` 生成），CLI 会校验 identity、必需依赖文件 bytes，并要求 execution record 绑定同一 manifest SHA 与 Git 状态。正确顺序：

1. 完成 CLI 和 subprocess tests（已通过）；
2. 将精确源码部署到新的隔离 snapshot 并生成该 snapshot 的 source manifest；
3. 先运行 `--dry-run`，确保不创建 output；
4. 使用新结果目录运行 read-only `Ne=1` smoke；
5. 复核 result/manifest/`DONE` hashes；
6. 更新证据文档。

## 7. Full-size / 8-cfg 路径

不能直接从 cfg10000 `Ne=1` smoke 跳到物理结论。必须先：

1. 枚举双 point-split endpoints 需要的 source-time coverage；
2. 判断 18-source VSV 是否足够，或是否需要 PSV/PSP/highmode 路径；
3. 对 directed Wilson `J4` 生成成本和 artifact 大小做资源估算；
4. 建立 input manifest、result schema 和新 output root；
5. 使用批准的 production 流程，而不是把生产补算伪装成自动测试；
6. 每个 job 启动持久 monitor。

资源参数若尚未有批准的 production contract，先向用户确认，不自动沿用历史 `72:00:00`。

## 8. 源码永久化操作

详细门禁见 `docs/permanence-and-deployment.md`。当前未经用户授权不得 staging/commit/push。

获授权后：

1. 更新 remote/other-agent preflight；
2. 用 `tools/build_handover_manifest.py check-files --root C:/Users/Lenovo/Project/lattice-flow-restart --files-from C:/Users/Lenovo/Project/lattice-flow-restart/docs/delivery-files.list` 验证逐文件 allowlist；
3. `git add --pathspec-from-file=docs/delivery-files.list`，禁止 `git add .` 或目录 pathspec；
4. 审查 `git diff --cached --name-status` 和 `git diff --cached`；
5. 跑 staged tree 对应测试；
6. reviewer 复核；
7. 分组 commit；
8. push 专用 remote branch；
9. 生成 `docs/handover-manifest.json`，单独 commit 并再次 push；
10. 从全新 clone 运行普通 verify 与 `--require-release-ready`；
11. Kunshan 只部署该 commit 或绑定完整 dirty snapshot identity。

## 9. 故障分类

- 本地没有数据：不是 blocker，去 Kunshan 查；
- manifest/hash/axis 不匹配：输入不兼容，不能静默复用；
- login node CuPy/QUDA 无设备：环境使用错误，改为 Slurm，不增加资源；
- Slurm terminal + 无完整 result/`DONE`：infrastructure/incomplete，不冒充代码失败或成功；
- pre-fix 历史 correlator：superseded，不复用；
- 其他 agent dirty 文件：不得覆盖；使用隔离 snapshot，必要时协调 owner；
- 物理公式未批准：阻止物理 normalization/结论，但不阻止 formula-neutral kernel 和 artifact smoke。

## 10. 结束一轮工作前

更新：

- `TASKBOARD.md` 的时间、当前任务、证据和下一步；
- `HANDOVER.md` 的验收状态；
- 新 Kunshan artifact 对应证据文档；
- `INVENTORY.md` 的 tracked/untracked/排除清单；
- `docs/handover-manifest.json` 的 verify 状态和 identity；
- 若状态稳定变化，更新 `docs/ledger/facts.md`；
- 若策略变化，更新 `docs/ledger/decisions.md`。

不要在动态状态没有同步时宣称工作“可移交”。
