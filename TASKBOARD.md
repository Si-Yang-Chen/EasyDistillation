# Taskboard

- project_id: `lattice-flow-restart`
- display_name: `EasyDistillation Wilson conserved-current / Current×Current`
- updated_at: `2026-09-03T03:40:00+08:00`
- current_task: `workspace 可移交快照已整理；永久 release 等待用户授权 commit/push 和 retention 决策`
- next_action: `用户处理永久化门禁时，工程主线恢复 pair-smoke CLI；每次交付文件变化后重建 manifest`
- blockers: `workspace 文档可移交无工程 blocker；永久 release 被未授权 commit/push 与 pending retention hard gate 阻止；H–J–H 另有物理合同 blocker`

## Tasks

| ID | Status | Description | Evidence | Next action |
| --- | --- | --- | --- | --- |
| CUR-01 | done | CurrentTerm/schema、Wilson conserved forward/backward terms、spin 和 endpoint policy | `lattice/insertion/current.py`; Current tests | 保持 API/schema 稳定 |
| CUR-02 | done | 八方向 directed one-link V2V raw generation | `calc_directed_current_raw`; temporal tests | 在共享 elemental 基线上做最终兼容审查 |
| ART-01 | done | directed-current content-addressed artifact persistence 与 source hashes | `lattice/current_elemental.py`; persistence tests | 永久化到 Git commit |
| CON-01 | done | 单 current point-split term-wise existing-VSV contraction | cfg10000 canonical v3 smoke | 保留 canonical evidence |
| CON-02 | done | 双 current temporal V2V term-pair kernel 与 pair-smoke CLI | kernel focused `6 passed`; pair CLI subprocess tests passed | cfg10000 real pair smoke (KUN-01) |
| KUN-01 | pending | cfg10000 `Ne=1` real `J4×J4` pair smoke | 可复用 artifact/VSV/record 已定位 | dry-run 后发布新结果目录 |
| INT-01 | pending | 与 localized-blending 共享模块兼容集成 | branch `192ee700372e2a34a5a6848e3655ea5caa1d7bdc` 修改共享代码 | 三方 diff、保留双方语义、跑兼容测试 |
| PROD-01 | pending | 8-cfg Wilson `J4×J4` input/result/resource contract | VSV/PSV/PSP/overlap inputs 完整 | endpoint coverage 与成本评估，用户确认资源 |
| PROD-02 | pending | 8-cfg Slurm production、monitor 和结果审计 | 尚无新 jobs | 每 attempt 新目录；原子 result + DONE |
| PHY-01 | blocked-physics | rho charge normalization / WT H–J–H measurement | readiness v4 `ready=false` | 获取批准 measurement contract 和 C3 |
| PERM-01 | workspace-complete | 用户交接文档与永久化规范 | reviewer P1/P2 已转为 hard gates；manifest+monitor focused `19 passed` | 普通 manifest verify；永久 release 交给 PERM-02/RET-01 |
| PERM-02 | awaiting-user-authorization | 精确 staging、review、commit、push | 当前 scoped files 仍 modified/untracked | 用户授权后执行 reviewed allowlist |
| RET-01 | awaiting-user-decision | 大型 Kunshan 数据第二副本和 retention policy | `docs/data-retention-decision.json` 为 `pending`，release-ready hard gate 必须失败 | 用户选择 `approved` secondary copy 或签署 `not-required` single-copy 风险 |

## Acceptance Snapshot

- pair kernel focused tests: `6 passed`；
- pre-pair full related suite: `166 passed, 1 skipped`；加入 pair kernel 后尚待重跑；
- cfg10000 single-current real artifact smoke: passed；
- readiness v4: `files_verified=true`, `ready=false`；
- handover manifest + persistent monitor focused tests: `19 passed`；manifest 子集 `14 passed`，覆盖 approved restore evidence、not-required single-copy 决策、ignored/directory/local-branch 拒绝和 bare-remote/fresh-clone release 模拟；
- staging/commit/push: not performed；reviewer 的永久交付 verdict 因此仍为 BLOCKED；
- retention: `pending`，`verify --require-release-ready` 必须失败；
- remote snapshot (`2026-09-03T02:54:52+08:00`): local/Kunshan HEAD `94f8fcdd67defdd14ebc4ff2a1a64b26b36fb28f`, `origin/master=1da08f0a3eb3e2938cb89149a04eb1433de5b0a8`, `origin/feature-stochastic=f6ba3fb024ae254688d8ad778cdea4766ec669c4` and is an ancestor of local HEAD;
- shared-change snapshot: Kunshan localized branch `192ee700372e2a34a5a6848e3655ea5caa1d7bdc` touches shared modules and remains unintegrated;
- Kunshan queue snapshot: array job `120729358_[0-1]`, name `lb-nz-spinfix-`, state `PENDING (Priority)`; belongs to another workflow until ownership is verified, so this project must not alter it.

## Status Vocabulary

- `done`: 代码和对应层级证据已完成；
- `in-progress`: 当前正在实施；
- `paused-at-checkpoint`: 已保存通过的中间实现，当前让位于明确的用户优先事项；
- `workspace-complete`: 当前工作区和 exact-byte manifest 可恢复，但不表示已 commit/push 的永久发布；
- `pending`: 前置工作已知，可继续执行；
- `blocked-physics`: 缺权威物理定义，不能由代码自行决定；
- `awaiting-user-authorization`: 操作受明确用户授权门禁；
- `awaiting-user-decision`: 需要用户选择策略/资源，而非技术失败。

## Output Policy

只把需要报告给用户的小型最终摘要放入 `final-results/`；临时测试、缓存、日志、中间 arrays 和大型格点数据不要放入该目录。任何 `final-results/` 文件进入 Git 前仍需逐项 ownership/size/hash 审核。
