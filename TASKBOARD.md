# 当前阶段任务板

- project_id: `lattice-flow-restart`
- phase: `J4×J4 ensemble readiness and safe Kunshan integration`
- updated_at: `2026-09-03T08:00:00+08:00`
- current_task: `文档收敛完成；推进 full-size 双 point-split endpoint/source coverage 和 8-cfg production contract`
- next_action: `重新核对 remote/Kunshan shared checkout，完成 coverage/cost audit，再决定是否提交 8-cfg production`
- blockers: `Kunshan 共享模块需兼容核对；production 资源需确认；retention pending；H–J–H 仍缺物理合同`
- long_term_plan: `PLAN.md`

## 当前任务

| ID | Status | Description | Evidence | Next action |
| --- | --- | --- | --- | --- |
| ENS-01 | in-progress | 8-cfg localized VSV/PSV/PSP/overlap 输入 coverage 与 endpoint 兼容性审计 | `docs/kunshan-easydistillation-data-map.md`；现有精确计数 | 确定双 `J4` 所需 source/time coverage 和内存成本 |
| ENS-02 | pending | 8-cfg Wilson `J4×J4` production input/result contract | cfg10000 pair smoke 已通过；无 ensemble result | 生成新 manifest、资源估算、输出根和 job ledger |
| ENS-03 | pending | 8-cfg Slurm production、持久 monitor 和结果审计 | 尚无新 production job | 资源确认后按批准流程提交，不伪装成测试 |
| INT-01 | pending | 与 localized-blending 共享模块兼容集成 | Kunshan branch `192ee700...` 触及共享路径 | 重新 diff 双方 ref，保留双方语义并跑兼容测试 |
| PUSH-01 | in-progress | 专用 branch remote 发布和 fresh-clone 验证 | 本地 commits 已完成；用户已授权后续 push | 文档提交后重新 fetch、push、fresh clone 验证 |
| RET-01 | awaiting-user-decision | 大型数据 retention | `docs/data-retention-decision.json` 为 `pending` | 用户选择 `approved` secondary copy 或签署 `not-required` |
| PHY-01 | blocked-by-authority | rho charge/WT H–J–H measurement | readiness v4 `ready=false` | 获得 approved measurement contract 和 C3 定义 |

## 已完成基线

- Wilson Current terms、八方向 directed raw、artifact persistence 和单流/双流 V2V kernel 已实现。
- pair-smoke CLI 已绑定 source manifest、execution record 和依赖文件 bytes。
- cfg10000 `Ne=1` 真实 `J4×J4` pair smoke 已通过；证据见 `docs/kunshan-current-vsv-pair-smoke.md`。
- 本地永久化已形成专用 branch 的分组 commits；tracked worktree 在文档收敛前干净。
- source-manifest 集成和文档收敛后相关测试通过 `195 passed, 1 skipped`；Ruff、format、precheck 和 diff check 通过。

## 最近动态 snapshot

以下只记录本阶段启动和部署判断所需的动态观察；下一次上线前必须重新获取：

- local branch/HEAD：`feature/wilson-current-j4` / `5d91ca6...`（本轮文档收敛待提交）；
- 最近目标 remote：专用 `wilson-current-j4` branch 尚未确认存在；push 已获持续授权；
- Kunshan shared checkout：HEAD `94f8fcd...`，存在 dirty/untracked 变化；不得覆盖；
- Kunshan localized branch：`192ee700...` 修改共享模块，当前未兼容合入；
- 其他 stochastic ref：已判定与本项目无交集时可记录后原样接受；
- 活动作业：上次观察到 `120729358_[0-1]` 为其他 workflow 的 pending job，不能操作。

## 当前阶段验收

- [x] cfg10000 `J4×J4` artifact smoke：通过；
- [x] 本地 delivery allowlist：逐文件检查通过；
- [x] source-manifest/执行记录绑定：pair CLI 测试覆盖；
- [x] source-manifest 集成和文档收敛后完整相关测试；
- [ ] shared checkout/branch 兼容 preflight；
- [ ] 8-cfg endpoint/source coverage 与资源契约；
- [ ] 8-cfg production result + monitor；
- [ ] remote push + fresh-clone verification；（push 已获持续授权）
- [ ] retention decision resolution；
- [ ] H–J–H 物理授权。

## 规则

- 当前任务板只记录当前阶段；长期路线放在 `PLAN.md`。
- 动态 HEAD、dirty、Job ID 和测试数字只放这里或对应 evidence，不写入稳定 ledger。
- 其他 agent 无交集改动记录后直接接受；共享 API/schema/data/runtime 改动必须兼容审查。
- 不覆盖共享 Kunshan checkout，不删除其他 agent 结果，不因本地缺数据停止查找。
