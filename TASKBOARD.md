# 当前阶段任务板

- project_id: `lattice-flow-restart`
- phase: `J4×J4 ensemble readiness and safe Kunshan integration`
- updated_at: `2026-09-04T01:45:00+08:00`
- current_task: `文档结构收敛和专用 branch 已 push；完成 manifest fresh-clone 验证，随后推进 full-size coverage audit`
- next_action: `刷新并 push published handover manifest，完成 strict fresh-clone 文档验证；随后重新核对 Kunshan shared checkout`
- blockers: `Kunshan 共享模块需兼容核对；production 资源需确认；retention pending；H–J–H 仍缺物理合同`
- long_term_plan: `PLAN.md`

## 当前任务

| ID | Status | Description | Evidence | Next action |
| --- | --- | --- | --- | --- |
| ENS-01 | in-progress | 8-cfg localized VSV/PSV/PSP/overlap 输入 coverage 与 endpoint 兼容性审计 | `docs/kunshan-easydistillation-data-map.md`；现有精确计数 | 确定双 `J4` 所需 source/time coverage 和内存成本 |
| ENS-02 | pending | 8-cfg Wilson `J4×J4` production input/result contract | cfg10000 pair smoke 已通过；无 ensemble result | 生成新 manifest、资源估算、输出根和 job ledger |
| ENS-03 | pending | 8-cfg Slurm production、持久 monitor 和结果审计 | 尚无新 production job | 资源确认后按批准流程提交，不伪装成测试 |
| INT-01 | pending | 与 localized-blending 共享模块兼容集成 | Kunshan branch `192ee700...` 触及共享路径 | 重新 diff 双方 ref，保留双方语义并跑兼容测试 |
| PUSH-01 | done | 专用 branch remote 发布和 fresh-clone 验证 | local/remote branch 已同步；fresh clone 的 70 项清单、manifest 和 import 验证通过 | release-ready 仍由 RET-01 独立控制 |
| RET-01 | awaiting-user-decision | 大型数据 retention | `docs/data-retention-decision.json` 为 `pending` | 用户选择 `approved` secondary copy 或签署 `not-required` |
| PHY-01 | blocked-by-authority | rho charge/WT H–J–H measurement | readiness v4 `ready=false` | 获得 approved measurement contract 和 C3 定义 |

## 已完成基线

- Wilson Current terms、八方向 directed raw、artifact persistence 和单流/双流 V2V kernel 已实现。
- pair-smoke CLI 已绑定 source manifest、execution record 和依赖文件 bytes。
- cfg10000 `Ne=1` 真实 `J4×J4` pair smoke 已通过；证据见 `docs/kunshan-current-vsv-pair-smoke.md`。
- 本地永久化已形成专用 branch 的分组 commits；tracked worktree 在文档收敛前干净。
- source-manifest 集成后的完整相关测试通过 `195 passed, 1 skipped`；文档收敛后的 manifest/monitor 测试 `20 passed`；Ruff、format、precheck 和 diff check 通过。

## 最近动态 snapshot

以下只记录本阶段启动和部署判断所需的动态观察；下一次上线前必须重新获取：

- local/remote branch：`feature/wilson-current-j4` / `8e83e334...`，已完成 fast-forward push；本轮 taskboard/manifest refresh 后待再次 push；
- published ref：`origin/feature/wilson-current-j4` 当前指向 `8e83e334...`；taskboard/manifest refresh 后需再次 push manifest commit；
- Kunshan shared checkout：HEAD `94f8fcdd...`，20 条 dirty/untracked 状态；不得覆盖；
- Kunshan localized branch：`192ee700372e2a34a5a6848e3655ea5caa1d7bdc` 修改共享模块，当前未兼容合入；
- 其他 stochastic ref：已判定与本项目无交集时可记录后原样接受；
- 活动作业：最近 preflight 时队列为空；此前 `120729358_[0-1]` 属于其他 workflow，不能操作。

## 当前阶段验收

- [x] cfg10000 `J4×J4` artifact smoke：通过；
- [x] 本地 delivery allowlist：逐文件检查通过；
- [x] source-manifest/执行记录绑定：pair CLI 测试覆盖；
- [x] source-manifest 集成和文档收敛后完整相关测试；
- [ ] shared checkout/branch 兼容 preflight；
- [ ] 8-cfg endpoint/source coverage 与资源契约；
- [ ] 8-cfg production result + monitor；
- [x] remote push + fresh-clone source/manifest/import verification；release-ready 仍受 RET-01 约束；
- [ ] retention decision resolution；
- [ ] H–J–H 物理授权。

## 规则

- 当前任务板只记录当前阶段；长期路线放在 `PLAN.md`。
- 动态 HEAD、dirty、Job ID 和测试数字只放这里或对应 evidence，不写入稳定 ledger。
- 其他 agent 无交集改动记录后直接接受；共享 API/schema/data/runtime 改动必须兼容审查。
- 不覆盖共享 Kunshan checkout，不删除其他 agent 结果，不因本地缺数据停止查找。
