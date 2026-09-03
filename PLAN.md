# Wilson Current 长期计划

本文件只记录跨阶段计划。当前正在做什么、最近作业和动态 blocker 只写入 `TASKBOARD.md`。

## 目标

### P1: 完成 Wilson `J4×J4` ensemble

状态：`planned`

- 审计 8 个配置的双 point-split endpoint/source coverage；
- 确定 full-size contraction 是否需要 VSV、PSV、PSP 或 high-mode 路径；
- 准备生产 input manifest、资源估算、result schema 和 Slurm ledger；
- 在 Kunshan 新隔离 snapshot 中运行 8-cfg production；
- 对结果做逐配置、逐时间、逐文件 hash 和统计完整性检查。

验收：8 个配置的机器可读结果、输入 lineage、作业记录和 monitor 均可复现。该结果仍需和物理定义分开，不自动等于 WT 或荷电归一化结论。

### P2: 完成 rho charge/WT 测量

状态：`blocked-by-authority`

- 获取 approved measurement contract；
- 明确 flavor/electric-charge weights、source/sink projector、C2/C3/ratio 公式；
- 明确 current/contact/plateau/boundary time policy；
- 生成真正的 hadron-current-hadron C3；
- 运行 `(Ncfg,Lt)` charge/WT ensemble gates。

当前 readiness 证据：`docs/kunshan-current-measurement-readiness.md`。决策清单：`docs/conserved-current-measurement-decisions.md`。

### P3: 完成共享模块集成

状态：`pending`

- 每次上线前重新 fetch remote；
- 无交集的其他 agent 改动记录 ref 后原样接受，不做逐行审查；
- 触及 `lattice/__init__.py`、generator、insertion、quark diagram、preset、slab reader、schema 或 launcher 的改动必须兼容审查；
- 使用新的 Kunshan 隔离源码目录，不覆盖其他 agent checkout。

### P4: 完成长期可恢复发布

状态：`pending`

- 专用 branch 已有本地分组 commits，后续 push 不需逐次授权；
- 从 remote fresh clone 验证源码和 handover manifest；
- retention decision 选择 `approved`（secondary copy + restore check）或用户签署 `not-required`；
- 保留 Kunshan 大型数据的 manifest、path、shape、axis、hash 和恢复责任；
- 明确大型数据第二副本和 retention 周期。

## 计划管理规则

- 长期计划只在本文件更新；不要把未来季度/阶段任务复制到 `TASKBOARD.md`。
- 完成一个长期目标后，在本文件保留状态和验收指针，在 `TASKBOARD.md` 只写当前新阶段。
- 计划状态不代替结果证据；每个 `done` 必须指向源码测试、Kunshan artifact 或物理审批记录。
