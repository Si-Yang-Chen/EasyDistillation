# Wilson Current 工作交接

## 先读

- [`AGENTS.md`](AGENTS.md)：强制开发、Kunshan、多 Agent 和非破坏性部署规则；详细永久化说明见 [`docs/archive/2026-09-03-kunshan-permanence.md`](docs/archive/2026-09-03-kunshan-permanence.md)。
- [`TASKBOARD.md`](TASKBOARD.md)：当前阶段和动态状态。
- [`PLAN.md`](PLAN.md)：长期计划。
- [`INVENTORY.md`](INVENTORY.md)：交付/排除边界。
- [`RESTART.md`](RESTART.md)：最小开发流程和 Kunshan 恢复入口。

## 当前目标

构造并验证 Wilson 点分裂守恒流的 temporal `J4×J4` connected V2V 工作流，先完成真实 artifact smoke，再推进 8-cfg ensemble。rho charge normalization / Ward--Takahashi 的 H–J–H C3 是独立工作线，不能由二点函数替代。

## 已完成

- `ConservedVectorCurrent` 八方向 terms、forward/backward Wilson coefficients、temporal endpoints 和 boundary policy；
- 四方向 gauge 加载与 directed raw basis；
- content-addressed Current artifact、manifest、source hashes；
- 单 current 和双 current 的 term-wise VSV contraction；
- pair-smoke CLI，绑定 source manifest、execution record 和依赖文件 bytes；
- source-manifest 集成后的完整相关测试通过 `195 passed, 1 skipped`；pair kernel 聚焦测试 `6 passed`；
- Kunshan cfg10000 `Ne=1` 真实 `J4×J4` pair smoke 通过。

## 真实证据

- 单流 smoke：[`docs/kunshan-current-vsv-smoke.md`](docs/kunshan-current-vsv-smoke.md)；
- 双流 smoke：[`docs/kunshan-current-vsv-pair-smoke.md`](docs/kunshan-current-vsv-pair-smoke.md)；
- 8-cfg 数据路径、shape、dtype、axis 和 hash：[`docs/kunshan-easydistillation-data-map.md`](docs/kunshan-easydistillation-data-map.md)；
- charge/WT readiness：[`docs/kunshan-current-measurement-readiness.md`](docs/kunshan-current-measurement-readiness.md)。

双流 smoke 是 ordered、connected、unflavored、unsigned raw trace，不包含 Wick sign、flavor weight、normalization 或物理结论。

## 当前阶段

详见 [`TASKBOARD.md`](TASKBOARD.md)。当前重点：

1. 审计 full-size 双 point-split endpoint/source coverage；
2. 准备 8-cfg production contract、资源估算和结果目录；
3. 每次 Kunshan 上传前重新核对 shared checkout、其他 agent 修改和活动作业；
4. 与 localized-blending 的共享模块做兼容审查后再部署。

## 永久化状态

- 本地分支已有分组 commits，后续专用 remote branch push 已获持续授权；
- Kunshan shared checkout 不得覆盖，其他 agent 无交集改动可记录后原样接受，共享依赖必须审查；详细永久化规则已归档，开发者无需日常阅读。
- 大型数据留在 Kunshan，Git 只保存 manifest、路径、shape、axis、hash 和小型证据；
- retention decision 仍为 `pending`，所以 `verify --require-release-ready` 必须失败；
- 交接文档 exact-byte manifest：[`docs/handover-manifest.json`](docs/handover-manifest.json)。

## 当前限制

- 尚未生成 8-cfg Wilson `J4×J4` ensemble result；
- 现有 legacy `current_elemental_all` 和 meson-current 二点不能直接当作新 Wilson `J4` 或 H–J–H C3；
- 物理公式、projector、flavor weights、ratio、contact/plateau policy 未授权前，不生成 charge/WT 物理结论；
- 不把本地测试或单配置 artifact smoke 冒充 ensemble 统计结论。
