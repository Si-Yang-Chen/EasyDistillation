# Wilson Current 稳定事实归档

> 归档日期：2026-09-03。当前开发入口见 `AGENTS.md`、`README.md`、`TASKBOARD.md` 和 `PLAN.md`。

## Current

- `lattice.insertion.current` 定义版本化 Current terms、Wilson point-split `ConservedVectorCurrent`、端点、spin-aware assembly 和验证接口。
- Conserved vector current 每个方向有 forward `-1/2(r-gamma_mu)` 与 backward `+1/2(r+gamma_mu)` 两项；temporal component 使用不同 bar/field 时间。
- `DirectedCurrentBasis` 顺序为 `+x,+y,+z,-x,-y,-z,+t,-t`。
- `calc_directed_current_raw()` 输出 `(direction,time,momentum,sink_ne,source_ne)`，保留四方向 gauge links。
- legacy spatial `calc_all()`/`current_elemental_all` 不满足 temporal-current contract；`CurrentVertexAdapter` 仅用于 equal-time assembled V2V。

## Artifact 与收缩

- directed-current artifact 使用 content-addressed NPY、原子 manifest、配置/动量/raw-contract、数据 hash 和 gauge/eigenvector source hashes。
- 单流 contraction 逐 term 消费已加载 VSV；backward raw channel 不二次 dagger。
- 双流 kernel 枚举两个 temporal Current 的四个 term pairs，以 `S(field_A->bar_B)` 和 `S(field_B->bar_A)` 闭合 ordered connected V2V trace。
- 双流 raw trace 不隐含 Wick sign、flavor factor、normalization、conjugation、取实部、source average 或 fit。

## 数据与证据

- Kunshan 是大型真实数据和 DCU 环境；Git 只保存 manifest、路径、shape、dtype、轴、hash 和小型证据摘要。
- 目标 8 cfg 的 localized VSV/PSV/PSP 与 overlap 已完成路径/coverage 审计；具体事实见 `docs/kunshan-easydistillation-data-map.md`。
- 单流和双流 cfg10000 smoke 分别见 `docs/kunshan-current-vsv-smoke.md` 与 `docs/kunshan-current-vsv-pair-smoke.md`。
- 现有 legacy current-elemental/meson-current 数据不能静默当作新 Wilson temporal `J4`；二点不能代替 H–J–H C3。
- charge/WT readiness 仍要求 approved measurement contract、projector、flavor weights、C2/C3/ratio 和 time/contact/plateau policy。

## 永久化边界

- 详细永久化、多 Agent 集成、Kunshan 部署、source manifest、结果原子发布、monitor、retention 和 fresh-clone gate 见 `docs/archive/2026-09-03-kunshan-permanence.md`。
- 动态 HEAD、dirty、Job ID、测试数字和当前 blockers 只写 `TASKBOARD.md`。
- retention 必须为 `approved` 或用户签署的 `not-required` 才能通过 release-ready；pending 不得宣告永久发布。
