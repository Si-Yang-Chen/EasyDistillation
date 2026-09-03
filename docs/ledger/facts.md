# 稳定事实台账

本文件只保留需要跨阶段稳定引用的技术事实；当前状态见 [`TASKBOARD.md`](../../TASKBOARD.md)，长期计划见 [`PLAN.md`](../../PLAN.md)。详细历史事实归档于 [`docs/archive/2026-09-03-current-facts.md`](../archive/2026-09-03-current-facts.md)。

## 稳定事实

- Wilson `ConservedVectorCurrent` 每个方向包含 forward `-1/2(r-gamma_mu)` 和 backward `+1/2(r+gamma_mu)` terms；temporal terms 有 distinct bar/field endpoints。
- directed basis 顺序为 `+x,+y,+z,-x,-y,-z,+t,-t`；raw axis 为 `(direction,time,momentum,sink_ne,source_ne)`。
- `CurrentVertexAdapter` 仅支持 equal-time assembled V2V；temporal point-split terms 使用 term-wise contraction bridge。
- directed artifacts 使用 content-addressed NPY、原子 manifest 和输入/data hashes；双流 kernel 是 ordered、connected、unflavored、unsigned raw trace。
- Kunshan 是大型数据与 DCU 环境；普通 Git 保存 manifest 和证据摘要，不保存大型数据本体。
- 真实二点不能代替 H–J–H C3；artifact smoke 不能代替 WT、荷电归一化或 ensemble 物理结论。

## 规则指针

- Kunshan、多 Agent、ownership 和非破坏性部署：`AGENTS.md`；
- 永久化和上线详细规则：`docs/archive/2026-09-03-kunshan-permanence.md`；
- 当前动态状态：`TASKBOARD.md`；
- 长期路线：`PLAN.md`。
