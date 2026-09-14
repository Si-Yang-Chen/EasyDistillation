# Localized Blending 文档索引

本文档是 Localized Blending 方法的主索引文档。完整的文档已拆分为以下几个部分：

## 文档结构

1. **[理论部分](localized_blending_theory.md)** - all-to-all 相关函数的无偏抽样理论与证据边界
   - 条件无偏性的定义，以及低模/补空间完整分解
   - 任意多点函数的联合包含概率与 Horvitz–Thompson 权重
   - 带标签集合划分、共享/独立点集、局域/非局域算符
   - current→meson、非等时/等时 current→current 和高阶函数
   - $N_e$、$N_p$ 对偏差、方差与成本的不同作用
   - v1.4 实测证据、统计限制、实现缺口与验证契约

2. **[程序实现部分](localized_blending_implementation.md)** - 代码实现和使用说明
   - 执行顺序导引
   - 未投影传播子（汇总与形状）
   - current_elemental（元算符）计算
   - 传播子读取与高模投影
   - current_elemental（元算符）读取
   - 夸克图展开（`QuarkDiagram`）
   - 缩并计算（`compute_diagrams_multitime`）

3. **[附录部分](localized_blending_appendix.md)** - 详细的辅助函数和内部实现
   - 附录 A：采样权重自动计算（步骤 3b/3c）
   - 附录 B：下标与约束处理
   - 附录 C：PropagatorWithCurrent 内部方法
   - 附录 D：Current 类内部实现

## 快速导航

- **理论背景**：查看 [理论部分](localized_blending_theory.md)
- **代码使用**：查看 [程序实现部分](localized_blending_implementation.md)
- **实现细节**：查看 [附录部分](localized_blending_appendix.md)

## 文档拆分说明

文档已拆分为多个文件以便于：
- 更快的检索：AI 可以聚焦相关文件
- 更好的上下文管理：减少单次处理的 token 数量
- 更易维护：各部分可以独立更新
- 更清晰的导航：结构更加明确
