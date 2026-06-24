# EasyDistillation 文档索引

**生成日期**: 2026-06-01 | **项目版本**: 1.0

---

## 📖 核心文档

### 🚀 [快速参考指南](QUICK_REFERENCE.md)
**适合**: 快速上手、日常参考
- 5分钟快速开始
- 核心API速查
- 常用操作示例
- 性能优化提示

### 🏗️ [项目架构文档](PROJECT_ARCHITECTURE.md)
**适合**: 深入理解、开发扩展
- 完整架构设计
- 详细模块说明
- 开发指南
- 部署运维

---

## 📚 专题文档

### 文件 I/O 系统
- **[FileData 详细文档](FILEDATA_DETAILED.md)** 🔥 **新增**
  - 基于源代码的深度技术分析
  - 所有文件格式的实现细节
  - 内存映射、性能优化、设计模式
  - 二进制、NumPy、ILDG、QDP 等格式对比

### 数据类型与形状
- [数据形状说明](doc/README.md) - 各种数据类型的文件格式
- [传播子理论与使用](doc/propagator_theory_and_usage.md) - 完整的传播子类型说明、形状、加载示例（替代旧 Propagator 系列文档）

### 高级主题
- [传统 Distillation 工作流](docs/DISTILLATION_WORKFLOW.md) - 逐步工作流与完整示例
- [矢量流两点收缩流程](WORKFLOW_ANALYSIS.md) - 基于 `4.contraction.py` 的 Localized Blending 流程
- [局域混合](doc/localized_blending/localized_blending.md) - 理论、实现与附录索引

---

## 💻 示例代码

### 基础示例
- [两点函数生成](example/gen_twopt.py) - 介子能谱计算
- [密度传播子](example/gen_density_peram.py) - VSSV 类型计算
- [动量矩阵](example/gen_twopt_matrix_mom.py) - 色散关系

### 高级示例
- [两粒子关联](example/gen_two_particle_corr.py) - 两介子系统
- [两粒子算符](example/gen_two_particle_opetators.py) - 算符构建
- [流算符](example/gen_two_particle_corr_mom.py) - 带动量

### 绘图工具
- [夸克图绘制](example/gen_multi_draw_diagrams.py) - 可视化工具

---

## 🧪 测试文档

### 单元测试
- [传播子测试](tests/test_perambulator.py)
- [基元测试](tests/test_elemental.py)
- [Gamma矩阵测试](tests/test_gamma.py)
- [收缩测试](tests/test_quark_contract.py)

### 集成测试
- [流算符收缩](test/test_current_contraction.py)
- [对称性验证](test/test_v2p_p2v_symmetry.py)

---

## 🔄 变更管理

### OpenSpec 记录
- [项目上下文](openspec/project.md) - 项目整体说明
- [AI代理指南](openspec/AGENTS.md) - AI辅助开发

### 变更历史
- [稀疏点传播子](openspec/changes/add-sparsened-point-propagator/) - 新功能添加
- [收缩框架](openspec/changes/add-contraction-framework/) - 框架改进

---

## 🎯 按场景查找

### 我是新手，想快速开始
1. 阅读 [快速参考指南](QUICK_REFERENCE.md)
2. 运行 [两点函数示例](example/gen_twopt.py)
3. 查看 [数据形状说明](doc/README.md)

### 我想深入理解架构
1. 阅读 [项目架构文档](PROJECT_ARCHITECTURE.md)
2. 研究核心模块源码
3. 查看测试用例

### 我想添加新功能
1. 参考 [开发指南](PROJECT_ARCHITECTURE.md#开发指南)
2. 查看类似功能的实现
3. 遵循 [代码风格](PROJECT_ARCHITECTURE.md#代码风格)

### 我遇到了问题
1. 查看 [常见问题](QUICK_REFERENCE.md#常见问题)
2. 运行相关测试
3. 检查 [已知问题](PROJECT_ARCHITECTURE.md#已知问题与改进方向)

### 我想优化性能
1. 阅读 [性能优化](PROJECT_ARCHITECTURE.md#性能优化)
2. 查看快速参考的 [优化提示](QUICK_REFERENCE.md#性能优化提示)
3. 检查内存和GPU使用

---

## 🔍 快速查找

### 按关键词

| 关键词 | 相关文档 |
|--------|---------|
| **传播子** | [传播子理论与使用](doc/propagator_theory_and_usage.md) |
| **两点函数** | [示例](example/gen_twopt.py) |
| **三点函数** | [流算符示例](example/gen_two_particle_corr.py) |
| **Gamma矩阵** | [测试](tests/test_gamma.py) |
| **对称性** | [对称性验证](test/test_v2p_p2v_symmetry.py) |
| **GPU加速** | [快速参考](QUICK_REFERENCE.md#后端管理) |
| **内存优化** | [性能优化](QUICK_REFERENCE.md#内存优化) |

### 按模块

| 模块 | 说明 | 文档 |
|------|------|------|
| `preset` | 数据类 | [数据形状](doc/README.md) |
| `generator` | 数据生成 | [架构文档](PROJECT_ARCHITECTURE.md#generator---数据生成器) |
| `quark_diagram` | 夸克图 | [架构文档](PROJECT_ARCHITECTURE.md#quark_diagram---夸克图与收缩) |
| `hadron` | 强子 | [架构文档](PROJECT_ARCHITECTURE.md#hadron---强子物理) |
| `insertion` | 插入算符 | [架构文档](PROJECT_ARCHITECTURE.md#insertion---插入对象) |

---

## 📝 文档贡献

### 如何更新文档
1. 文档位于项目根目录和 `doc/` 目录
2. 使用 Markdown 格式
3. 更新后修改本索引文件

### 文档风格
- 清晰简洁
- 代码示例丰富
- 包含使用场景
- 标注版本信息

---

## 🆘 获取帮助

### 文档问题
- 如果发现文档错误或不清楚的地方
- 建议在 GitHub Issues 中反馈

### 技术问题
1. 先查看相关文档和示例
2. 运行测试验证
3. 查看 Issues 是否有类似问题
4. 提出新 Issue（附上最小示例）

---

## 📊 文档统计

| 文档类型 | 数量 | 说明 |
|---------|------|------|
| 核心文档 | 2 | 架构、快速参考 |
| 专题文档 | 10+ | 数据类型、高级主题 |
| 示例代码 | 11 | 基础到高级 |
| 测试文件 | 19+ | 单元测试、集成测试 |

---

**提示**: 建议从 [快速参考指南](QUICK_REFERENCE.md) 开始！
