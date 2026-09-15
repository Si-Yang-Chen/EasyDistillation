# EasyDistillation

EasyDistillation 是一个 lattice QCD distillation 计算框架，覆盖从 gauge field / Laplacian eigenvector 到 elemental、perambulator，再到 quark diagram 收缩与关联函数计算的完整流水线，支持 CPU（NumPy）与 GPU（CuPy/PyQuda）双后端。

## 功能一览

- **多后端**：通过 `set_backend` 在 NumPy（CPU）与 CuPy（GPU）之间切换；`check_QUDA` 探测 PyQuda（QUDA GPU 求解器）可用性
- **惰性数据加载**：`FileData` 体系对大型格点数据集做 mmap / 分片（timeslice）读取
- **全类型 propagator**：VSV（perambulator）、VSP、PSV、PSP 四种 layout，及 current elemental 的 V2V / V2P / P2V / P2P
- **自动收缩**：`QuarkDiagram` 把 quark 收缩拓扑编译为 opt_einsum 下标表达式，附带 SymPy 符号化简
- **对称性分析**：cubic group（Oh 双群）irrep 生成、little group 投影、interpolating operator 构造
- **可视化**：从 `Diagram` 对象直接绘制夸克图（可选依赖 feynman/matplotlib）

## 安装

```bash
pip install -e .            # 基础安装（NumPy/SciPy/SymPy/opt-einsum）
pip install -e .[gpu]       # CuPy (CUDA 11.x)
pip install -e .[mpi]       # mpi4py
pip install -e .[viz]       # feynman + matplotlib（quark 图绘制）
pip install -e .[dev]       # pytest/ruff/mypy
```

要求 Python ≥ 3.9。

## 最小示例

π 介子两点函数（完整脚本见 [example/gen_twopt.py](example/gen_twopt.py)）：

```python
from lattice import set_backend, get_backend, preset
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName
from lattice.insertion.mom_dict import momDict_mom9
from lattice.correlator.one_particle import twopoint

set_backend("numpy")  # 或 "cupy"

# 1. 构造插值算符：γ 结构 × 导数 × irrep 投影 × 动量字典
pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

# 2. 声明数据源（lazy，按 cfg 名加载）
elemental = preset.ElementalNpy("<prefix>", ".mom9.npy", [4, 123, 128, 70, 70], 70)
perambulator = preset.PerambulatorNpy("<prefix>", ".peram.npy", [128, 128, 4, 4, 70, 70], 70)
e = elemental.load("<cfg>")
p = perambulator.load("<cfg>")

# 3. 收缩出两点函数，形状 (Nop, Lt)
twopt = twopoint([op_pi], e, p, list(range(128)), 128)
```

更多端到端流程（两粒子关联、GEVP、quark 图、perambulator 生成）见 [docs/workflows.md](docs/workflows.md)。

## 文档

全部文档由代码分析梳理生成，论断附 `path:line` 出处：

| 文档 | 内容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 总体架构、模块依赖图、核心数据流 |
| [docs/modules/](docs/modules/) | 各模块参考（API、数据结构、算法、不变量、相关测试） |
| [docs/data-formats.md](docs/data-formats.md) | 文件格式与 IO 契约（npy/binary/ILDG/timeslice、propagator layout 总表） |
| [docs/workflows.md](docs/workflows.md) | 基于 example/ 的端到端工作流解析 |
| [docs/testing.md](docs/testing.md) | 测试组织、markers、测试数据清单 |

`archive/` 下为旧版文档（只读归档），针对早期代码布局编写，不反映当前代码。

## 运行测试

```bash
pytest -m "not gpu and not mpi"   # CPU 子集（当前 478 passed, 13 skipped）
pytest -m gpu                     # 需要 CuPy
pytest -m mpi                     # 需要 MPI + PyQuda
```

详见 [docs/testing.md](docs/testing.md)。

## License

MIT — see [LICENSE](LICENSE).
