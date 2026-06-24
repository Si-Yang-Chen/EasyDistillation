# EasyDistillation 项目架构文档

**生成日期**: 2026-06-01
**版本**: 1.0
**作者**: AI Assistant (GLM-5)

---

## 📋 目录

1. [项目概述](#项目概述)
2. [技术栈与依赖](#技术栈与依赖)
3. [项目结构](#项目结构)
4. [核心架构设计](#核心架构设计)
5. [模块详解](#模块详解)
6. [数据流与工作流](#数据流与工作流)
7. [使用指南](#使用指南)
8. [开发指南](#开发指南)
9. [测试策略](#测试策略)
10. [部署与运维](#部署与运维)
11. [已知问题与改进方向](#已知问题与改进方向)

---

## 🎯 项目概述

### 简介
EasyDistillation 是一个基于 Python 的格点 QCD（量子色动力学）蒸馏计算框架。它提供了完整的工具链，用于：
- 生成蒸馏基矢量和传播子
- 构建强子算符和相关函数
- 计算介子、重子能谱
- 处理两点和三点函数

### 核心特性
- ✅ **模块化设计**: 清晰的模块划分，便于扩展和维护
- ✅ **多后端支持**: 支持 NumPy、CuPy、PyQuda 等多种计算后端
- ✅ **GPU 加速**: 集成 QUDA 库实现 GPU 加速计算
- ✅ **MPI 并行**: 支持分布式计算，适用于大规模格点
- ✅ **惰性加载**: 磁盘数据惰性加载，优化内存使用
- ✅ **符号计算**: 使用 SymPy 处理群论和对称性问题

### 适用场景
- 格点 QCD 能谱计算
- 强子结构研究
- 算符构建和收缩计算
- 蒸馏方法相关研究

---

## 🔧 技术栈与依赖

### 核心依赖
```python
# 数值计算
numpy          # 基础数值计算
scipy          # 科学计算工具
opt_einsum     # 优化的 Einstein 求和

# 符号计算
sympy          # 群论和对称性计算

# GPU 加速
cupy           # CUDA 加速 (可选)
pyquda         # QUDA 接口 (可选)

# 并行计算
mpi4py         # MPI 并行 (可选)
```

### 可选依赖
- **可视化**: matplotlib, graphviz
- **数据格式**: h5py, pickle
- **性能分析**: line_profiler, memory_profiler

---

## 📁 项目结构

### 目录树
```
EasyDistillation/
├── lattice/                 # 核心库代码
│   ├── __init__.py
│   ├── backend.py          # 后端管理（NumPy/CuPy）
│   ├── base_types.py       # 基础类型定义
│   ├── constant.py         # 物理常数
│   ├── data.py             # 数据容器
│   ├── dispatch.py         # 多派发系统
│   ├── hadron.py           # 强子类定义与关联函数计算
│   ├── hadron_irrep.py     # 强子不可约表示
│   ├── quark_diagram.py    # 夸克图、收缩与夸克收缩算法（主入口）
│   ├── quark_draw.py       # 夸克图绘制
│   ├── quark_contract.py   # 夸克收缩算法（DDD 图符号计算）
│   ├── flavor_structure.py # 味道结构
│   ├── spatial_structure.py# 空间结构
│   ├── group_projection.py # 群投影
│   ├── preset.py           # 预定义数据类
│   │
│   ├── quark_diagram/      # 夸克图子包（架构拆分中）
│   │   ├── _diagram_core.py    # 核心数据结构与算法
│   │   ├── _types.py           # 类型定义
│   │   ├── _propagator.py      # 传播子相关逻辑
│   │   ├── _contraction.py     # 收缩计算
│   │   ├── _diagram.py         # 图的表示
│   │   └── _expr.py            # 表达式与 quark_contract
│   │
│   ├── filedata/           # 文件 I/O 模块
│   │   ├── abstract.py     # 抽象基类
│   │   ├── binary.py       # 二进制文件
│   │   ├── ildg.py         # ILDG 格式
│   │   ├── ndarray.py      # NumPy 数组文件
│   │   ├── timeslice.py    # 时间片格式
│   │   └── sliceloader.py  # 切片加载器
│   │
│   ├── generator/          # 数据生成模块
│   │   ├── eigenvector.py      # 本征矢量生成器
│   │   ├── elemental.py        # 基元生成器
│   │   ├── perambulator.py     # 传播子生成器
│   │   ├── noisevector.py      # 噪声矢量生成器
│   │   ├── density_perambulator.py
│   │   ├── generalized_perambulator.py
│   │   ├── displacement_elemental.py
│   │   └── sparsened_point.py  # 稀疏点生成
│   │
│   ├── insertion/          # 插入对象模块
│   │   ├── gamma.py        # Gamma 矩阵
│   │   ├── derivative.py   # 导数算符
│   │   ├── gauge_link.py   # 规范链
│   │   ├── phase.py        # 相位因子
│   │   └── mom_dict.py     # 动量字典
│   │
│   ├── symmetry/           # 对称性模块
│   │   ├── group_generator.py  # 群生成器
│   │   ├── hardcoded_rep.py    # 硬编码表示
│   │   ├── gen_hardcoded_rep.py# 硬编码表示生成
│   │   ├── two_particle.py     # 两粒子系统
│   │   ├── sympy_utils.py      # SymPy 工具函数
│   │   └── utils.py            # 工具函数
│   │
│   └── correlator/         # 关联函数模块
│       ├── one_particle.py     # 单粒子关联函数
│       ├── two_particles.py    # 两粒子关联函数
│       └── disperion_relation.py # 色散关系
│
├── tests/                  # 单元测试 (24 个测试文件)
│   ├── test_perambulator.py
│   ├── test_elemental.py
│   ├── test_eigenvector.py
│   ├── test_gamma.py
│   ├── test_gauge_link.py
│   ├── test_quark_contract.py
│   ├── test_quark_diagram_contraction.py
│   ├── test_current_elemental.py
│   ├── test_propagator_with_current.py
│   ├── test_sampling_weight.py
│   ├── test_diagram.py
│   ├── test_flavor_structure.py
│   ├── test_spatial_structure.py
│   ├── test_group_projection.py
│   ├── test_symmery.py
│   └── ... (24 个测试文件)
│
├── test/                   # 集成测试和临时测试 (16 个文件)
│   ├── test_current_contraction.py
│   ├── test_current_vertex.py
│   ├── test_v2p_p2v_symmetry.py
│   ├── test_gauge_links_methods.py
│   └── ...
│
├── example/                # 使用示例 (9 个文件)
│   ├── gen_twopt.py            # 两点函数生成
│   ├── gen_twopt_diagram.py    # 两点函数图生成
│   ├── gen_twopt_matrix_mom.py # 动量矩阵两点函数
│   ├── gen_density_peram.py    # 密度传播子
│   ├── gen_two_particle_corr.py     # 两粒子关联
│   ├── gen_two_particle_corr_mom.py # 动量两粒子关联
│   ├── gen_two_particle_opetators.py # 两粒子算符
│   ├── gen_multi_draw_diagrams.py   # 多图绘制
│   └── hardcoding_OhD.py            # O_h^D 群硬编码
│
├── examples/               # 新示例 (2 个文件)
│   ├── test_propagator_psv.py     # PSV 传播子测试
│   └── 4_contraction.py           # 四点收缩
│
├── doc/                    # 文档
│   ├── README.md           # 数据形状说明
│   ├── propagator_theory_and_usage.md # 传播子理论与使用
│   ├── unify_vertex_point_color_indices.md # QuarkDiagram方法文档
│   └── localized_blending/  # 局域混合文档
│
├── docs/                   # 项目文档
│   └── DISTILLATION_WORKFLOW.md # 蒸馏工作流说明
│
├── openspec/               # 规范和变更管理
│   ├── project.md          # 项目上下文
│   ├── AGENTS.md           # AI 代理说明
│   └── changes/            # 变更记录
│       ├── add-sparsened-point-propagator/
│       └── add-contraction-framework/
│
└── 4.contraction.py        # 当前开发中的收缩脚本
```

### 统计信息
- **Python 文件总数**: 105 个
- **核心代码行数**: ~49,853 行
- **测试文件**: 40 个（`tests/` 24 个 + `test/` 16 个）
- **示例文件**: 11 个（`example/` 9 个 + `examples/` 2 个）

---

## 🏗️ 核心架构设计

### 设计哲学

1. **分层架构**
   ```
   用户脚本层
       ↓
   高层 API 层 (hadron, quark_diagram)
       ↓
   中层抽象层 (generator, insertion, correlator)
       ↓
   底层数据层 (filedata, preset)
       ↓
   基础设施层 (backend, dispatch)
   ```

2. **组合优于继承**
   - 大量使用 Mixin 模式（如 `PerambulatorBinary(BinaryFile, Perambulator)`）
   - 功能组合而非深层继承

3. **惰性加载**
   - 文件数据按需加载
   - 减少内存占用

4. **多派发机制**
   - 使用 `Dispatch` 类实现基于类型的函数派发
   - 支持扩展新的数据类型

### 核心类关系

#### 数据类型层次
```
FileData (抽象基类)
    ├── BinaryFile
    ├── NdarrayFile
    ├── IldgFile
    └── QDPLazyDiskMapObjFile

物理数据类型 (Mixin 类)
    ├── GaugeField
    ├── Eigenvector
    ├── Elemental
    ├── Perambulator
    ├── PropagatorPSV/VSP/PSP
    └── PointSource

具体实现类 (组合)
    ├── GaugeFieldIldg(IldgFile, GaugeField)
    ├── EigenvectorNpy(NdarrayFile, Eigenvector)
    ├── PerambulatorNpy(NdarrayFile, Perambulator)
    └── PropagatorPSVNpy(NdarrayFile, PropagatorPSV)
```

#### 生成器层次
```
Generator (抽象概念)
    ├── EigenvectorGenerator
    ├── ElementalGenerator
    ├── PerambulatorGenerator
    ├── DensityPerambulatorGenerator
    ├── GeneralizedPerambulatorGenerator
    └── DisplacementElementalGenerator
```

---

## 📦 模块详解

### 1. `lattice.backend` - 后端管理

**功能**: 统一管理数值计算后端（NumPy/CuPy）

**关键函数**:
```python
get_backend() -> module      # 获取当前后端
set_backend(name: str)       # 设置后端 ('numpy' or 'cupy')
check_QUDA() -> bool         # 检查 QUDA 是否可用
log_gpu_memory()             # 记录 GPU 内存使用
```

**设计模式**: 策略模式 + 单例模式

---

### 2. `lattice.filedata` - 文件 I/O

**功能**: 处理各种格式的格点数据文件，提供统一的惰性加载与内存映射接口。

**详细文档**: [FILEDATA_DETAILED.md](FILEDATA_DETAILED.md)

#### 架构概要

```
File (文件加载器) → get_file_data() → FileData (数据访问) → __getitem__() → ndarray
```

#### 支持的文件格式

| 格式 | 类 | 典型扩展名 | 时间片支持 |
|------|-----|----------|----------|
| **Binary** | `BinaryFile` | `.peram`, `.bin` | ❌ |
| **NumPy** | `NdarrayFile` | `.npy` | ❌ |
| **NumPy时间片** | `NdarrayTimeslicesFile` | `.t???.npy` | ✅ |
| **ILDG** | `IldgFile` | `.lime` | ❌ |
| **QDP时间片** | `QDPLazyDiskMapObjFile` | `.mod` | ✅ |

#### 使用示例

```python
from lattice import PerambulatorTimeslicesNpy

# 推荐：时间片格式，按需加载
peram = PerambulatorTimeslicesNpy(
    prefix="/data/cfg_",
    suffix=".t???.npy",
    shape=[128, 128, 4, 4, 70, 70],
    totNe=70,
)
peram_data = peram.load("1000")
data_t10 = peram_data[(10,)]
```

内存映射、页对齐、字节序转换、性能统计等实现细节见 [FILEDATA_DETAILED.md](FILEDATA_DETAILED.md)。

---

### 3. `lattice.preset` - 预定义数据类

**功能**: 提供常用的物理数据类实现，所有类均继承自 `File` 基类

#### 规范场 (GaugeField)
| 类名 | 基类 | 文件格式 |
|------|------|---------|
| `GaugeFieldTimeSlice` | `QDPLazyDiskMapObjFile` | QDP .mod |
| `GaugeFieldIldg` | `IldgFile` | ILDG .lime |
| `GaugeFieldBinary` | `BinaryFile` | 二进制 |

#### 本征矢量 (Eigenvector)
| 类名 | 基类 | 文件格式 |
|------|------|---------|
| `EigenvectorTimeSlice` | `QDPLazyDiskMapObjFile` | QDP .mod |
| `EigenvectorNpy` | `NdarrayFile` | NumPy .npy |

#### 传播子 (Perambulator/Propagator)
| 类名 | 基类 | 文件格式 |
|------|------|---------|
| `PerambulatorBinary` | `BinaryFile` | 二进制 |
| `PerambulatorNpy` | `NdarrayFile` | NumPy .npy |
| `PerambulatorTimeslicesNpy` | `NdarrayTimeslicesFile` | NumPy 时间片 |
| `PropagatorPSVNpy` | `NdarrayFile` | NumPy .npy |
| `PropagatorPSVTimeslicesNpy` | `NdarrayTimeslicesFile` | NumPy 时间片 |
| `PropagatorVSPNpy` | `NdarrayFile` | NumPy .npy |
| `PropagatorVSPTimeslicesNpy` | `NdarrayTimeslicesFile` | NumPy 时间片 |
| `PropagatorPSPNpy` | `NdarrayFile` | NumPy .npy |
| `PropagatorPSPTimeslicesNpy` | `NdarrayTimeslicesFile` | NumPy 时间片 |

#### 元素基 (Elemental)
| 类名 | 基类 | 文件格式 |
|------|------|---------|
| `ElementalBinary` | `BinaryFile` | 二进制 |
| `ElementalNpy` | `NdarrayFile` | NumPy .npy |
| `CurrentElementalV2P` | `NdarrayFile` | NumPy .npy |
| `CurrentElementalP2V` | `NdarrayFile` | NumPy .npy |

#### 其他
| 类名 | 基类 | 文件格式 |
|------|------|---------|
| `PointSourceNpy` | `NdarrayFile` | NumPy .npy |
| `OverlapMatrixNpy` | `NdarrayFile` | NumPy .npy |
| `OnePointNpy` | `NdarrayFile` | NumPy .npy |
| `Jpsi2gammaNpy` | `NdarrayFile` | NumPy .npy |
| `Jpsi2gammaBinary` | `BinaryFile` | 二进制 |

---

### 4. `lattice.generator` - 数据生成器

**功能**: 生成各种蒸馏方法所需的数据

#### 关键生成器

##### `EigenvectorGenerator`
生成 Laplace 算符的本征矢量和本征值
```python
gen = EigenvectorGenerator(
    latt_size=[16, 16, 16, 64],
    gauge_field=gauge_field,
    Ne=70,        # 本征矢量数量
    tol=1e-9      # 收敛容差
)
evecs, evals = gen.calc(t=0)  # 返回 (本征矢量, 本征值)
# evecs shape: [Ne, Lz, Ly, Lx, Nc]
# evals shape: [Ne]
```

##### `ElementalGenerator`
生成蒸馏基元（elemental）
```python
gen = ElementalGenerator(
    eigenvector=eigenvector,
    noise_vector=noise_vector
)
elementals = gen.calc()
```

##### `PerambulatorGenerator`
生成传播子（最核心）
```python
gen = PerambulatorGenerator(
    latt_size=[16, 16, 16, 64],
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,
    mass=0.09253,
    tol=1e-9,
    maxiter=1000,
    xi_0=4.8965,  # 各向异性参数
    nu=0.86679,
    clover_coeff_t=0.8549165664,
    clover_coeff_r=2.32582045,
    t_boundary=-1,
    multigrid=False,  # 可选多重网格
    MRHS=False  # 可选多右端项
)
perams = gen.calc()
```

**GPU 加速**: 通过 PyQuda 调用 QUDA 库实现 GPU 加速

##### `generate_sparsened_points`
生成稀疏点源（独立函数，非类）
```python
from lattice.generator import generate_sparsened_points

points = generate_sparsened_points(
    latt_size=[16, 16, 16, 64],
    num_points=216,
    seed=42  # 可选随机种子
)
# 返回形状: (Np, Lt, 3) 的 int32 坐标数组
# points[p, t, :] = [x, y, z], 其中每个时间片 t 内所有 Np 个点互不重复
```

---

### 5. `lattice.hadron` - 强子物理

**功能**: 定义强子态和不可约表示

#### `Hadron` 类
```python
class Hadron:
    def __init__(self, irep_row: Expr, flavor_structure: Expr):
        self._irrep_row = irep_row  # 不可约表示行
        self._flavor_structure = flavor_structure  # 味道结构

    def set_time(self, time: int) -> "Hadron":
        """设置时间标签"""

    def conjugate(self) -> "Hadron":
        """共轭变换"""
```

#### 辅助函数
- `set_time_in_expr()`: 修改表达式中的时间标签
- `operator_conjugate()`: 算符共轭

---

### 6. `lattice.insertion` - 插入对象

**功能**: 定义各种插入算符

#### `gamma.py` - Gamma 矩阵
```python
from lattice.insertion import gamma

# Dirac gamma 矩阵
gamma_1, gamma_2, gamma_3, gamma_4 = gamma.Gamma1, gamma.Gamma2, ...

# Gamma 矩阵组合
gamma_5 = gamma.Gamma5
gamma_mu_nu = gamma.GammaMuNu(mu, nu)
```

#### `derivative.py` - 导数算符
```python
from lattice.insertion import derivative

# 协变导数
D_mu = derivative.CovariantDerivative(mu)
```

#### `gauge_link.py` - 规范链
```python
from lattice.insertion import gauge_link

# 构建规范链
link = gauge_link.GaugeLink(path=[(0, 1), (1, 1), ...])
```

---

### 7. `lattice.symmetry` - 对称性与群论

**功能**: 处理格点对称性、不可约表示

#### 关键模块

##### `group_generator.py`
生成点群、小群等对称群
```python
from lattice.symmetry import genLittleGroupIrrep

# 生成小群不可约表示
irreps = genLittleGroupIrrep(momentum, group_type="Oh")
```

##### `hardcoded_rep.py`
硬编码的群表示矩阵（性能优化）

##### `two_particle.py`
两粒子系统的对称性分析

**数学工具**:
- Wigner-Eckart 定理
- 群投影算符
- 小群约化

---

### 8. `lattice.correlator` - 关联函数

**功能**: 计算各种关联函数

#### `one_particle.py`
单粒子关联函数（能谱）

#### `two_particles.py`
两粒子关联函数（散射）

#### `disperion_relation.py`
色散关系分析

---

## 🔄 数据流与工作流

### 数据依赖图
```
GaugeField
    ↓
EigenvectorGenerator
    ↓
Eigenvector ──────┐
    ↓              │
ElementalGenerator │
    ↓              │
Elemental          │
    ↓              │
PerambulatorGenerator
    ↓              │
Perambulator       │
    ↓              │
Meson/Current ◄────┘
    ↓
Correlator
```

### 详细工作流文档

传统 distillation 工作流的完整说明，包括：
- 各步骤的数学背景
- 详细的代码示例
- 参数配置说明
- 性能优化建议

详见: [docs/DISTILLATION_WORKFLOW.md](docs/DISTILLATION_WORKFLOW.md)

**理论基础**: [arXiv:0905.2160](https://arxiv.org/abs/0905.2160)

---

## 📚 使用指南

### 安装

#### 基础安装
```bash
# 克隆仓库
git clone <repository-url>
cd EasyDistillation

# 安装依赖
pip install numpy scipy sympy opt-einsum
```

#### GPU 支持（可选）
```bash
# 安装 CuPy
pip install cupy-cuda11x  # 根据你的 CUDA 版本

# 安装 PyQuda
pip install pyquda
```

#### MPI 支持（可选）
```bash
pip install mpi4py
```

### 快速开始

日常 API 速查与最小示例见 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)。

完整 distillation 工作流（数学背景、逐步示例、性能建议）见 [docs/DISTILLATION_WORKFLOW.md](docs/DISTILLATION_WORKFLOW.md)。

含 Localized Blending 的矢量流两点收缩流程见 [WORKFLOW_ANALYSIS.md](WORKFLOW_ANALYSIS.md)。

内存管理、GPU 加速、批处理等最佳实践见 [QUICK_REFERENCE.md#常用操作](QUICK_REFERENCE.md#常用操作) 与 [QUICK_REFERENCE.md#性能优化提示](QUICK_REFERENCE.md#性能优化提示)。

---

## 🛠️ 开发指南

### 代码风格

#### 命名约定
- **类名**: PascalCase（如 `PerambulatorGenerator`）
- **函数名**: snake_case（如 `gen_correlator`）
- **变量名**: snake_case
- **常量**: UPPER_SNAKE_CASE（如 `Nc`, `Ns`）

#### 文档字符串
```python
def calculate_weight(self, L: int, Np: int) -> float:
    """
    Calculate sampling weight for point sources.

    Args:
        L: Spatial lattice size
        Np: Number of points in sampling set

    Returns:
        Sampling weight (compensation factor)

    Example:
        >>> weight = calculate_weight(16, 216)
        >>> print(f"Weight: {weight:.2f}")
    """
```

#### 类型注解
```python
from typing import List, Tuple, Optional, Union

def load_data(
    filename: str,
    shape: List[int],
    dtype: str = "<c16"
) -> np.ndarray:
    """Load data from file."""
```

### 添加新功能

#### 1. 添加新的数据类型
```python
# 在 lattice/preset.py 中定义
class MyNewData:
    def __init__(self, elem: FileMetaData, param: int):
        self.elem = deepcopy(elem)
        self.param = param

# 创建具体实现
class MyNewDataNpy(NdarrayFile, MyNewData):
    def __init__(self, prefix: str, suffix: str, param: int):
        super().__init__()
        MyNewData.__init__(self, FileMetaData([...], "<c16", 0), param)
        self.prefix = prefix
        self.suffix = suffix

# 在 lattice/__init__.py 中导出
from .preset import MyNewData, MyNewDataNpy
```

#### 2. 添加新的生成器
```python
# 在 lattice/generator/my_generator.py 中
class MyGenerator:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def calc(self) -> np.ndarray:
        """Generate data."""
        # 实现逻辑
        return result

# 在 lattice/generator/__init__.py 中导出
from .my_generator import MyGenerator
```

### 调试技巧

#### 1. 检查数据形状
```python
from lattice import PerambulatorNpy

peram = PerambulatorNpy(...)
data = peram.load("1000")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")
print(f"Memory: {data.nbytes / 1024**3:.2f} GB")
```

#### 2. 性能分析
```python
import time
from lattice import log_gpu_memory

start = time.time()
result = expensive_operation()
log_gpu_memory()
print(f"Time: {time.time() - start:.2f}s")
```

#### 3. 使用测试
```bash
# 运行特定测试
pytest tests/test_perambulator.py -v

# 运行所有测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=lattice
```

---

## 🧪 测试策略

### 测试组织

#### 单元测试 (`tests/`)
- `test_gamma.py`: Gamma 矩阵测试
- `test_perambulator.py`: 传播子测试
- `test_elemental.py`: 基元测试
- `test_eigenvector.py`: 本征矢量测试
- `test_quark_contract.py`: 收缩测试
- `test_quark_diagram_contraction.py`: 夸克图收缩测试
- `test_current_elemental.py`: 流算符基元测试
- `test_propagator_with_current.py`: 含流传播子测试
- `test_sampling_weight.py`: 采样权重测试
- `test_diagram.py`: 图结构测试
- `test_flavor_structure.py`: 味道结构测试
- `test_spatial_structure.py`: 空间结构测试
- `test_group_projection.py`: 群投影测试
- `test_symmery.py`: 对称性测试
- `test_sparsened_point.py`: 稀疏点测试
- `test_insertion_simple.py`: 插入模块简化测试
- `test_load.py`: 加载测试
- `test_mesonspetrum.py`: 介子谱测试
- `test_perambulator_mpi.py`: MPI 传播子测试
- `test_perambulator_phase3.py`: Phase 3 传播子测试
- `test_displacement_elemental.py`: 位移基元测试
- `test_simplify.py`: 简化测试
- `test_gauge_link.py`: 规范链测试

#### 集成测试 (`test/`)
- `test_current_contraction.py`: 流算符收缩（所有四类传播子）
- `test_current_vertex.py`: 流顶点展开测试
- `test_v2p_p2v_symmetry.py`: V2P/P2V 对称性验证
- `test_gauge_links_methods.py`: 规范链方法测试
- `test_calc_all_comparison.py`: 全面计算对比
- `test_calc_calc_disp_consistency.py`: 位移一致性检查
- `test_calc_diagram_bind.py`: 图绑定测试
- `test_calc_v2v_comparison.py`: V2V 对比测试
- `test_detailed_comparison.py`: 详细对比测试
- `test_disp_vs_product.py`: 位移与乘积对比
- `test_einsum_analysis.py`: einsum 分析
- `test_step_by_step.py`: 逐步测试
- `t2pp_skeleton.py`: 两粒子骨架
- `profile_vertex_map_timing_gap.py`: 性能分析

### 测试覆盖率

| 模块 | 覆盖率 | 说明 |
|------|--------|------|
| `preset.py` | 高 | 核心数据类 |
| `generator/` | 中 | 需要实际数据 |
| `quark_diagram.py` | 高 | 算法测试 |
| `symmetry/` | 中 | 群论验证 |

### 运行测试

```bash
# 快速测试（跳过 GPU 测试）
pytest tests/ -m "not gpu"

# 完整测试
pytest tests/ -v

# 特定模块
pytest tests/test_perambulator.py -v
```

### 测试数据

测试数据位于 `tests/` 目录：
- 小格点（4³×8）用于快速测试
- 示例规范场、本征矢量文件

---

## 🚀 部署与运维

### 生产环境建议

#### 硬件要求
- **CPU**: 多核处理器（推荐 16+ 核心）
- **内存**: 64GB+ RAM（取决于格点大小）
- **GPU**: NVIDIA GPU（推荐 V100/A100）
- **存储**: SSD 推荐（I/O 密集）

#### 软件环境
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -e .

# 配置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
```

### 性能优化

#### 1. 内存优化
- 使用时间片格式
- 及时释放大对象
- 使用内存映射

#### 2. 计算优化
- 启用 GPU 加速
- 使用 MPI 并行
- 批处理多个配置

#### 3. I/O 优化
- 使用 SSD 存储
- 并行读取文件
- 压缩存储格式

### 监控与日志

```python
import logging
from lattice import log_gpu_memory

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 记录 GPU 内存
log_gpu_memory("correlator_compute")  # tag 用于标识日志来源
```

---

## ⚠️ 已知问题与改进方向

### 已知问题

1. **文档不足**
   - ❌ 缺少完整的 API 文档
   - ❌ 示例代码不够丰富
   - ✅ 已有传播子理论与使用文档（doc/propagator_theory_and_usage.md）

2. **测试覆盖**
   - ⚠️ 部分模块测试覆盖不足
   - ⚠️ GPU 测试需要特殊环境
   - ✅ 核心功能有测试

3. **性能优化**
   - ⚠️ 大规模并行计算优化空间
   - ⚠️ 内存使用可进一步优化
   - ✅ 已有 GPU 加速

4. **代码质量**
   - ⚠️ 部分函数过长
   - ⚠️ 类型注解不完整
   - ✅ 代码结构清晰

### 改进方向

#### 短期（1-3 个月）
1. ✅ 完善文档（本文档是第一步）
2. 增加更多使用示例
3. 补充单元测试
4. 添加类型注解

#### 中期（3-6 个月）
1. 性能优化
2. 增加更多物理算符
3. 改进错误处理
4. 增强可视化工具

#### 长期（6-12 个月）
1. 重构核心模块
2. 支持新的格点格式
3. 集成机器学习工具
4. 开发 GUI 界面

---

## 📖 参考资料

### 内部文档
- [数据形状说明](doc/README.md)
- [传播子理论与使用](doc/propagator_theory_and_usage.md)
- [局域混合理论](doc/localized_blending/)
- [OpenSpec 变更记录](openspec/changes/)

### 示例代码
- [两点函数生成](example/gen_twopt.py)
- [密度传播子](example/gen_density_peram.py)
- [两粒子关联](example/gen_two_particle_corr.py)

### 外部资源
- [QUDA 文档](https://github.com/lattice/quda)
- [SymPy 文档](https://docs.sympy.org/)
- [格点 QCD 教程](https://github.com/lattice/latticeguild)

---

## 🤝 贡献指南

### 如何贡献
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码审查标准
- ✅ 代码风格一致
- ✅ 添加了文档字符串
- ✅ 包含测试用例
- ✅ 通过所有现有测试
- ✅ 性能无明显退化

---

## 📞 支持与联系

### 问题反馈
- GitHub Issues: [项目地址]
- 邮件: [维护者邮箱]

### 社区
- 用户讨论组: [链接]
- 开发者邮件列表: [链接]

---

## 📜 许可证

[待添加许可证信息]

---

## 📊 文档元数据

- **文档版本**: 1.0
- **生成日期**: 2026-06-01
- **生成工具**: Claude Code (GLM-5)
- **项目状态**: 活跃开发中
- **最后更新**: 2026-06-01

---

**注**: 本文档基于项目当前状态（Git commit: b4c7933）生成，可能随项目更新而变化。建议定期更新此文档以保持同步。
