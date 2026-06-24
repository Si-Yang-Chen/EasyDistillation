# 矢量流两点关联函数计算流程分析

基于 `4.contraction.py` 脚本的完整工作流程分析。

> **相关文档**
> - 数据加载与传播子形状：[doc/README.md](doc/README.md)、[doc/propagator_theory_and_usage.md](doc/propagator_theory_and_usage.md)
> - Localized Blending 理论与实现：[doc/localized_blending/localized_blending.md](doc/localized_blending/localized_blending.md)
> - 传统 distillation 工作流：[docs/DISTILLATION_WORKFLOW.md](docs/DISTILLATION_WORKFLOW.md)

## 一、总体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          数据准备阶段                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. 规范场 GaugeField (U_μ)         →  GaugeFieldIldg                       │
│  2. 本征向量 Eigenvector (ξ_i)      →  EigenvectorNpy                       │
│  3. 稀疏化点 PointSource (η_x,a)    →  PointSourceNpy                       │
│  4. Overlap Matrix (M_{xi,a})       →  OverlapMatrixNpy                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                          中间数据生成                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  5. 传播子 VSV (Perambulator)        →  PerambulatorTimeslicesNpy           │
│  6. 传播子 PSV (PropagatorPSV)       →  PropagatorPSVTimeslicesNpy          │
│  7. 算符矩阵元 Elemental (V2V)       →  ElementalNpy                         │
│  8. 流算符矩阵元 P2V                 →  CurrentElementalP2V                  │
│  9. 流算符矩阵元 P2P                 →  CurrentElementalP2P                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                          收缩计算阶段                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  10. 构建算符 Insertion/Operator     →  InsertionGaugeLink, Operator        │
│  11. 构建顶点 Meson/Current          →  Meson, Current                      │
│  12. 构建传播子对象                   →  PropagatorWithCurrent              │
│  13. 构建夸克图                       →  QuarkDiagram                       │
│  14. 执行收缩计算                     →  compute_diagrams_multitime        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、详细计算步骤

### 步骤 1: 加载规范场 (GaugeField)

**数据来源**: ILDG 格式的规范场配置文件

**加载类**: `GaugeFieldIldg`

**代码示例**:
```python
from lattice import GaugeFieldIldg

gauge_field = GaugeFieldIldg(
    prefix="/path/to/configurations/cfg_",
    suffix=".lime",
    shape=[Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)
gauge_field.load("10000")  # 加载 cfg_10000.lime
```

**数据形状**: `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]` (complex128)

**依赖**: 无

**测试覆盖**:
- ✅ `tests/test_load.py` - 基本加载测试
- ⚠️ 需要GPU/CUDA进行规范场操作

---

### 步骤 2: 加载本征向量 (Eigenvector)

**数据来源**: Laplace算符本征向量，由GPU计算生成

**加载类**: `EigenvectorNpy`

**代码示例**:
```python
from lattice import EigenvectorNpy

eigenvector = EigenvectorNpy(
    prefix="/path/to/eigenvectors/",
    suffix=".npy",
    shape=[Lt, Ne, Lz, Ly, Lx, Nc],
    Ne=Ne
)
eigenvector.load("10000")
```

**数据形状**: `[Lt, Ne, Lz, Ly, Lx, Nc]` (complex128)

**依赖**: 需要先计算 Laplace 算符本征值/本征向量

**测试覆盖**:
- ❌ `tests/test_eigenvector.py` - 需要 GPU/CUDA
- ⚠️ GPU依赖测试，需手动运行

---

### 步骤 3: 加载/生成稀疏化点 (PointSource)

**数据来源**: 空间点采样位置

**加载类**: `PointSourceNpy`

**生成函数**: `generate_sparsened_points()`

**代码示例**:
```python
from lattice import PointSourceNpy
from lattice.generator.sparsened_point import generate_sparsened_points

# 生成稀疏化点
points = generate_sparsened_points(
    latt_size=[L, L, L, T],
    num_points=Np,
    seed=42
)  # shape: [Np, Lt, 3]

# 或加载已有数据
point_source = PointSourceNpy(
    prefix="/path/to/points/",
    suffix=".npy",
    shape=[Np, Lt, 3],
    Np=Np
)
```

**数据形状**: `[Np, Lt, 3]` 或 `[Np, 3]`

**测试覆盖**:
- ✅ `tests/test_sparsened_point.py` - 完整测试覆盖 (35 tests)

---

### 步骤 4: 加载 Overlap Matrix

**数据来源**: 本征向量与点源的重叠矩阵

**数学定义**: $M_{xi,a} = \langle\eta_{x,a}|\xi_i\rangle$

**加载类**: `OverlapMatrixNpy`

**代码示例**:
```python
from lattice import OverlapMatrixNpy

overlap_matrix = OverlapMatrixNpy(
    prefix="/path/to/overlap/",
    suffix=".overlap_matrix.npy",
    shape=[Lt, Ne, Np, Nc],
    Ne=Ne,
    Np=Np
)
overlap_matrix.load("10000")
```

**数据形状**: `[Lt, Ne, Np, Nc]` (complex128)

**测试覆盖**:
- ⚠️ 无专门测试，需添加

---

### 步骤 5: 加载传播子 (Perambulator/Propagator)

**数据来源**: 由 `PerambulatorGenerator` 生成的四种传播子

#### 5.1 VSV (Perambulator)

**加载类**: `PerambulatorTimeslicesNpy`

**数学定义**: $\tau_{i,j} = \langle\xi_i|S|\xi_j\rangle$

**数据形状**: `[Lt, Ne_snk, Ne_src]` 或 `[Lt, Lt, Ns, Ns, Ne_snk, Ne_src]`

```python
from lattice import PerambulatorTimeslicesNpy

vsv = PerambulatorTimeslicesNpy(
    prefix="/path/to/vsv/",
    suffix=".t???.npy",
    shape=[Lt, Ne, Ne],
    Ne=Ne
)
```

#### 5.2 PSV (PropagatorPSV)

**加载类**: `PropagatorPSVTimeslicesNpy`

**数学定义**: $S_{xa,j} = \langle\eta_{x,a}|S|\xi_j\rangle$

**数据形状**: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` 或 `[Lt, Ne, Np, Nc]`

```python
from lattice import PropagatorPSVTimeslicesNpy

psv = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/psv/",
    suffix=".t???.npy",
    shape=[Lt, Ne, Np, Nc],
    Np=Np,
    Ne=Ne
)
```

**测试覆盖**:
- ❌ `tests/test_perambulator.py` - 需要 PyQuda (GPU)
- ❌ `tests/test_perambulator_mpi.py` - 需要 PyQuda (GPU)
- ❌ `tests/test_perambulator_phase3.py` - 需要 PyQuda (GPU)
- ⚠️ GPU依赖测试，需手动运行

---

### 步骤 6: 加载算符矩阵元 (Elemental)

**数据来源**: 由 `ElementalGenerator` 生成的介子算符矩阵元

**加载类**: `ElementalNpy`

**代码示例**:
```python
from lattice import ElementalNpy

elemental = ElementalNpy(
    prefix="/path/to/elemental/",
    suffix=".npy",
    shape=[Lt, Ne, Ne],
    Ne=Ne
)
elemental.load("10000")
```

**数据形状**: `[Lt, Ne, Ne]` 或 `[num_disp, num_mom, Lt, Ne, Ne]`

**测试覆盖**:
- ✅ `tests/test_elemental.py` - 基本测试
- ⚠️ 生成需要 GPU

---

### 步骤 7: 加载流算符矩阵元 (CurrentElemental)

**数据来源**: 由 `CurrentElementalGenerator` 生成

**数学定义**:
- V2P: $O_{i,xa} = \langle\xi_i|O|\eta_{x,a}\rangle$
- P2V: $O_{xa,j} = \langle\eta_{x,a}|O|\xi_j\rangle$
- P2P: $O_{xa,yb} = \langle\eta_{x,a}|O|\eta_{y,b}\rangle$

**加载类**: `CurrentElementalV2P`, `CurrentElementalP2V`, `CurrentElementalP2P`

**代码示例**:
```python
from lattice import CurrentElementalV2P, CurrentElementalP2V, CurrentElementalP2P

v2p_data = CurrentElementalV2P(
    prefix="/path/to/current_elemental/",
    suffix="_v2p.npy",
    shape=[Lt, num_disp, Ne, Np, Nc],
    Ne=Ne,
    Np=Np
)

p2v_data = CurrentElementalP2V(
    prefix="/path/to/current_elemental/",
    suffix="_p2v.npy",
    shape=[Lt, num_disp, Np, Nc, Ne],
    Ne=Ne,
    Np=Np
)

p2p_data = CurrentElementalP2P(
    prefix="/path/to/current_elemental/",
    suffix=None  # HDF5 format
)
```

**测试覆盖**:
- ⚠️ 无专门测试，需添加

---

### 步骤 8: 构建算符 (Insertion/Operator)

**功能**: 定义介子和流算符的物理结构

**类**: `Insertion`, `InsertionGaugeLink`, `Operator`

**代码示例**:
```python
from lattice.insertion import (
    Insertion, InsertionGaugeLink, Operator,
    GammaName, DerivativeName, ProjectionName
)
from lattice.insertion.mom_dict import momDict_test
from lattice.symmetry.hardcoded_rep import gauge_link

# 定义介子算符 (ρ 介子)
ins_meson = InsertionGaugeLink(
    GammaName.RHO,       # γ 矩阵
    "A_1g+",             # 表示名
    0,                    # 导数阶数
    "T_1",                # 目标表示
    momDict_test,         # 动量字典
    gauge_link            # 规范链对象
)

# 定义流算符 (矢量流)
ins_current = InsertionGaugeLink(
    GammaName.A0,        # γ 矩阵
    "T_1u-",             # 表示名
    0,                    # 导数阶数
    "T_1",                # 目标表示
    momDict_test,
    gauge_link
)

# 构建 Operator 对象
op_meson = Operator("rho", [ins_meson[0](0, 0, 0)], [1])
op_current = Operator("v", [ins_current[0](0, 0, 0)], [1])
```

**测试覆盖**:
- ✅ `tests/test_insertion_simple.py` - 模块结构测试
- ✅ `tests/test_gauge_link.py` - 规范链测试 (28 tests)
- ✅ `tests/test_gamma.py` - γ 矩阵测试
- ⚠️ `tests/test_displacement_elemental.py` - 需要 GPU

---

### 步骤 9: 构建顶点对象 (Meson/Current)

**功能**: 封装算符数据和加载逻辑

**类**: `Meson`, `Current`

**代码示例**:
```python
from lattice.quark_diagram import Meson, Current

# 介子顶点 (无流)
meson = Meson(elemental, op_meson, source=False)

# 流顶点 (有流)
current = Current(
    elemental,
    op_current,
    source=True,           # 是否为源端
    p2v_data=p2v_data,     # P2V 数据
    p2p_data=p2p_data,     # P2P 数据
    debug=True
)

# 加载数据
meson.load("10000", usedNe=20)
current.load("10000", usedNe=20, usedNp=100)
```

**测试覆盖**:
- ⚠️ 无专门测试，需添加

---

### 步骤 10: 构建传播子对象 (PropagatorWithCurrent)

**功能**: 封装传播子数据和高模投影计算

**类**: `PropagatorWithCurrent`

**代码示例**:
```python
from lattice.quark_diagram import PropagatorWithCurrent

propagator = PropagatorWithCurrent(
    vsv=vsv,               # Perambulator (V2V)
    vsp=None,              # PropagatorVSP (可选)
    psv=psv,               # PropagatorPSV (P2V)
    psp=None,              # PropagatorPSP (可选)
    overlap_matrix=overlap_matrix,
    Lt=Lt,
    debug=True
)

# 加载数据
propagator.load("10000", usedNe=20)
```

**高模投影公式** (自动调用):
- `get_VSP_highmode()`: $\tilde{S}_{i,xa} = S_{i,xa} - \sum_j S_{i,j} M_{jx,a}^*$
- `get_PSV_highmode()`: $\tilde{S}_{xa,i} = S_{xa,i} - \sum_j M_{xj,a} S_{j,i}$
- `get_PSP_highmode()`: 双端投影

**测试覆盖**:
- ✅ `tests/test_sampling_weight.py` - 高模投影公式验证
- ⚠️ 集成测试需要 GPU

---

### 步骤 11: 构建夸克图 (QuarkDiagram)

**功能**: 分析夸克收缩拓扑并展开

**类**: `QuarkDiagram`

**代码示例**:
```python
from lattice.quark_diagram import QuarkDiagram

# 两点函数: Meson - Propagator - Current - Propagator - Meson
# 邻接矩阵: [[0, 1], [1, 0]]
#   - vertex 0 → vertex 1: propagator 0
#   - vertex 1 → vertex 0: propagator 1
# vertex_list: [0, 1] 表示 vertex[0] 是普通顶点，vertex[1] 是流顶点

diagram = QuarkDiagram(
    adjacency_matrix=[[0, 1], [1, 0]],
    vertex_list=[0, 1],    # 0=普通顶点, 1=流顶点
    debug=True,
    usedNp=Np,
    L=L
)

# 自动展开为 16 种状态组合 (4^2)
# 每种组合对应不同的传播子类型 (VSV/VSP/PSV/PSP)
```

**展开过程**:
1. **状态展开**: 每个流顶点有 4 种状态 (V2V/V2P/P2V/P2P)
2. **场景展开**: 考虑点重合情况，生成场景权重
3. **自动合并**: 相同结构的 sub-diagram 合并

**测试覆盖**:
- ✅ `tests/test_diagram.py` - Diagram 简化测试
- ✅ `tests/test_sampling_weight.py` - 场景枚举测试
- ⚠️ 流顶点展开测试不完整

---

### 步骤 12: 执行收缩计算 (compute_diagrams_multitime)

**功能**: 执行最终的关联函数收缩计算

**函数**: `compute_diagrams_multitime()`

**代码示例**:
```python
from lattice.quark_diagram import compute_diagrams_multitime

output = np.zeros((Lt, Lt), dtype=np.complex128)

for t_src in range(Lt):
    # 计算 t_src 时刻源的两点函数
    result = compute_diagrams_multitime(
        [diagram],
        [t_src, np.arange(Lt)],    # [源时间, 所有汇时间]
        [meson, current],          # 顶点对象
        [None, propagator],        # 传播子对象 (None 表示无传播子)
        multitime_shape=True,
        debug=True
    )

    # result.shape: [num_expanded_diagrams, Lt]
    # 对所有展开的图求和
    output[t_src] = np.roll(backend.sum(result, axis=0).get(), -t_src, axis=0)
```

**计算流程**:
1. 自动展开所有 sub-diagram
2. 对每个 sub-diagram:
   - 根据 `propagator_types` 获取正确的传播子数据
   - 根据 `vertex_types` 获取正确的顶点数据
   - 执行 Einstein 求和收缩
   - 应用 `scene_weight`
3. 返回所有 sub-diagram 的结果数组

**测试覆盖**:
- ⚠️ 无完整集成测试
- ⚠️ 需要 GPU 和完整数据流程

---

## 三、测试覆盖分析

### 可运行测试 (197 tests, 193 passed, 4 failed)

| 测试文件 | 测试数 | 状态 | 覆盖内容 |
|---------|-------|------|---------|
| `test_diagram.py` | 10 | ✅ | Diagram 简化 |
| `test_flavor_structure.py` | 17 | ⚠️ 1失败 | 味道结构 |
| `test_gauge_link.py` | 28 | ✅ | 规范链 |
| `test_group_projection.py` | 7 | ✅ | 群投影 |
| `test_insertion_simple.py` | 17 | ⚠️ 2失败 | 算符模块 |
| `test_sampling_weight.py` | 42 | ✅ | 采样权重 |
| `test_simplify.py` | 8 | ✅ | 表达式简化 |
| `test_sparsened_point.py` | 35 | ✅ | 稀疏化点生成 |
| `test_spatial_structure.py` | 7 | ✅ | 空间结构 |
| `test_symmery.py` | 18 | ✅ | 对称群 |

### GPU 依赖测试 (需手动运行)

| 测试文件 | 状态 | 依赖 |
|---------|------|------|
| `test_eigenvector.py` | ❌ | CUDA/cuPy |
| `test_perambulator.py` | ❌ | PyQuda |
| `test_perambulator_mpi.py` | ❌ | PyQuda + MPI |
| `test_perambulator_phase3.py` | ❌ | PyQuda |
| `test_displacement_elemental.py` | ⚠️ | GPU |

### 导入错误测试 (需修复)

| 测试文件 | 错误 |
|---------|------|
| `test_mesonspetrum.py` | 缺少 `OperatorDisplacement` |
| `test_quark_contract.py` | 缺少 `quark_contract` 函数 |

---

## 四、缺失测试清单

### 高优先级

1. **`PropagatorWithCurrent` 类测试**
   - 初始化测试
   - `get_VSP_highmode()` 测试
   - `get_PSV_highmode()` 测试
   - `get_PSP_highmode()` 测试
   - 高模投影与未投影对比测试

2. **`Current` 类测试**
   - 初始化测试
   - `get()`, `get_v2p()`, `get_p2v()`, `get_p2p()` 测试
   - usedNe/usedNp 切片测试

3. **`Meson` 类测试**
   - 初始化测试
   - `get()` 测试
   - usedNe 切片测试

4. **`compute_diagrams_multitime()` 测试**
   - 单时间片测试
   - 多时间片测试
   - 自动展开测试
   - scene_weight 应用测试

### 中优先级

5. **`CurrentElemental` 类测试**
   - V2P/P2V/P2P 数据加载测试
   - 形状验证测试

6. **`OverlapMatrixNpy` 类测试**
   - 加载测试
   - 形状验证测试

7. **`QuarkDiagram` 流顶点展开测试**
   - 状态展开测试
   - 场景生成测试
   - 权重计算测试

### 低优先级

8. **数据加载类边界测试**
   - 文件不存在处理
   - 形状不匹配处理
   - 内存管理测试

---

## 五、完整工作流程脚本

```python
#!/usr/bin/env python3
"""
矢量流两点关联函数完整计算流程

数据准备 → 收缩计算 → 结果输出
"""

import os
import numpy as np
from lattice import set_backend, get_backend
set_backend("cupy")
backend = get_backend()

# ============================================================
# 1. 参数设置
# ============================================================
L = 24
T = 72
latt_size = [L, L, L, T]
Lx, Ly, Lz, Lt = latt_size
Np = 6**3  # 216 采样点
Ne = 128  # 本征向量数量
num_nabla = 1  # 导数阶数
num_momentum = 0  # 动量级别

# ============================================================
# 2. 加载输入数据
# ============================================================
from lattice import (
    GaugeFieldIldg,
    EigenvectorNpy,
    PointSourceNpy,
    OverlapMatrixNpy,
    PerambulatorTimeslicesNpy,
    PropagatorPSVTimeslicesNpy,
    ElementalNpy,
    CurrentElementalP2V,
    CurrentElementalP2P,
)

gauge_field = GaugeFieldIldg(
    "/path/to/cfg_",
    ".lime",
    [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)

eigenvector = EigenvectorNpy(
    "/path/to/eigen/",
    ".npy",
    [Lt, Ne, Lz, Ly, Lx, Nc],
    Ne
)

point_source = PointSourceNpy(
    "/path/to/points/",
    ".npy",
    [Np, Lt, 3],
    Np
)

overlap_matrix = OverlapMatrixNpy(
    "/path/to/overlap/",
    ".overlap_matrix.npy",
    [Lt, Ne, Np, Nc],
    Ne, Np
)

# ============================================================
# 3. 加载传播子
# ============================================================
vsv = PerambulatorTimeslicesNpy(
    "/path/to/vsv/",
    ".t???.npy",
    [Lt, Ne, Ne],
    Ne
)

psv = PropagatorPSVTimeslicesNpy(
    "/path/to/psv/",
    ".t???.npy",
    [Lt, Ne, Np, Nc],
    Np, Ne
)

# ============================================================
# 4. 加载算符矩阵元
# ============================================================
elemental = ElementalNpy(
    "/path/to/elemental/",
    ".npy",
    [Lt, Ne, Ne],
    Ne
)

p2v_data = CurrentElementalP2V(
    "/path/to/current_elemental/",
    "_p2v.npy",
    [Lt, num_disp, Np, Nc, Ne],
    Ne, Np
)

p2p_data = CurrentElementalP2P(
    "/path/to/current_elemental/",
    None  # HDF5
)

# ============================================================
# 5. 构建算符和顶点
# ============================================================
from lattice.insertion import InsertionGaugeLink, Operator, GammaName
from lattice.insertion.mom_dict import momDict_test
from lattice.symmetry.hardcoded_rep import gauge_link
from lattice.quark_diagram import Meson, Current, QuarkDiagram, PropagatorWithCurrent, compute_diagrams_multitime

# 定义算符
ins_meson = InsertionGaugeLink(
    GammaName.RHO, "A_1g+", 0, "T_1",
    momDict_test, gauge_link
)
ins_current = InsertionGaugeLink(
    GammaName.A0, "T_1u-", 0, "T_1",
    momDict_test, gauge_link
)

op_meson = Operator("rho", [ins_meson[0](0, 0, 0)], [1])
op_current = Operator("v", [ins_current[0](0, 0, 0)], [1])

# 构建顶点对象
meson = Meson(elemental, op_meson, source=False)
current = Current(
    elemental, op_current, source=True,
    p2v_data=p2v_data, p2p_data=p2p_data, debug=True
)

# 构建传播子对象
propagator = PropagatorWithCurrent(
    vsv=vsv, psv=psv, overlap_matrix=overlap_matrix,
    Lt=Lt, debug=True
)

# 构建夸克图
diagram = QuarkDiagram(
    [[0, 1], [1, 0]],
    vertex_list=[0, 1],
    debug=True, usedNp=Np, L=L
)

# ============================================================
# 6. 执行收缩计算
# ============================================================
output = np.zeros((Lt, Lt), dtype=np.complex128)
cfg = "10000"

with propagator, meson, current:
    propagator.load(cfg, usedNe=20)
    meson.load(cfg, usedNe=20)
    current.load(cfg, usedNe=20, usedNp=100)

    for t_src in range(Lt):
        result = compute_diagrams_multitime(
            [diagram],
            [t_src, np.arange(Lt)],
            [meson, current],
            [None, propagator],
            multitime_shape=True,
            debug=True
        )
        output[t_src] = np.roll(backend.sum(result, axis=0).get(), -t_src, axis=0)

# ============================================================
# 7. 输出结果
# ============================================================
np.save("/path/to/output/correlator.npy", output)
print("Correlator computed successfully!")
print(f"Shape: {output.shape}")
```

---

## 六、关键依赖关系

```
GaugeField ──┐
             ├──→ PerambulatorGenerator ──→ VSV, PSV, VSP, PSP
Eigenvector ─┘

Eigenvector ──┐
              ├──→ OverlapMatrix
PointSource ──┘

GaugeField ──┐
             ├──→ ElementalGenerator ──→ Elemental (V2V)
Eigenvector ─┘

GaugeField ──┐
             ├──→ CurrentElementalGenerator ──→ V2P, P2V, P2P
Eigenvector ─┤
PointSource ─┘

VSV ─────────┐
PSV ─────────┤
VSP ─────────┼──→ PropagatorWithCurrent
PSP ─────────┤
Overlap ─────┘

Elemental ───┐
Operator ────┼──→ Meson / Current

PropagatorWithCurrent ──┐
Meson / Current ────────┼──→ QuarkDiagram ──→ compute_diagrams_multitime ──→ Correlator
```
