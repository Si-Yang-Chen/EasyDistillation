# PropagatorPSV Classes Usage Guide

这个文档说明如何使用新添加的三个 PSV propagator 类。

## 类概述

### 1. `PropagatorPSV` (基类)

这是所有 PSV propagator 类的基类，定义了基本接口。

**参数:**
- `elem`: FileMetaData - 文件元数据
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量

**数据形状说明:**
PSV propagator 代表从点源到本征向量的传播子：
- 完整形状: `[Lt, Ns, Ns, Np, Ne]`
  - `Lt`: 时间维度
  - `Ns`: 旋量维度 (通常是4)
  - `Np`: 点源数量
  - `Ne`: 本征向量数量
- 简化形状: `[Lt, Np, Ne]` (某个旋量分量或旋量求迹后)

### 2. `PropagatorPSVNpy` (单文件版本)

从单个 `.npy` 文件加载 PSV propagator 数据。

**参数:**
- `prefix`: str - 文件路径前缀
- `suffix`: str - 文件后缀 (默认: ".npy")
- `shape`: List[int] - 数据形状
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量
- `dtype`: str - 数据类型 (默认: "<c16")

**使用示例:**

```python
from lattice import PropagatorPSVNpy

# 示例 1: 完整的旋量结构 [Lt, Ns, Ns, Np, Ne]
psv = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psv.npy",
    shape=[72, 4, 4, 216, 70],
    Np=216,
    Ne=70,
    dtype="<c16"
)

# 加载配置 "1000"
data = psv.load("1000")  # 从 /path/to/data/cfg_1000.psv.npy 加载
print(data.shape)  # (72, 4, 4, 216, 70)

# 示例 2: 简化形状 [Lt, Np, Ne] (例如，已选择特定旋量分量)
psv_simple = PropagatorPSVNpy(
    prefix="/path/to/data/psv_",
    suffix=".npy",
    shape=[72, 216, 70],
    Np=216,
    Ne=70
)
data_simple = psv_simple.load("cfg_1000")
print(data_simple.shape)  # (72, 216, 70)
```

### 3. `PropagatorPSVTimeslicesNpy` (时间片分离版本)

从按时间片分离保存的多个 `.npy` 文件加载 PSV propagator 数据。

这个类适用于每个源时间片单独保存为一个文件的情况，类似于 `PerambulatorTimeslicesNpy`。

**文件命名规范:**
```
{prefix}{cfg}.t{t_src:03d}{suffix}
```

**示例文件名:**
```
/path/to/data/cfg_1000.t000.npy
/path/to/data/cfg_1000.t001.npy
/path/to/data/cfg_1000.t002.npy
...
/path/to/data/cfg_1000.t071.npy
```

**参数:**
- `prefix`: str - 文件路径前缀 (包括目录和配置前缀)
- `suffix`: str - 文件后缀 (默认: ".npy")
- `shape`: List[int] - **完整** propagator 的形状 (包含所有时间片)
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量
- `dtype`: str - 数据类型 (默认: "<c16")

**使用示例:**

```python
from lattice import PropagatorPSVTimeslicesNpy

# 示例 1: 完整的旋量结构，每个源时间保存一个文件
psv_timeslices = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 4, 4, 216, 70],  # 完整形状
    Np=216,
    Ne=70,
    dtype="<c16"
)

# 加载配置 "1000"
# 会自动加载 cfg_1000.t000.npy, cfg_1000.t001.npy, ..., cfg_1000.t071.npy
data = psv_timeslices.load("1000")
print(data.shape)  # (72, 4, 4, 216, 70)

# 示例 2: 简化形状，时间片分离保存
psv_simple_timeslices = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/psv_cfg_",
    suffix=".npy",
    shape=[72, 216, 70],
    Np=216,
    Ne=70
)
data_simple = psv_simple_timeslices.load("1000")
print(data_simple.shape)  # (72, 216, 70)
```

## 实际应用场景

### 场景 1: 单文件保存 (PropagatorPSVNpy)

适用于：
- 整个 PSV propagator 保存在一个文件中
- 文件体积较小，可以一次性加载
- 不需要按时间片分别处理

### 场景 2: 时间片分离保存 (PropagatorPSVTimeslicesNpy)

适用于：
- PSV propagator 按源时间片分别保存
- 文件较大，需要分片存储和加载
- 需要并行生成不同源时间的 propagator
- 与 `2.gen_propagator.py` 中的保存方式一致

**示例：结合生成和使用**

生成 PSV (从 `2.gen_propagator.py` 修改)：
```python
import numpy as np
from lattice import PerambulatorGenerator

# 生成 PSV propagator
perambulator = PerambulatorGenerator(...)
for cfg in cfg_list:
    perambulator.load(cfg)
    for t_src in range(Lt):
        VSV, PSV, VSP, PSP = perambulator.calc(t_src)
        
        # 保存为时间片分离的文件
        peramb_PSV = np.roll(PSV.get(), -t_src, 0)
        np.save(f"{save_dir}/cfg_{cfg}.t{t_src:03d}.npy", peramb_PSV)
```

使用 PSV (在收缩计算中)：
```python
from lattice import PropagatorPSVTimeslicesNpy

# 加载时间片分离的 PSV
psv_propagator = PropagatorPSVTimeslicesNpy(
    prefix=f"{save_dir}/cfg_",
    suffix=".npy",
    shape=[Lt, 4, 4, Np, Ne],
    Np=Np,
    Ne=Ne
)

# 使用
for cfg in cfg_list:
    psv_data = psv_propagator.load(cfg)
    # 进行收缩计算...
```

## 与 Perambulator 类的对比

| 特性 | Perambulator | PropagatorPSV |
|------|-------------|---------------|
| 形状 | `[Lt, Lt, Ns, Ns, Ne, Ne]` 或简化 `[Lt, Ne, Ne]` | `[Lt, Ns, Ns, Np, Ne]` 或简化 `[Lt, Np, Ne]` |
| 含义 | 本征向量到本征向量 (V→V) | 点源到本征向量 (P→V) 或本征向量到点源 (V→P) |
| 存储位置 | `Perambulator.Np` | `PropagatorPSV.Np`, `PropagatorPSV.Ne` |
| 典型用途 | 强子两点函数 | 流算符插入、三点函数 |

## 注意事项

1. **形状约定**: 根据实际数据格式选择合适的形状
   - 如果包含完整的 Dirac 旋量结构: `[Lt, Ns, Ns, Np, Ne]` (Ns=4)
   - 如果已经进行了旋量收缩或选择: `[Lt, Np, Ne]`

2. **数据类型**: 
   - `<c16` 表示小端序复数双精度 (complex128)
   - `<c8` 表示小端序复数单精度 (complex64)

3. **文件命名**: `PropagatorPSVTimeslicesNpy` 要求严格的文件命名格式：
   - 时间片编号必须是3位数字，例如 `t000`, `t001`, ..., `t071`

4. **内存考虑**: 
   - 完整形状 `[72, 4, 4, 216, 70]` 约占用 1.4 GB (complex128)
   - 简化形状 `[72, 216, 70]` 约占用 87 MB (complex128)

## 导入方式

```python
# 从 lattice 包导入
from lattice import PropagatorPSV, PropagatorPSVNpy, PropagatorPSVTimeslicesNpy

# 或者直接从 preset 模块导入
from lattice.preset import PropagatorPSV, PropagatorPSVNpy, PropagatorPSVTimeslicesNpy
```

