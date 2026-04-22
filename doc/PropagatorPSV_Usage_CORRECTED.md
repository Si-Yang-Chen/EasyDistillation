# PropagatorPSV Classes Usage Guide (Updated)

这个文档说明如何使用新添加的三个 PSV propagator 类。

**重要更新**: PSV propagator 的正确形状包含颜色指标 (Nc)。

## 类概述

### 1. `PropagatorPSV` (基类)

这是所有 PSV propagator 类的基类，定义了基本接口。

**参数:**
- `elem`: FileMetaData - 文件元数据
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量

**数据形状说明:**
PSV propagator 代表从点源到本征向量的传播子：

**单文件版本完整形状**: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- 第一个 `Lt`: sink 时间
- 第二个 `Lt`: source 时间  
- 第一个 `Ns`: sink 旋量指标 (= 4)
- 第二个 `Ns`: source 旋量指标 (= 4)
- `Np`: 点源数量
- `Nc`: 颜色指标 (= 3)
- `Ne`: 本征向量数量

**时间片版本形状**: `[Lt, Ns, Ns, Np, Nc, Ne]`
- 每个文件对应一个 source 时间
- `Lt`: sink 时间（相对于 source 时间）
- 其余维度同上

### 2. `PropagatorPSVNpy` (单文件版本)

从单个 `.npy` 文件加载完整的 PSV propagator 数据。

**参数:**
- `prefix`: str - 文件路径前缀
- `suffix`: str - 文件后缀 (默认: ".npy")
- `shape`: List[int] - 数据形状: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量
- `dtype`: str - 数据类型 (默认: "<c16")

**使用示例:**

```python
from lattice import PropagatorPSVNpy

# 完整形状 [Lt, Lt, Ns, Ns, Np, Nc, Ne]
psv = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psv.npy",
    shape=[72, 72, 4, 4, 216, 3, 70],
    Np=216,
    Ne=70,
    dtype="<c16"
)

# 加载配置 "1000"
data = psv.load("1000")  # 从 /path/to/data/cfg_1000.psv.npy 加载
print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
print(f"内存占用: ~{data.nbytes / 1e9:.2f} GB")
```

### 3. `PropagatorPSVTimeslicesNpy` (时间片分离版本)

从按时间片分离保存的多个 `.npy` 文件加载 PSV propagator 数据。

这个类适用于每个源时间片单独保存为一个文件的情况。

**文件命名规范:**
```
{prefix}{cfg}.t{t_src:03d}{suffix}
```

**示例文件名:**
```
/path/to/data/cfg_1000.t000.npy  # shape: [Lt, Ns, Ns, Np, Nc, Ne]
/path/to/data/cfg_1000.t001.npy  # shape: [Lt, Ns, Ns, Np, Nc, Ne]
/path/to/data/cfg_1000.t002.npy  # shape: [Lt, Ns, Ns, Np, Nc, Ne]
...
/path/to/data/cfg_1000.t071.npy  # shape: [Lt, Ns, Ns, Np, Nc, Ne]
```

**参数:**
- `prefix`: str - 文件路径前缀 (包括目录和配置前缀)
- `suffix`: str - 文件后缀 (默认: ".npy")
- `shape`: List[int] - **完整** propagator 的形状: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- `Np`: int - 点源数量
- `Ne`: int - 本征向量数量
- `dtype`: str - 数据类型 (默认: "<c16")

**使用示例:**

```python
from lattice import PropagatorPSVTimeslicesNpy, Ns, Nc

# 时间片分离版本
Lt, Np, Ne = 72, 216, 70

psv_timeslices = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],  # 完整形状: [72, 72, 4, 4, 216, 3, 70]
    Np=Np,
    Ne=Ne,
    dtype="<c16"
)

# 加载配置 "1000"
# 会自动加载 cfg_1000.t000.npy, cfg_1000.t001.npy, ..., cfg_1000.t071.npy
# 每个文件形状: [72, 4, 4, 216, 3, 70]
data = psv_timeslices.load("1000")
print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
```

## 实际应用场景

### 场景 1: 单文件保存 (PropagatorPSVNpy)

适用于：
- 整个 PSV propagator 保存在一个文件中
- 可以承受大文件的存储和加载
- 简单的文件管理

**内存占用估算**:
- 形状 `[72, 72, 4, 4, 216, 3, 70]`
- 大小: 72 × 72 × 4 × 4 × 216 × 3 × 70 × 16 bytes ≈ **148 GB** ⚠️ 非常大！

### 场景 2: 时间片分离保存 (PropagatorPSVTimeslicesNpy) ✅ 推荐

适用于：
- PSV propagator 按源时间片分别保存
- 文件较大，需要分片存储和加载
- 需要并行生成不同源时间的 propagator
- 与 `2.gen_propagator.py` 中的保存方式一致

**内存占用估算** (每个时间片文件):
- 形状 `[72, 4, 4, 216, 3, 70]`
- 大小: 72 × 4 × 4 × 216 × 3 × 70 × 16 bytes ≈ **2.1 GB** per file
- 总共 72 个文件 ≈ 150 GB

**示例：结合生成和使用**

生成 PSV (从 `2.gen_propagator.py`)：
```python
import numpy as np
from lattice import PerambulatorGenerator

# 生成 PSV propagator
perambulator = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,
    eigenvector_snk=eigenvector,
    point_snk=point_source,  # 指定点源
    ...
)

for cfg in cfg_list:
    perambulator.load(cfg)
    perambulator.stout_smear(20, 0.12)
    
    for t_src in range(Lt):
        VSV, PSV, VSP, PSP = perambulator.calc(t_src)
        
        # PSV.get() 返回形状: [Lt, Ns, Ns, Np, Nc, Ne]
        peramb_PSV = np.roll(PSV.get(), -t_src, 0)
        
        # 保存为时间片分离的文件
        np.save(f"{save_dir}/cfg_{cfg}.t{t_src:03d}.npy", peramb_PSV)
```

使用 PSV (在收缩计算中)：
```python
from lattice import PropagatorPSVTimeslicesNpy, Ns, Nc

# 加载时间片分离的 PSV
psv_propagator = PropagatorPSVTimeslicesNpy(
    prefix=f"{save_dir}/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=Np,
    Ne=Ne
)

# 使用
for cfg in cfg_list:
    psv_data = psv_propagator.load(cfg)
    # psv_data shape: [Lt, Lt, Ns, Ns, Np, Nc, Ne]
    # 进行收缩计算...
    result = compute_diagrams_multitime(
        diagrams,
        time_slices,
        operators,
        [None, vsv_propagator, psv_propagator]
    )
```

## 与 Perambulator 类的对比

| 特性 | Perambulator (V2V) | PropagatorPSV (P2V/V2P) |
|------|-------------------|------------------------|
| **完整形状** | `[Lt, Lt, Ns, Ns, Ne, Ne]` | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` |
| **时间片形状** | `[Lt, Ns, Ns, Ne, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| **含义** | 本征向量到本征向量 | 点源到本征向量（含颜色） |
| **存储位置** | `Perambulator.Ne` | `PropagatorPSV.Np`, `PropagatorPSV.Ne` |
| **典型用途** | 强子两点函数 | 流算符插入、三点函数 |
| **单文件大小** | ~91 GB (70 eigenvectors) | ~148 GB (216 points, 70 eigenvectors) |

## 数据索引说明

### 单文件版本 `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`

```python
PSV[t_snk, t_src, s_snk, s_src, p, c, e]
```

表示：从 `t_src` 时刻的点源 `p` 传播到 `t_snk` 时刻的本征向量 `e`，
包含旋量指标 `s_src`, `s_snk` 和颜色指标 `c`。

### 时间片版本 `[Lt, Ns, Ns, Np, Nc, Ne]`

```python
PSV_t_src[t_snk, s_snk, s_src, p, c, e]
```

表示：从固定 `t_src` 时刻的点源 `p` 传播到 `t_snk` 时刻的本征向量 `e`。

## 注意事项

1. **形状约定**: 
   - 单文件: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` (7 维)
   - 时间片: `[Lt, Ns, Ns, Np, Nc, Ne]` (6 维)
   - **必须包含颜色维度 Nc**

2. **数据类型**: 
   - `<c16` 表示小端序复数双精度 (complex128) - 推荐
   - `<c8` 表示小端序复数单精度 (complex64) - 节省空间但精度降低

3. **文件命名**: `PropagatorPSVTimeslicesNpy` 要求严格的文件命名格式：
   - 时间片编号必须是3位数字，例如 `t000`, `t001`, ..., `t071`

4. **内存考虑**: 
   - 完整形状非常大 (~148 GB)，**强烈推荐使用时间片分离版本**
   - 每个时间片文件约 2.1 GB，更易于管理和并行处理

5. **与 PerambulatorGenerator 的对应**:
   ```python
   VSV, PSV, VSP, PSP = perambulator.calc(t_src)
   # PSV 的形状就是 [Lt, Ns, Ns, Np, Nc, Ne]
   ```

## 导入方式

```python
# 从 lattice 包导入
from lattice import PropagatorPSV, PropagatorPSVNpy, PropagatorPSVTimeslicesNpy
from lattice import Ns, Nc  # 导入常量

# 或者直接从 preset 模块导入
from lattice.preset import PropagatorPSV, PropagatorPSVNpy, PropagatorPSVTimeslicesNpy
from lattice.constant import Ns, Nc
```

## 完整示例

```python
from lattice import PropagatorPSVTimeslicesNpy, Ns, Nc
from lattice.quark_diagram import Current, compute_diagrams_multitime

# 参数设置
L, T = 24, 72
Np, Ne = 216, 70

# 创建 PSV propagator 对象
psv = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/psv_data/cfg_",
    suffix=".npy",
    shape=[T, T, Ns, Ns, Np, Nc, Ne],  # [72, 72, 4, 4, 216, 3, 70]
    Np=Np,
    Ne=Ne
)

# 加载并使用
for cfg in cfg_list:
    psv_data = psv.load(cfg)
    print(f"Loaded PSV data shape: {psv_data.shape}")
    
    # 在收缩计算中使用
    result = compute_diagrams_multitime(
        diagram_structure,
        time_slices,
        [meson, current],
        [None, vsv_perambulator, psv]
    )
```

## 性能提示

1. **使用时间片版本**: 更好的并行化和内存管理
2. **按需加载**: 如果只需要某些时间片，考虑修改加载逻辑
3. **数据压缩**: 考虑使用压缩格式（如 `.npz`）减少磁盘空间
4. **精度权衡**: 如果精度允许，使用 `<c8` (complex64) 可节省50%空间

