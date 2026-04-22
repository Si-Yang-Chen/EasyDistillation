# 完整的 Propagator 类文档

本文档全面介绍项目中所有的 propagator 和 perambulator 类。

## 类概览

| 类型 | 基类 | 单文件类 | 时间片类 | 形状 (单文件) | 形状 (时间片) |
|------|------|---------|---------|--------------|--------------|
| **V2V** | `Perambulator` | `PerambulatorNpy` | `PerambulatorTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Ne, Ne]` | `[Lt, Ns, Ns, Ne, Ne]` |
| **P2V** | `PropagatorPSV` | `PropagatorPSVNpy` | `PropagatorPSVTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| **V2P** | `PropagatorVSP` | `PropagatorVSPNpy` | `PropagatorVSPTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| **P2P** | `PropagatorPSP` | `PropagatorPSPNpy` | `PropagatorPSPTimeslicesNpy` | `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]` | `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` |

## 维度说明

- `Lt`: 时间维度
- `Ns`: Dirac 旋量维度 (= 4)
- `Nc`: 颜色维度 (= 3)
- `Ne`: 本征向量数量
- `Np`: 点源数量
- `Np_src`: 源点数量
- `Np_snk`: 汇点数量

## 详细说明

### 1. V2V (Eigenvector ↔ Eigenvector) - Perambulator

**物理含义**: 本征向量到本征向量的传播

**形状**:
- 单文件: `[Lt, Lt, Ns, Ns, Ne, Ne]`
- 时间片: `[Lt, Ns, Ns, Ne, Ne]` (每个文件对应一个 t_src)

**类**:
```python
from lattice import PerambulatorNpy, PerambulatorTimeslicesNpy

# 单文件版本
vsv = PerambulatorNpy(
    prefix="/path/to/data/cfg_",
    suffix=".vsv.npy",
    shape=[72, 72, 4, 4, 70, 70],
    Ne=70
)

# 时间片版本 (推荐)
vsv = PerambulatorTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 72, 4, 4, 70, 70],
    Ne=70
)
```

**用途**: 
- 强子两点函数
- 介子算符
- 重子算符

**内存占用** (Lt=72, Ne=70, complex128):
- 单文件: ~91 GB ⚠️
- 简化形状 `[Lt, Ne, Ne]`: ~5.6 MB ✅

---

### 2. P2V (Point ↔ Eigenvector) - PropagatorPSV

**物理含义**: 点源到本征向量的传播

**形状**:
- 单文件: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- 时间片: `[Lt, Ns, Ns, Np, Nc, Ne]` (每个文件对应一个 t_src)

**类**:
```python
from lattice import PropagatorPSVNpy, PropagatorPSVTimeslicesNpy, Ns, Nc

# 单文件版本
psv = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psv.npy",
    shape=[72, 72, 4, 4, 216, 3, 70],
    Np=216,
    Ne=70
)

# 时间片版本 (推荐)
psv = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 72, Ns, Ns, 216, Nc, 70],
    Np=216,
    Ne=70
)
```

**用途**: 
- 三点函数（点源插入）
- 流算符矩阵元
- Current 类计算

**内存占用** (Lt=72, Np=216, Ne=70, complex128):
- 单文件: ~148 GB ⚠️
- 时间片单个文件: ~2.1 GB ✅

---

### 3. V2P (Eigenvector ↔ Point) - PropagatorVSP

**物理含义**: 本征向量到点源的传播

**形状**:
- 单文件: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- 时间片: `[Lt, Ns, Ns, Np, Nc, Ne]` (每个文件对应一个 t_src)

**注意**: VSP 和 PSV 具有相同的形状，但代表不同的传播方向。

**类**:
```python
from lattice import PropagatorVSPNpy, PropagatorVSPTimeslicesNpy, Ns, Nc

# 单文件版本
vsp = PropagatorVSPNpy(
    prefix="/path/to/data/cfg_",
    suffix=".vsp.npy",
    shape=[72, 72, 4, 4, 216, 3, 70],
    Np=216,
    Ne=70
)

# 时间片版本 (推荐)
vsp = PropagatorVSPTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 72, Ns, Ns, 216, Nc, 70],
    Np=216,
    Ne=70
)
```

**用途**: 
- 三点函数（本征向量源）
- 逆向传播计算
- 特定类型的矩阵元

**内存占用**: 与 PSV 相同

---

### 4. P2P (Point ↔ Point) - PropagatorPSP

**物理含义**: 点源到点源的传播

**形状**:
- 单文件: `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]`
- 时间片: `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` (每个文件对应一个 t_src)

**类**:
```python
from lattice import PropagatorPSPNpy, PropagatorPSPTimeslicesNpy, Ns, Nc

# 单文件版本
psp = PropagatorPSPNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psp.npy",
    shape=[72, 72, 4, 4, 216, 3, 216],
    Np_snk=216,
    Np_src=216
)

# 时间片版本 (推荐)
psp = PropagatorPSPTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 72, Ns, Ns, 216, Nc, 216],
    Np_snk=216,
    Np_src=216
)
```

**用途**: 
- 点源到点源关联函数
- 特定的三点函数
- 点对点分析

**内存占用** (Lt=72, Np=216, complex128):
- 单文件: ~32 TB ❌ 极其庞大！
- 时间片单个文件: ~450 GB ⚠️ 仍然很大

---

## 数据生成 (从 PerambulatorGenerator)

```python
from lattice import PerambulatorGenerator

generator = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,
    eigenvector_snk=eigenvector,
    point_src=point_source,
    point_snk=point_sink,
    ...
)

for t_src in range(Lt):
    VSV, PSV, VSP, PSP = generator.calc(t_src)
    
    # VSV: [Lt, Ns, Ns, Ne_snk, Ne_src] - Eigenvector to Eigenvector
    # PSV: [Lt, Ns, Ns, Np_snk, Ne_src] - Point to Eigenvector
    # VSP: [Lt, Ns, Ns, Np_snk, Ne_src] - Eigenvector to Point
    # PSP: [Lt, Ns, Ns, Np_snk, Np_src] - Point to Point
    
    # 保存
    np.save(f"{dir_vsv}/cfg_{cfg}.t{t_src:03d}.npy", np.roll(VSV.get(), -t_src, 0))
    np.save(f"{dir_psv}/cfg_{cfg}.t{t_src:03d}.npy", np.roll(PSV.get(), -t_src, 0))
    np.save(f"{dir_vsp}/cfg_{cfg}.t{t_src:03d}.npy", np.roll(VSP.get(), -t_src, 0))
    np.save(f"{dir_psp}/cfg_{cfg}.t{t_src:03d}.npy", np.roll(PSP.get(), -t_src, 0))
```

## 完整使用示例

```python
from lattice import (
    PerambulatorNpy,
    PropagatorPSVTimeslicesNpy,
    PropagatorVSPTimeslicesNpy,
    PropagatorPSPTimeslicesNpy,
    Ns, Nc
)

# 参数
L, T = 24, 72
Np, Ne = 216, 70

# 加载所有类型的 propagators
vsv = PerambulatorNpy(
    prefix=f"{data_dir}/vsv/cfg_",
    suffix=".npy",
    shape=[T, Ne, Ne],  # 简化形状
    Ne=Ne
)

psv = PropagatorPSVTimeslicesNpy(
    prefix=f"{data_dir}/psv/cfg_",
    suffix=".npy",
    shape=[T, T, Ns, Ns, Np, Nc, Ne],
    Np=Np, Ne=Ne
)

vsp = PropagatorVSPTimeslicesNpy(
    prefix=f"{data_dir}/vsp/cfg_",
    suffix=".npy",
    shape=[T, T, Ns, Ns, Np, Nc, Ne],
    Np=Np, Ne=Ne
)

psp = PropagatorPSPTimeslicesNpy(
    prefix=f"{data_dir}/psp/cfg_",
    suffix=".npy",
    shape=[T, T, Ns, Ns, Np, Nc, Np],
    Np_snk=Np, Np_src=Np
)

# 使用
for cfg in cfg_list:
    vsv_data = vsv.load(cfg)  # [T, Ne, Ne]
    psv_data = psv.load(cfg)  # [T, T, Ns, Ns, Np, Nc, Ne]
    vsp_data = vsp.load(cfg)  # [T, T, Ns, Ns, Np, Nc, Ne]
    psp_data = psp.load(cfg)  # [T, T, Ns, Ns, Np, Nc, Np]
    
    # 进行计算...
```

## 形状对比表

### 单文件版本

| 类型 | 第1维 | 第2维 | 第3维 | 第4维 | 第5维 | 第6维 | 第7维 |
|------|------|------|------|------|------|------|------|
| V2V | Lt (snk) | Lt (src) | Ns (snk) | Ns (src) | Ne (snk) | Ne (src) | - |
| P2V | Lt (snk) | Lt (src) | Ns (snk) | Ns (src) | Np | Nc | Ne |
| V2P | Lt (snk) | Lt (src) | Ns (snk) | Ns (src) | Np | Nc | Ne |
| P2P | Lt (snk) | Lt (src) | Ns (snk) | Ns (src) | Np (snk) | Nc | Np (src) |

### 时间片版本 (每个文件)

| 类型 | 第1维 | 第2维 | 第3维 | 第4维 | 第5维 | 第6维 |
|------|------|------|------|------|------|------|
| V2V | Lt (snk) | Ns (snk) | Ns (src) | Ne (snk) | Ne (src) | - |
| P2V | Lt (snk) | Ns (snk) | Ns (src) | Np | Nc | Ne |
| V2P | Lt (snk) | Ns (snk) | Ns (src) | Np | Nc | Ne |
| P2P | Lt (snk) | Ns (snk) | Ns (src) | Np (snk) | Nc | Np (src) |

## 内存占用对比 (Lt=72, Ne=70, Np=216, complex128)

| 类型 | 单文件大小 | 时间片单个文件 | 推荐方式 |
|------|-----------|--------------|----------|
| V2V (完整) | ~91 GB | ~1.3 GB | ✅ 时间片 |
| V2V (简化) | ~5.6 MB | - | ✅ 单文件 |
| P2V | ~148 GB | ~2.1 GB | ✅ 时间片 |
| V2P | ~148 GB | ~2.1 GB | ✅ 时间片 |
| P2P | ~32 TB | ~450 GB | ⚠️ 极大，慎用 |

## 推荐实践

1. **V2V (Perambulator)**: 
   - 使用简化形状 `[Lt, Ne, Ne]` 的单文件版本
   - 如果需要完整旋量结构，使用时间片版本

2. **P2V, V2P**: 
   - **强烈推荐**使用时间片版本
   - 单文件版本太大 (~148 GB)

3. **P2P**: 
   - 谨慎使用，数据量极大
   - 考虑是否真的需要所有点对点的组合
   - 可能需要更激进的数据压缩或选择性保存

4. **数据类型**:
   - 优先使用 `<c16` (complex128) 保证精度
   - 如空间受限，可考虑 `<c8` (complex64)，但需验证精度影响

5. **文件组织**:
   - 不同类型的 propagator 使用不同的目录
   - 清晰的命名规范，如 `vsv/`, `psv/`, `vsp/`, `psp/`

## 导入所有类

```python
# 从 lattice 包导入
from lattice import (
    # 常量
    Ns, Nc, Nd,
    
    # V2V (Perambulator)
    PerambulatorNpy,
    PerambulatorTimeslicesNpy,
    
    # P2V
    PropagatorPSV,
    PropagatorPSVNpy,
    PropagatorPSVTimeslicesNpy,
    
    # V2P
    PropagatorVSP,
    PropagatorVSPNpy,
    PropagatorVSPTimeslicesNpy,
    
    # P2P
    PropagatorPSP,
    PropagatorPSPNpy,
    PropagatorPSPTimeslicesNpy,
)
```

## 相关文档

- [PropagatorPSV 详细使用文档](PropagatorPSV_Usage_CORRECTED.md)
- [PropagatorPSV 快速参考](PropagatorPSV_QuickRef_CORRECTED.md)
- [Propagator 类对比](Propagator_Classes_Comparison.md)

