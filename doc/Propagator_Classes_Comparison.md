# Propagator 和 Perambulator 类对比

本文档对比了项目中不同的 propagator 和 perambulator 类。

## 类层次结构

```
FileData (基类)
├── BinaryFile
│   ├── PerambulatorBinary
│   └── ...
├── NdarrayFile
│   ├── PerambulatorNpy
│   ├── PropagatorPSVNpy ⭐ 新增
│   ├── EigenvectorNpy
│   ├── PointSourceNpy
│   └── ...
└── NdarrayTimeslicesFile
    ├── PerambulatorTimeslicesNpy
    └── PropagatorPSVTimeslicesNpy ⭐ 新增
```

## 主要类对比

### 1. Perambulator 系列（本征向量 ↔ 本征向量）

| 类名 | 存储方式 | 数据形状 | 物理含义 |
|------|---------|---------|---------|
| `Perambulator` | 基类 | - | V2V 传播子接口 |
| `PerambulatorBinary` | 二进制文件 | `[Lt, Lt, Ns, Ns, Ne, Ne]` | 从二进制文件加载 |
| `PerambulatorNpy` | 单个 `.npy` | `[Lt, Lt, Ns, Ns, Ne, Ne]` 或 `[Lt, Ne, Ne]` | 从单个 npy 文件加载 |
| `PerambulatorTimeslicesNpy` | 多个 `.npy` | `[Lt, Lt, Ns, Ns, Ne, Ne]` 或 `[Lt, Ne, Ne]` | 按时间片分离存储 |

**用途**: 强子两点函数、介子算符等

### 2. PropagatorPSV 系列（点源 ↔ 本征向量）⭐ 新增

| 类名 | 存储方式 | 数据形状 | 物理含义 |
|------|---------|---------|---------|
| `PropagatorPSV` | 基类 | - | PSV 传播子接口 |
| `PropagatorPSVNpy` | 单个 `.npy` | `[Lt, Ns, Ns, Np, Ne]` 或 `[Lt, Np, Ne]` | 从单个 npy 文件加载 |
| `PropagatorPSVTimeslicesNpy` | 多个 `.npy` | `[Lt, Ns, Ns, Np, Ne]` 或 `[Lt, Np, Ne]` | 按时间片分离存储 |

**用途**: 流算符插入、三点函数、Current 类

### 3. 其他相关类

| 类名 | 数据形状 | 物理含义 |
|------|---------|---------|
| `Eigenvector` | `[Lt, Ne, Lz, Ly, Lx, Nc]` | Laplace 本征向量 |
| `PointSource` | `[Np, Lt, 3]` | 点源坐标 |
| `Elemental` | `[derivative_num, momentum_num, Lt, Ne, Ne]` | 基本算符矩阵元 |

## 维度含义说明

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `Lt` | 时间维度 | 64, 72, 96, 128 |
| `Lx, Ly, Lz` | 空间维度 | 16, 24, 32, 48 |
| `Ns` | Dirac 旋量维度 | 4 |
| `Nc` | 颜色维度 | 3 |
| `Ne` | 本征向量数量 | 64, 70, 128, 256 |
| `Np` | 点源数量 | 64, 128, 216 (6³), 512 (8³) |
| `Nd` | 时空维度 | 4 |

## 数据形状详解

### Perambulator 完整形状: `[Lt, Lt, Ns, Ns, Ne, Ne]`

```python
perambulator[t_snk, t_src, s_snk, s_src, e_snk, e_src]
```

- 第一个 `Lt`: sink 时间
- 第二个 `Lt`: source 时间
- 第一个 `Ns`: sink 旋量指标
- 第二个 `Ns`: source 旋量指标
- 第一个 `Ne`: sink 本征向量指标
- 第二个 `Ne`: source 本征向量指标

**简化形状**: `[Lt, Ne, Ne]` - 已收缩时间和旋量

### PropagatorPSV 完整形状: `[Lt, Ns, Ns, Np, Ne]`

```python
psv[t, s_snk, s_src, p, e]
```

- `Lt`: 相对时间（sink 时间 - source 时间）
- 第一个 `Ns`: sink 旋量指标
- 第二个 `Ns`: source 旋量指标
- `Np`: 点源指标
- `Ne`: 本征向量指标

**简化形状**: `[Lt, Np, Ne]` - 已收缩旋量

## 使用场景对比

### 两点函数计算

```python
# 使用 Perambulator (V2V)
perambulator = PerambulatorNpy(...)
meson = Meson(elemental, operator, source)
result = meson.get(t)  # 计算介子两点函数
```

### 三点函数计算

```python
# 使用 Perambulator + PropagatorPSV
vsv = PerambulatorNpy(...)        # V2V
psv = PropagatorPSVNpy(...)       # P2V 或 V2P

current = Current(elemental, operator, source, eigenvector, point, gauge)
result = compute_diagrams_multitime(
    diagrams,
    time_slices,
    [meson, current],
    [None, vsv, psv]  # 不同的 propagator
)
```

## 文件命名约定

### 单文件方式

```
Perambulator:      cfg_1000.peram.npy
PropagatorPSV:     cfg_1000.psv.npy
```

### 时间片分离方式

```
Perambulator:      cfg_1000.t000.npy, cfg_1000.t001.npy, ...
PropagatorPSV:     cfg_1000.t000.npy, cfg_1000.t001.npy, ...
```

**注意**: 确保不同类型的数据使用不同的目录或文件名前缀/后缀

## 生成方式

### Perambulator (V2V)

```python
from lattice import PerambulatorGenerator

gen = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,
    eigenvector_snk=eigenvector,
    ...
)

VSV, PSV, VSP, PSP = gen.calc(t_src)
# VSV 就是 Perambulator (V2V)
```

### PropagatorPSV (P2V)

```python
from lattice import PerambulatorGenerator

gen = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,
    eigenvector_snk=eigenvector,
    point_snk=point_source,  # ⭐ 指定点源
    ...
)

VSV, PSV, VSP, PSP = gen.calc(t_src)
# PSV 就是 PropagatorPSV (P2V)
```

## 内存占用估算

假设使用 complex128 (`<c16`, 16 bytes per element):

### Perambulator

**完整形状** `[72, 72, 4, 4, 70, 70]`:
- 大小: 72 × 72 × 4 × 4 × 70 × 70 × 16 bytes ≈ 91 GB ❌ 太大！

**简化形状** `[72, 70, 70]`:
- 大小: 72 × 70 × 70 × 16 bytes ≈ 5.6 MB ✅ 可行

### PropagatorPSV

**完整形状** `[72, 4, 4, 216, 70]`:
- 大小: 72 × 4 × 4 × 216 × 70 × 16 bytes ≈ 1.4 GB ⚠️ 较大但可行

**简化形状** `[72, 216, 70]`:
- 大小: 72 × 216 × 70 × 16 bytes ≈ 87 MB ✅ 推荐

## 性能考虑

| 因素 | Perambulator | PropagatorPSV |
|------|-------------|---------------|
| **计算成本** | O(Ne²) | O(Ne × Np) |
| **存储大小** | 较小 (Ne² < Ne×Np) | 较大 |
| **生成时间** | 快 | 慢（需点提取） |
| **应用** | 两点函数 | 三点函数、流插入 |

**建议**:
- 两点函数: 仅使用 Perambulator (V2V)
- 三点函数: 同时使用 Perambulator (V2V) 和 PropagatorPSV (P2V)

## 总结

| 需求 | 推荐类 | 简化形状 | 完整形状 |
|------|--------|---------|---------|
| 强子两点函数 | `PerambulatorNpy` | `[Lt, Ne, Ne]` | `[Lt, Lt, Ns, Ns, Ne, Ne]` |
| 流算符插入 | `PropagatorPSVNpy` | `[Lt, Np, Ne]` | `[Lt, Ns, Ns, Np, Ne]` |
| 三点函数 | 两者都需要 | - | - |
| 按时间片生成 | `*TimeslicesNpy` 版本 | - | - |

更多详情请参考:
- [PropagatorPSV 使用指南](PropagatorPSV_Usage.md)
- [PropagatorPSV 快速参考](PropagatorPSV_QuickRef.md)

