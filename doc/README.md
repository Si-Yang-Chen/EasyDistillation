# 数据类型与形状参考

本文档提供各数据类型的基本形状信息。详细的理论解释和使用方法请参考 [propagator_theory_and_usage.md](propagator_theory_and_usage.md)。

## 常量

```python
from lattice import Ns, Nc, Nd
# Ns = 4  (Dirac 旋量维度)
# Nc = 3  (颜色维度)
# Nd = 4  (时空维度)
```

## 数据形状速查表

### GaugeField (规范场)

| 格式 | 类 | 形状 |
|------|-----|------|
| ILDG (.lime) | `GaugeFieldIldg` | [Lt, Lz, Ly, Lx, Nd, Nc, Nc] |
| Binary (.dat) | `GaugeFieldBinary` | [Lt, Lz, Ly, Lx, Nd, Nc, Nc] |

### Eigenvector (本征向量)

| 格式 | 类 | 形状 |
|------|-----|------|
| NPY | `EigenvectorNpy` | [Lt, Ne, Lz, Ly, Lx, Nc] |
| MOD (QDP) | `EigenvectorTimeSlice` | [Lt, Ne, Lz, Ly, Lx, Nc] |

### Elemental (基本算符矩阵元)

| 格式 | 类 | 形状 |
|------|-----|------|
| NPY | `ElementalNpy` | [num_deriv, num_mom, Lt, Ne, Ne] |

### Perambulator / Propagator

参见 [propagator_theory_and_usage.md](propagator_theory_and_usage.md) 获取详细理论说明。

| 类型 | 物理含义 | 完整形状 | 简化形状 |
|------|---------|---------|---------|
| **V2V** | ξ → ξ | [Lt, Lt, Ns, Ns, Ne, Ne] | [Lt, Ne, Ne] |
| **P2V** | η → ξ | [Lt, Lt, Ns, Ns, Np, Nc, Ne] | - |
| **V2P** | ξ → η | [Lt, Lt, Ns, Ns, Ne, Np, Nc] | - |
| **P2P** | η → η | [Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src] | - |

### CurrentElemental (流算符矩阵元)

用于 Localized Blending 方法中的点采样。

| 类型 | 物理含义 | 形状 |
|------|---------|------|
| **V2P** | ξ → η | [Lt, num_disp, Ne, Np, Nc] |
| **P2V** | η → ξ | [Lt, num_disp, Np, Nc, Ne] |
| **P2P** | η → η | 稀疏存储 (HDF5/NPZ) |

### OverlapMatrix

| 类型 | 形状 |
|------|------|
| Overlap | [Lt, Ne, Np, Nc] |

---

## 文件命名规范

### 单文件方式

```
{prefix}{cfg}{suffix}

示例:
cfg_1000.peram.npy      # Perambulator
cfg_1000.psv.npy        # P2V propagator
cfg_1000.elemental.npy  # Elemental
```

### 时间片分离方式

```
{prefix}{cfg}.t{t_src:03d}{suffix}

示例:
cfg_1000.t000.npy       # t_src = 0
cfg_1000.t001.npy       # t_src = 1
...
cfg_1000.t071.npy       # t_src = 71
```

---

## 详细文档

- **传播子理论与使用**: [propagator_theory_and_usage.md](propagator_theory_and_usage.md)
- **Localized Blending 理论**: [localized_blending/localized_blending_theory.md](localized_blending/localized_blending_theory.md)
- **Localized Blending 实现**: [localized_blending/localized_blending_implementation.md](localized_blending/localized_blending_implementation.md)
- **QuarkDiagram 统一接口**: [unify_vertex_point_color_indices.md](unify_vertex_point_color_indices.md)
