# 传播子 (Propagator) 理论与使用

## 理论基础：基底分解

在 Distillation 方法中，颜色空间的 Hilbert 空间（维数 3·L³）有两种基底表示：

### 1. 空间点基底 |ηₓ,a⟩

- x: 空间坐标
- a: 颜色指标 (1,2,3)
- 完备性: I = Σₓₐ |ηₓ,a⟩⟨ηₓ,a|

### 2. 本征向量基底 |ξᵢ⟩

- Laplace 算符本征态: ∇²|ξᵢ⟩ = λᵢ|ξᵢ⟩
- 按本征值排序，前 Nᵥ 个为"低模" (low modes)

### 单位矩阵分解

```
I = I_low + I_high
I_low  = Σᵢ₌₁ᴺᵛ |ξᵢ⟩⟨ξᵢ|      (低模投影算符)
I_high = I - I_low              (高模投影算符)
```

---

## 理论核心：算符的四项分解

任意算符 O 可分解为四项：

```
O = O_low + O_high + O_low-high + O_high-low
```

| 分量 | 数学形式 | 矩阵元表示 | 对应传播子类型 |
|------|---------|-----------|--------------|
| **O_low** | I_low O I_low | O_{i,j} | **V2V** |
| **O_high** | (I-I_low) O (I-I_low) | O_{xa,yb} | **P2P** |
| **O_low-high** | I_low O (I-I_low) | O_{i,yb} | **V2P** |
| **O_high-low** | (I-I_low) O I_low | O_{xa,j} | **P2V** |

---

## 传播子类型与物理含义

### V2V (Perambulator): 本征向量 → 本征向量

**理论定义**: τ_{i,j} = ⟨ξᵢ| S |ξⱼ⟩

**物理含义**: 在低模子空间内的传播

**代码中的形状**:
```python
# 完整形状（含旋量）
[Lt, Lt, Ns, Ns, Ne, Ne]  # 单文件
[Lt, Ns, Ns, Ne, Ne]       # 时间片文件

# 简化形状（已收缩旋量）
[Lt, Ne, Ne]
```

**存储类**:
- `PerambulatorNpy`: 单文件加载
- `PerambulatorTimeslicesNpy`: 按源时间片分离存储

**用途**:
- 两点函数计算
- 低模贡献的主要部分

---

### P2V (PropagatorPSV): 点 → 本征向量

**理论定义**: S_{xa,j} = ⟨ηₓ,a| S |ξⱼ⟩

**物理含义**: 从点源出发，投影到低模

**矩阵元表示**:
```
PSV[t_snk, t_src, s_snk, s_src, p, c, e]
    ↓       ↓       ↓       ↓     ↓  ↓  ↓
  sink时间 source  sink旋量 src旋量 点 颜色 本征向量
```

**代码中的形状**:
```python
# 完整形状
[Lt, Lt, Ns, Ns, Np, Nc, Ne]  # 单文件
[Lt, Ns, Ns, Np, Nc, Ne]       # 时间片文件
```

**存储类**:
- `PropagatorPSVNpy`: 单文件加载
- `PropagatorPSVTimeslicesNpy`: 按源时间片分离存储

**用途**:
- O_high-low 算符的传播
- 高模采样的关键数据

---

### V2P (PropagatorVSP): 本征向量 → 点

**理论定义**: S_{i,yb} = ⟨ξᵢ| S |ηᵧ,b⟩

**物理含义**: 从低模出发，投影到点

**矩阵元表示**:
```
VSP[t_snk, t_src, s_snk, s_src, e, p, c]
    ↓       ↓       ↓       ↓     ↓  ↓  ↓
  sink时间 source  sink旋量 src旋量 本征向量 点 颜色
```

**代码中的形状**:
```python
# 完整形状
[Lt, Lt, Ns, Ns, Ne, Np, Nc]  # 单文件
[Lt, Ns, Ns, Ne, Np, Nc]       # 时间片文件
```

**存储类**:
- `PropagatorVSPNpy`: 单文件加载
- `PropagatorVSPTimeslicesNpy`: 按源时间片分离存储

**用途**:
- O_low-high 算符的传播
- 与 P2V 的对称性关系: (O)_{i,xa} = (O)_{xa,i}^T

---

### P2P (PropagatorPSP): 点 → 点

**理论定义**: S_{xa,yb} = ⟨ηₓ,a| S |ηᵧ,b⟩

**物理含义**: 纯高模空间内的传播

**矩阵元表示**:
```
PSP[t_snk, t_src, s_snk, s_src, p_snk, c, p_src]
    ↓       ↓       ↓       ↓       ↓      ↓    ↓
  sink时间 source  sink旋量 src旋量  sink点 颜色  src点
```

**代码中的形状**:
```python
# 完整形状
[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]  # 单文件
[Lt, Ns, Ns, Np_snk, Nc, Np_src]       # 时间片文件
```

**存储类**:
- `PropagatorPSPNpy`: 单文件加载
- `PropagatorPSPTimeslicesNpy`: 按源时间片分离存储

**注意**: 数据量极大 (O(Np²))，通常采用稀疏存储或按需计算

---

## 数据生成流程

所有四种传播子由 `PerambulatorGenerator` 统一生成：

```python
from lattice import PerambulatorGenerator

gen = PerambulatorGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector_src=eigenvector,   # 源端本征向量
    eigenvector_snk=eigenvector,   # 汇端本征向量
    point_src=point_source,        # 点源（PSV, PSP 需要）
    point_snk=point_sink,          # 点汇（V2P, PSP 需要）
)

for t_src in range(Lt):
    VSV, PSV, VSP, PSP = gen.calc(t_src)

    # VSV: [Lt, Ns, Ns, Ne_snk, Ne_src] → V2V
    # PSV: [Lt, Ns, Ns, Np_snk, Nc, Ne_src] → P2V
    # VSP: [Lt, Ns, Ns, Ne_snk, Nc, Np_src] → V2P
    # PSP: [Lt, Ns, Ns, Np_snk, Nc, Np_src] → P2P
```

---

## 高模投影传播子

### 理论公式 (公式 5.1-5.3)

在 Localized Blending 方法中，需要对传播子进行高模投影：

```
S̃_{xa,yb} = S_{xa,yb}
           - Σⱼ M_{xj,a} S_{j,yb}
           - S_{xa,j} M_{jy,b}*
           + Σᵢⱼ M_{xi,a} S_{i,j} M_{jy,b}*

S̃_{xa,i} = S_{xa,i} - Σⱼ M_{xj,a} S_{j,i}
S̃_{i,xa} = S_{i,xa} - Σⱼ S_{i,j} M_{jx,a}*
```

其中 M_{xi,a} = ⟨ηₓ,a|ξᵢ⟩ 是 overlap matrix。

### 代码实现

由 `PropagatorWithCurrent` 类处理：

```python
from lattice.quark_diagram import PropagatorWithCurrent

prop = PropagatorWithCurrent(
    perambulator=vsv,           # V2V: S_{i,j}
    propagator_psv=psv,         # P2V: S_{xa,i}
    propagator_vsp=vsp,         # V2P: S_{i,xa}
    propagator_psp=psp,         # P2P: S_{xa,yb}
    overlap_matrix=overlap,     # M_{xi,a}
)

# 获取高模投影后的传播子
prop.get_psv_highmode()  # S̃_{xa,i}
prop.get_vsp_highmode()  # S̃_{i,xa}
prop.get_psp_highmode()  # S̃_{xa,yb}
```

---

## 存储类快速参考

| 类型 | 基类 | 单文件类 | 时间片类 | 完整形状 |
|------|------|---------|---------|---------|
| V2V | `Perambulator` | `PerambulatorNpy` | `PerambulatorTimeslicesNpy` | [Lt,Lt,Ns,Ns,Ne,Ne] |
| P2V | `PropagatorPSV` | `PropagatorPSVNpy` | `PropagatorPSVTimeslicesNpy` | [Lt,Lt,Ns,Ns,Np,Nc,Ne] |
| V2P | `PropagatorVSP` | `PropagatorVSPNpy` | `PropagatorVSPTimeslicesNpy` | [Lt,Lt,Ns,Ns,Ne,Np,Nc] |
| P2P | `PropagatorPSP` | `PropagatorPSPNpy` | `PropagatorPSPTimeslicesNpy` | [Lt,Lt,Ns,Ns,Np_snk,Nc,Np_src] |

### 导入示例

```python
from lattice import (
    # V2V
    PerambulatorNpy,
    PerambulatorTimeslicesNpy,
    # P2V
    PropagatorPSVNpy,
    PropagatorPSVTimeslicesNpy,
    # V2P
    PropagatorVSPNpy,
    PropagatorVSPTimeslicesNpy,
    # P2P
    PropagatorPSPNpy,
    PropagatorPSPTimeslicesNpy,
    # 常量
    Ns, Nc, Nd,
)
```

### 加载示例

```python
# V2V (推荐使用简化形状)
vsv = PerambulatorNpy(
    prefix="/data/vsv/cfg_",
    suffix=".npy",
    shape=[Lt, Ne, Ne],
    totNe=Ne
)

# P2V (推荐使用时间片版本)
psv = PropagatorPSVTimeslicesNpy(
    prefix="/data/psv/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=Np,
    Ne=Ne
)

# 加载数据
vsv_data = vsv.load("1000")  # [Lt, Ne, Ne]
psv_data = psv.load("1000")  # [Lt, Lt, Ns, Ns, Np, Nc, Ne]
```

---

## 内存估算 (complex128, Lt=72, Ne=70, Np=216)

| 类型 | 单文件 | 时间片单文件 | 推荐 |
|------|--------|-------------|------|
| V2V (完整) | ~91 GB | ~1.3 GB | 时间片或简化形状 |
| V2V (简化) | ~5.6 MB | - | 单文件 |
| P2V | ~148 GB | ~2.1 GB | 时间片 |
| V2P | ~148 GB | ~2.1 GB | 时间片 |
| P2P | ~32 TB | ~450 GB | 稀疏存储/按需计算 |

---

## 在收缩计算中的使用

```python
from lattice.quark_diagram import compute_diagrams_multitime

# 传播子对象传入收缩函数
result = compute_diagrams_multitime(
    diagrams=diagrams,
    time_slices=time_slices,
    vertex_objects=[meson, current],
    propagators=[None, vsv, psv],  # 根据需要传入对应的传播子
)
```

**传播子类型选择**:
- 纯低模收缩: 仅需 V2V
- 含点采样收缩: 需要 P2V, V2P, P2P
- 高模投影: 使用 `PropagatorWithCurrent` 包装

---

## 相关文档

- [Localized Blending 理论](localized_blending/localized_blending_theory.md)
- [Localized Blending 实现](localized_blending/localized_blending_implementation.md)
- [Distillation 工作流程](../docs/DISTILLATION_WORKFLOW.md)
