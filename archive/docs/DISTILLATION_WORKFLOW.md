# 传统 Distillation 工作流

本文档详细描述 EasyDistillation 框架中传统 distillation 方法的完整工作流程。

---

## 目录

1. [概述](#概述)
2. [数据流图](#数据流图)
3. [步骤详解](#步骤详解)
   - [步骤 1: 规范场加载](#步骤-1-规范场加载)
   - [步骤 2: 本征矢量生成](#步骤-2-本征矢量生成)
   - [步骤 3: 传播子计算](#步骤-3-传播子计算)
   - [步骤 4: 元素基生成](#步骤-4-元素基生成)
   - [步骤 5: 算符构建](#步骤-5-算符构建)
   - [步骤 6: 关联函数收缩](#步骤-6-关联函数收缩)
4. [完整示例](#完整示例)
5. [性能优化建议](#性能优化建议)

---

## 概述

Distillation 方法是一种用于格点 QCD 计算的降维技术，通过 Laplace 算符的本征矢量空间来减小计算复杂度。核心思想是：

1. **降维**: 将格点上的自由度投影到有限维本征矢量空间
2. **因子化**: 将传播子计算与收缩计算分离
3. **可复用**: 本征矢量和传播子可被多个算符复用

### 关键概念

| 术语 | 符号 | 形状 | 说明 |
|------|------|------|------|
| **本征矢量** | $V$ | `[Lt, Ne, L³, Nc]` | Laplace 算符的低能本征态 |
| **传播子** | $\tau$ | `[Lt, Lt, Ns, Ns, Ne, Ne]` | 本征空间中的夸克传播子 |
| **元素基** | $\Phi$ | `[Ndiagrams, Lt, Ne, Ne]` | 介子/重子的基础构建块 |
| **点源传播子** | $P$ | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | 点源到本征空间的传播子 |

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         传统 Distillation 工作流                         │
└─────────────────────────────────────────────────────────────────────────┘

                              GaugeField
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
        ┌─────────────────────┐     ┌─────────────────────┐
        │  EigenvectorGenerator │     │  ElementalGenerator │  (步骤 2, 4)
        │   Stout Smear        │     │   load(gauge, eig)  │
        │   Laplace Solver     │     │   stout_smear()     │
        └─────────────────────┘     │   calc(t)           │
                    │               └─────────────────────┘
                    ▼                           │
               Eigenvector                      │
                    │                           ▼
                    │                      Elemental
                    │                      (实时计算)
                    │                           │
                    ▼                           │
        ┌─────────────────────┐                 │
        │ PerambulatorGenerator│                 │
        │   (PyQuda solver)    │                 │
        └─────────────────────┘                 │
                    │                           │
                    ▼                           │
              Perambulator                      │
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   Meson / Baryon    │  (步骤 5)
                      │   Operator Builder  │
                      └─────────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │   Contraction       │  (步骤 6)
                      │   twopoint / 3pt    │
                      └─────────────────────┘
                                  │
                                  ▼
                             Correlator
```

---

## 步骤详解

### 步骤 1: 规范场加载

**目的**: 加载规范场配置，这是所有后续计算的起点。

#### 可用数据类

| 类名 | 文件格式 | 典型形状 |
|------|---------|---------|
| `GaugeFieldIldg` | ILDG .lime | `[Lt, L³, Nd, Nc, Nc]` |
| `GaugeFieldTimeSlice` | QDP .mod | `[Lt, L³, Nd, Nc, Nc]` |
| `GaugeFieldBinary` | 二进制 | `[Lt, L³, Nd, Nc, Nc]` |

#### 代码示例

```python
from lattice import GaugeFieldIldg

# 加载 ILDG 格式规范场
gauge = GaugeFieldIldg(
    prefix="/data/gauge/cfg_",
    suffix=".lime",
    shape=[Lt, Lz, Ly, Lx, Nd, Nc, Nc]  # [128, 16, 16, 16, 4, 3, 3]
)

# 加载特定配置
gauge_data = gauge.load("1000")
```

#### 注意事项

- ILDG 格式通常是大端序 (`>c16`)，框架会自动转换为小端序
- 规范场通常需要 Stout smearing 后再用于本征矢量计算
- 形状顺序: `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]` (QDP/ILDG 标准)

---

### 步骤 2: 本征矢量生成

**目的**: 求解 Laplace 算符的本征矢量，构建 distillation 基底。

#### 数学背景

Laplace 算符本征方程：
$$-\nabla^2 V_n(x) = \lambda_n V_n(x)$$

其中 $n = 1, 2, \ldots, N_e$ (通常 $N_e = 70 \sim 300$)

#### 代码示例

```python
from lattice import EigenvectorGenerator, GaugeFieldIldg

# 初始化生成器
eigenvector_gen = EigenvectorGenerator(
    latt_size=[Lx, Ly, Lz, Lt],  # [16, 16, 16, 128]
    gauge_field=gauge,
    num_eigs=70,                  # Ne
    tolerance=1e-9                # 求解精度
)

# 加载规范场并进行 stout smearing
eigenvector_gen.load("1000")
eigenvector_gen.stout_smear(n=20, rho=0.12)  # 20 次 stout smearing

# 对每个时间片求解本征矢量
eigen_vecs = backend.zeros((Lt, Ne, Lz, Ly, Lx, Nc), "<c16")
eigen_vals = backend.zeros((Lt, Ne), "<c16")

for t in range(Lt):
    eigen_vecs[t], eigen_vals[t] = eigenvector_gen.calc(t)
    
# 保存结果
backend.save(f"/data/eigs/cfg_1000.npy", eigen_vecs)
backend.save(f"/data/evals/cfg_1000.npy", eigen_vals)
```

#### 可用数据类（加载已有数据）

| 类名 | 文件格式 | 形状 |
|------|---------|------|
| `EigenvectorNpy` | NumPy .npy | `[Lt, Ne, L³, Nc]` |
| `EigenvectorTimeSlice` | QDP .mod | `[Lt, Ne, L³, Nc]` |

```python
from lattice import EigenvectorNpy

eigs = EigenvectorNpy(
    prefix="/data/eigs/cfg_",
    suffix=".npy",
    shape=[Lt, Ne, Lz, Ly, Lx, Nc],
    totNe=70
)

eigen_vecs = eigs.load("1000")[:]
```

---

### 步骤 3: 传播子计算

**目的**: 计算本征空间中的夸克传播子。

#### 传播子类型

| 类型 | 符号 | 形状 (单文件) | 形状 (时间片) | 物理含义 |
|------|------|--------------|--------------|---------|
| **V2V** | $\tau$ | `[Lt, Lt, Ns, Ns, Ne, Ne]` | `[Lt, Ns, Ns, Ne, Ne]` | 本征矢量→本征矢量 |
| **P2V** | $P_{PSV}$ | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` | 点源→本征矢量 |
| **V2P** | $P_{VSP}$ | `[Lt, Lt, Ns, Ns, Ne, Np, Nc]` | `[Lt, Ns, Ns, Ne, Np, Nc]` | 本征矢量→点 |
| **P2P** | $P_{PSP}$ | `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]` | `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` | 点→点 |

#### 代码示例 - Perambulator (V2V)

```python
from lattice import PerambulatorNpy, PerambulatorTimeslicesNpy

# 单文件格式
peram = PerambulatorNpy(
    prefix="/data/peram/cfg_",
    suffix=".peram.npy",
    shape=[Lt, Lt, Ns, Ns, Ne, Ne],  # [128, 128, 4, 4, 70, 70]
    totNe=70
)

# 时间片格式（内存效率高）
peram_ts = PerambulatorTimeslicesNpy(
    prefix="/data/peram/cfg_",
    suffix=".t???.npy",  # 文件模式
    shape=[Lt, Lt, Ns, Ns, Ne, Ne],
    totNe=70
)

# 加载数据
peram_data = peram.load("1000")
# 访问特定源时间片 (时间片格式)
# peram_data_t10 = peram_ts.load("1000")[(10,)]
```

#### 代码示例 - 点源传播子 (PSV/VSP)

```python
from lattice import PropagatorPSVNpy, PropagatorVSPNpy

# 点源→本征矢量传播子
psv = PropagatorPSVNpy(
    prefix="/data/psv/cfg_",
    suffix=".psv.npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=216,  # 点源数量
    Ne=70
)

# 本征矢量→点传播子
vsp = PropagatorVSPNpy(
    prefix="/data/vsp/cfg_",
    suffix=".vsp.npy",
    shape=[Lt, Lt, Ns, Ns, Ne, Np, Nc],
    Np=216,
    Ne=70
)
```

---

### 步骤 4: 元素基生成

**目的**: 构建介子/重子算符的基础构建块。

#### 数学定义

元素基的定义：
$$\Phi^{ab}(t) = \sum_x V^{\dagger a}(x,t) \Gamma V^b(x,t)$$

对于带导数的元素基：
$$\Phi^{ab}_{\nabla}(t) = \sum_x V^{\dagger a}(x,t) \Gamma \nabla V^b(x,t)$$

#### 元素基生成 (ElementalGenerator)

使用 `ElementalGenerator` 从规范场和本征矢量计算元素基：

```python
from lattice import GaugeFieldIldg, EigenvectorNpy, ElementalGenerator, Nd, Nc

# 参数设置
latt_size = [Lx, Ly, Lz, Lt]  # [16, 16, 16, 128]
Ne = 70
Nnabla = 2  # 导数阶数
mom_list = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]

# 加载输入数据
gauge_field = GaugeFieldIldg(
    prefix="/data/gauge/cfg_",
    suffix=".lime",
    shape=[Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)
eigenvector = EigenvectorNpy(
    prefix="/data/eigs/cfg_",
    suffix=".npy",
    shape=[Lt, Ne, Lz, Ly, Lx, Nc],
    totNe=Ne
)

# 创建生成器
elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Nnabla,           # 导数阶数
    momentum_list=mom_list,      # 动量列表
    usedNe=Ne,                   # 使用的本征矢量数
)

# 加载配置数据
elemental_gen.load("1000")

# 可选: stout smearing
elemental_gen.stout_smear(nstep=10, rho=0.12)

# 计算并保存
num_deriv = (3 ** (Nnabla + 1) - 1) // 2
num_mom = len(mom_list)
data = backend.zeros((Lt, num_deriv, num_mom, Ne, Ne), "<c16")

for t in range(Lt):
    data[t] = elemental_gen.calc(t)

# 保存到文件
backend.save("/data/elemental/cfg_1000.npy", data.transpose(1, 2, 0, 3, 4))
```

#### ElementalGenerator 关键参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `latt_size` | `[Lx, Ly, Lz, Lt]` | 格点尺寸 |
| `gauge_field` | `GaugeField` | 规范场数据加载器 |
| `eigenvector` | `Eigenvector` | 本征矢量数据加载器 |
| `num_nabla` | `int` | 协变导数阶数 (0=无导数, 1=一阶, 等) |
| `momentum_list` | `List[Tuple[int,int,int]]` | 动量列表 |
| `usedNe` | `int` | 使用的本征矢量数量 |
| `calc_mode` | `"calc_deriv"` \| `"calc_disp"` | 计算模式 |

#### 计算模式

**导数模式** (`calc_mode="calc_deriv"`, 默认):
- 计算协变导数形式的元素基
- 结果形状: `[num_derivative, num_momentum, Ne, Ne]`
- `num_derivative = (3^(Nnabla+1) - 1) // 2`

**位移模式** (`calc_mode="calc_disp"`):
- 计算规范链位移形式的元素基
- 结果形状: `[num_disp, num_momentum, Ne, Ne]`

```python
elemental_gen = ElementalGenerator(
    ...,
    calc_mode="calc_disp"
)
```

#### 元素基加载 (ElementalNpy)

计算完成后保存到文件，使用时通过 FileData 类加载：

```python
from lattice import ElementalNpy

elemental = ElementalNpy(
    prefix="/data/elemental/cfg_",
    suffix=".npy",
    shape=[num_deriv, num_mom, Lt, Ne, Ne],
    totNe=70
)

# 加载数据
elemental_data = elemental.load("1000")[:]
```

#### 元素基形状说明

| 参数 | 说明 | 计算公式 |
|------|------|---------|
| `num_derivative` | 导数项数量 | `(3^(Nnabla+1) - 1) // 2` |
| `num_momentum` | 动量数 | `len(momentum_list)` |
| `Lt` | 时间维度 | 用户指定 |
| `Ne` | 本征矢量数量 | 用户指定 |

典型形状: `[num_derivative, num_momentum, Lt, Ne, Ne]`

#### CurrentElementalGenerator

用于电流算符的元素基生成：

```python
from lattice import CurrentElementalGenerator, PointSource

point = PointSource(
    prefix="/data/points/cfg_",
    suffix=".npy",
    shape=[Np, Lt, 3]
)

current_gen = CurrentElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    point=point,
    num_nabla=Nnabla,
    momentum_list=mom_list,
    usedNe=Ne,
    usedNp=Np
)

current_gen.load("1000")

for t in range(Lt):
    v2p = current_gen.calc_v2p(t)  # [num_disp, Ne, Np, Nc]
    p2v = current_gen.calc_p2v(t)  # [num_disp, Np, Ne, Nc]
    v2v = current_gen.calc_v2v(t)  # [num_disp, num_mom, Ne, Ne]
```

#### 内存优化建议

```python
# 使用 usedNe 限制本征矢量数量
elemental_gen = ElementalGenerator(..., usedNe=50)

# 分批计算并保存
for cfg in cfg_list:
    elemental_gen.load(cfg)
    elemental_gen.stout_smear(10, 0.12)
    
    data = backend.zeros((Lt, num_deriv, num_mom, usedNe, usedNe), "<c16")
    for t in range(Lt):
        data[t] = elemental_gen.calc(t)
    
    # 立即保存并释放内存
    backend.save(f"/data/elemental/cfg_{cfg}.npy", data)
    del data
    backend.cuda.Stream.null.synchronize()
```

---

### 步骤 5: 算符构建

**目的**: 构建具有特定量子数的介子/重子算符。

#### 插入算符 (Insertion)

```python
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName
from lattice.insertion.mom_dict import momDict_mom9

# 构建单个插入算符
# π 介子: Γ = γ₅, 无导数, A1 表示
pi_A1 = Insertion(
    GammaName.PI,           # γ₅
    DerivativeName.IDEN,    # 无导数
    ProjectionName.A1,      # A1 表示
    momDict_mom9            # 动量字典
)

# ρ 介子: Γ = γᵢ, 无导数, T1 表示
rho_T1 = Insertion(
    GammaName.RHO,          # γᵢ
    DerivativeName.IDEN,
    ProjectionName.T1,
    momDict_mom9
)

# 带导数的算符
b1xnabla_A1 = Insertion(
    GammaName.B1,
    DerivativeName.NABLA,   # ∇ 导数
    ProjectionName.A1,
    momDict_mom9
)
```

#### 算符 (Operator)

```python
# 单一结构的算符
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

# 混合结构的算符 (线性组合)
op_pi2 = Operator("pi2", 
    [pi_A1[0](0, 0, 0), b1xnabla_A1[0](0, 0, 0)],  # 多个插入
    [3, 1]  # 系数
)
```

#### 可用的 Gamma 矩阵

| 名称 | 符号 | 物理意义 |
|------|------|---------|
| `GammaName.IDEN` | $I$ | 单位矩阵 |
| `GammaName.GAMMA5` | $\gamma_5$ | 手征矩阵 |
| `GammaName.PI` | $\gamma_5$ | π 介子 |
| `GammaName.RHO` | $\gamma_i$ | ρ 介子 |
| `GammaName.B1` | $\gamma_5\gamma_i$ | B1 表示 |
| `GammaName.A1` | $\gamma_4$ | A1 表示 |

#### 可用的导数

| 名称 | 符号 | 说明 |
|------|------|------|
| `DerivativeName.IDEN` | $I$ | 无导数 |
| `DerivativeName.NABLA` | $\nabla_i$ | 协变导数 |
| `DerivativeName.DELTA` | $\Delta_i$ | 拉普拉斯 |

---

### 步骤 6: 关联函数收缩

**目的**: 通过张量收缩计算关联函数。

#### 两点函数

**方法 1: 使用 `twopoint` 函数**

```python
from lattice.correlator.one_particle import twopoint, twopoint_matrix

# 计算单算符两点函数
twopt = twopoint(
    [op_pi, op_pi],         # 源和汇算符
    elemental_data,          # 元素基数据
    peram_data,              # 传播子数据
    t_snk_list,              # 汇时间片列表
    Lt                       # 总时间长度
)

# 计算 2×2 矩阵两点函数
twopt_matrix = twopoint_matrix(
    [op_pi, op_pi2],         # 算符列表
    elemental_data,
    peram_data,
    t_snk_list,
    Lt
)
```

**方法 2: 使用 `Meson` 和 `QuarkDiagram`**

```python
from lattice import QuarkDiagram, compute_diagrams_multitime, Meson, Propagator, PropagatorLocal

# 定义夸克图拓扑
connected = QuarkDiagram([
    [0, 1],  # 第一条夸克线: 源→汇
    [1, 0],  # 第二条夸克线: 汇→源
])

disconnected = QuarkDiagram([
    [2, 0],
    [0, 2],
])

# 构建介子对象
eta_src = Meson(elemental, op_pi2, is_source=True)
eta_snk = Meson(elemental, op_pi2, is_source=False)

# 构建传播子对象
propag = Propagator(perambulator, Lt)
propag_local = PropagatorLocal(perambulator, Lt)

# 加载数据
eta_src.load(cfg, usedNe=50)
eta_snk.load(cfg, usedNe=50)
propag.load(cfg, usedNe=50)
propag_local.load(cfg, usedNe=50)

# 计算关联函数
for t_src in range(Lt):
    result = compute_diagrams_multitime(
        [connected, disconnected],
        [t_src, t_snk],
        [eta_src, eta_snk],
        [None, propag, propag_local],
    )
```

#### 三点函数

```python
# 三点函数拓扑: ρ → ππ
rho2pipi = QuarkDiagram([
    [0, 1, 0],  # 夸克线
    [0, 0, 2],  # 反夸克线
    [1, 0, 0],  # 另一条夸克线
])

# 计算三点函数
result = compute_diagrams_multitime(
    [rho2pipi],
    [t_src, t_ins, t_snk],    # 源、插入、汇时间
    [rho_src, pi_snk, pi_snk],
    [None, propag, propag_local],
)
```

#### 色散关系计算

```python
from lattice.correlator.disperion_relation import twopoint_mom2

# 计算动量平方为 p²=2 的两点函数
twopt_mom2 = twopoint_mom2(
    pi_A1[0],
    p_squared=2,
    elemental_data,
    peram_data,
    t_snk_list,
    Lt
)
```

---

## 完整示例

### 示例 1: π 介子两点函数 (从文件加载)

```python
from lattice import set_backend, get_backend
set_backend("cupy")
backend = get_backend()

# ===== 参数设置 =====
Lt, Lx, Ly, Lz = 128, 16, 16, 16
Ne = 70

# ===== 步骤 1: 加载数据 =====
from lattice import preset

elemental = preset.ElementalNpy(
    "/data/elemental/cfg_",
    ".mom9.npy",
    [4, Lt, Ne, Ne],  # [Ndiagrams, Lt, Ne, Ne]
    totNe=70,
)

perambulator = preset.PerambulatorNpy(
    "/data/peram/cfg_",
    ".peram.npy",
    [Lt, Lt, 4, 4, Ne, Ne],
    totNe=70,
)

# ===== 步骤 2: 构建算符 =====
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName
from lattice.insertion.mom_dict import momDict_mom9

pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

# ===== 步骤 3: 计算关联函数 =====
from lattice.correlator.one_particle import twopoint

cfg = "2000"
e = elemental.load(cfg)
p = perambulator.load(cfg)

t_snk = list(range(Lt))
twopt = twopoint([op_pi, op_pi], e, p, t_snk, Lt).real

# ===== 步骤 4: 提取有效质量 =====
# m_eff(t) = arccosh[(C(t-1) + C(t+1)) / (2*C(t))]
m_eff = backend.arccosh(
    (backend.roll(twopt, -1) + backend.roll(twopt, 1)) / (2 * twopt)
)
print(m_eff)
```

### 示例 1b: 实时生成元素基并计算两点函数

```python
from lattice import set_backend, get_backend
set_backend("cupy")
backend = get_backend()

# ===== 参数设置 =====
latt_size = [16, 16, 16, 128]  # [Lx, Ly, Lz, Lt]
Lx, Ly, Lz, Lt = latt_size
Ne = 70
Nnabla = 2

# ===== 步骤 1: 加载规范场和本征矢量 =====
from lattice import GaugeFieldIldg, EigenvectorNpy, ElementalGenerator, Nd, Nc

gauge_field = GaugeFieldIldg(
    "/data/gauge/cfg_", ".lime",
    [Lt, Lz, Ly, Lx, Nd, Nc, Nc]
)
eigenvector = EigenvectorNpy(
    "/data/eigs/cfg_", ".npy",
    [Lt, Ne, Lz, Ly, Lx, Nc], Ne
)

# ===== 步骤 2: 创建元素基生成器 =====
mom_list = [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Nnabla,
    momentum_list=mom_list,
    usedNe=Ne
)

# ===== 步骤 3: 加载配置并计算元素基 =====
cfg = "2000"
elemental_gen.load(cfg)
elemental_gen.stout_smear(10, 0.12)  # 可选: stout smearing

num_deriv = (3 ** (Nnabla + 1) - 1) // 2
num_mom = len(mom_list)
elemental_data = backend.zeros((num_deriv, num_mom, Lt, Ne, Ne), "<c16")

from time import perf_counter
for t in range(Lt):
    s = perf_counter()
    elemental_data[:, :, t] = elemental_gen.calc(t)
    print(f"t={t}: {perf_counter() - s:.2f} sec")

# ===== 步骤 4: 加载传播子并构建算符 =====
from lattice import PerambulatorNpy
from lattice.insertion import Insertion, Operator, GammaName, DerivativeName, ProjectionName
from lattice.insertion.mom_dict import momDict_mom9

peram = PerambulatorNpy(
    "/data/peram/cfg_", ".peram.npy",
    [Lt, Lt, 4, 4, Ne, Ne], totNe=Ne
)
peram_data = peram.load(cfg)

pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])

# ===== 步骤 5: 计算两点函数 =====
from lattice.correlator.one_particle import twopoint

t_snk = list(range(Lt))
twopt = twopoint([op_pi, op_pi], elemental_data, peram_data, t_snk, Lt).real
print(twopt)
```

### 示例 2: 多算符矩阵

```python
from lattice.correlator.one_particle import twopoint_matrix

# 构建多个算符
pi_A1 = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)
pi2_A1 = Insertion(GammaName.PI_2, DerivativeName.IDEN, ProjectionName.A1, momDict_mom9)

op_pi = Operator("pi", [pi_A1[0](0, 0, 0)], [1])
op_pi2 = Operator("pi2", [pi2_A1[0](0, 0, 0)], [1])

# 计算 2×2 矩阵
twopt_matrix = twopoint_matrix(
    [op_pi, op_pi2],
    e, p, t_snk, Lt
).real

# 对角元素给出各算符的有效质量
m_eff_00 = backend.arccosh(
    (backend.roll(twopt_matrix[0, 0], -1) + backend.roll(twopt_matrix[0, 0], 1)) 
    / (2 * twopt_matrix[0, 0])
)
m_eff_11 = backend.arccosh(
    (backend.roll(twopt_matrix[1, 1], -1) + backend.roll(twopt_matrix[1, 1], 1)) 
    / (2 * twopt_matrix[1, 1])
)
```

### 示例 3: 使用时间片格式节省内存

```python
from lattice import PerambulatorTimeslicesNpy

# 时间片格式: 每个源时间单独文件
peram_ts = PerambulatorTimeslicesNpy(
    prefix="/data/peram/cfg_",
    suffix=".t???.npy",
    shape=[Lt, Lt, 4, 4, Ne, Ne],
    totNe=70
)

# 只加载需要的源时间片
peram_data = peram_ts.load("2000")

# 访问特定源时间
for t_src in range(Lt):
    peram_slice = peram_data[(t_src,)]  # 只加载 t_src 的数据
    # 进行计算...
```

---

## 性能优化建议

### 1. 内存管理

- **使用时间片格式**: 对于大格点，使用 `Timeslices` 版本的类，按需加载数据
- **设置 `usedNe`**: 在 `load()` 时使用 `usedNe` 参数限制本征矢量数量
- **及时释放**: 使用 `del` 和垃圾回收释放不再需要的数据

```python
import gc

# 加载部分数据
peram.load(cfg, usedNe=50)  # 只用 50 个本征矢量

# 计算完成后释放
del peram_data
gc.collect()
```

### 2. GPU 加速

```python
from lattice import set_backend
set_backend("cupy")  # 使用 CuPy GPU 后端

# 设置 CuPy 加速器
import os
os.environ["CUPY_ACCELERATORS"] = "cub,cutensor"
```

### 3. 批量配置处理

```python
# 预加载算符信息，避免重复计算
from lattice.correlator.one_particle import twopoint

cfg_list = ["1000", "2000", "3000"]
results = []

for cfg in cfg_list:
    e = elemental.load(cfg)
    p = perambulator.load(cfg)
    
    twopt = twopoint([op_pi, op_pi], e, p, t_snk, Lt)
    results.append(twopt)
    
    # 及时释放
    del e, p
    gc.collect()

# 对所有配置取平均
final_result = backend.mean(results, axis=0)
```

### 4. 性能监控

```python
from lattice.backend import log_gpu_memory

log_gpu_memory("Before loading")
peram.load(cfg, usedNe=50)
log_gpu_memory("After loading peram")
meson.load(cfg, usedNe=50)
log_gpu_memory("After loading meson")
```

---

## 附录: 文件命名约定

### 时间片文件

```
{prefix}{cfg}.t{t_src:03d}{suffix}

示例:
/data/peram/cfg_2000.t000.npy
/data/peram/cfg_2000.t001.npy
...
/data/peram/cfg_2000.t127.npy
```

### 配置文件

```
{prefix}{cfg}{suffix}

示例:
/data/gauge/cfg_2000.lime
/data/peram/cfg_2000.peram.npy
/data/eigs/cfg_2000.npy
```

---

## 相关文档

- [PROJECT_ARCHITECTURE.md](../PROJECT_ARCHITECTURE.md) - 项目整体架构
- [QUICK_REFERENCE.md](../QUICK_REFERENCE.md) - 快速参考
- [FILEDATA_DETAILED.md](../FILEDATA_DETAILED.md) - FileData 详解

---

## 理论基础

Distillation 方法的理论基础详见:

> **Hadron Structure from Lattice QCD with Distillation**  
> M. Peardon, J. Bulava, J. Foley, C. Hölbling, J. Hund, C. Morningstar, C. Urbach  
> arXiv:0905.2160  
> https://arxiv.org/abs/0905.2160

该论文提出了 distillation 技术，核心思想包括：

1. **Laplace 本征基**: 使用 Laplace 算符的低能本征态构建正交基底
2. **降维投影**: 将格点场投影到有限维本征空间
3. **因子化计算**: 分离传播子计算与算符构建，提高效率
4. **本征矢量可复用**: 同一本征矢量集可用于多个物理量的计算
