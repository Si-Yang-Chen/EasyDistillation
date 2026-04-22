# Propagator 类快速参考卡片

## 快速选择指南

| 我需要 | 使用的类 | 形状 (时间片) |
|--------|---------|--------------|
| 本征向量 → 本征向量 | `PerambulatorTimeslicesNpy` | `[Lt, Ns, Ns, Ne, Ne]` |
| 点源 → 本征向量 | `PropagatorPSVTimeslicesNpy` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| 本征向量 → 点源 | `PropagatorVSPTimeslicesNpy` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| 点源 → 点源 | `PropagatorPSPTimeslicesNpy` | `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` |

## 快速导入

```python
from lattice import (
    Ns, Nc,  # 常量: Ns=4, Nc=3
    PerambulatorTimeslicesNpy,
    PropagatorPSVTimeslicesNpy,
    PropagatorVSPTimeslicesNpy,
    PropagatorPSPTimeslicesNpy,
)
```

## 快速代码模板

### V2V (两点函数)

```python
vsv = PerambulatorTimeslicesNpy(
    prefix=f"{data_dir}/vsv/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Ne, Ne],  # 或简化 [Lt, Ne, Ne]
    Ne=Ne
)
```

### P2V (三点函数 - 点源插入)

```python
psv = PropagatorPSVTimeslicesNpy(
    prefix=f"{data_dir}/psv/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=Np, Ne=Ne
)
```

### V2P (三点函数 - 本征向量源)

```python
vsp = PropagatorVSPTimeslicesNpy(
    prefix=f"{data_dir}/vsp/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=Np, Ne=Ne
)
```

### P2P (点对点)

```python
psp = PropagatorPSPTimeslicesNpy(
    prefix=f"{data_dir}/psp/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src],
    Np_snk=Np_snk, Np_src=Np_src
)
```

## 形状速查表

### 完整形状定义

| 维度位置 | [0] | [1] | [2] | [3] | [4] | [5] | [6] |
|---------|-----|-----|-----|-----|-----|-----|-----|
| **V2V** | Lt₁ | Lt₂ | Ns₁ | Ns₂ | Ne₁ | Ne₂ | - |
| **P2V** | Lt₁ | Lt₂ | Ns₁ | Ns₂ | Np | Nc | Ne |
| **V2P** | Lt₁ | Lt₂ | Ns₁ | Ns₂ | Np | Nc | Ne |
| **P2P** | Lt₁ | Lt₂ | Ns₁ | Ns₂ | Np₁ | Nc | Np₂ |

- Lt₁: sink 时间, Lt₂: source 时间
- Ns₁: sink 旋量, Ns₂: source 旋量
- Np, Np₁, Np₂: 点源指标
- Nc: 颜色 (=3)
- Ne, Ne₁, Ne₂: 本征向量指标

### 时间片文件形状

每个时间片文件去掉第一个 Lt 维度（source 时间固定）

| 类型 | 时间片文件形状 |
|------|--------------|
| V2V | `[Lt, Ns, Ns, Ne, Ne]` |
| P2V | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| V2P | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| P2P | `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` |

## 文件命名规范

```
{prefix}{cfg}.t{t_src:03d}{suffix}

示例:
cfg_1000.t000.npy
cfg_1000.t001.npy
...
cfg_1000.t071.npy
```

## 内存快速估算 (complex128)

Lt=72, Ne=70, Np=216 的情况：

| 类型 | 时间片单文件 | 总大小 (72文件) |
|------|-------------|----------------|
| V2V (完整) | 1.3 GB | 91 GB |
| V2V (简化) | - | 5.6 MB |
| P2V | 2.1 GB | 150 GB |
| V2P | 2.1 GB | 150 GB |
| P2P | 450 GB | 32 TB ⚠️ |

## 常用参数

| 参数 | 类型 | 说明 | 典型值 |
|------|------|------|--------|
| `Lt` | int | 时间维度 | 64, 72, 96, 128 |
| `Ns` | int | 旋量维度 | 4 |
| `Nc` | int | 颜色维度 | 3 |
| `Ne` | int | 本征向量数 | 64, 70, 128 |
| `Np` | int | 点源数 | 64, 216 (6³), 512 (8³) |

## 生成数据

```python
from lattice import PerambulatorGenerator

gen = PerambulatorGenerator(
    eigenvector_src=eigenvector,
    eigenvector_snk=eigenvector,
    point_src=point_source,  # P2V, P2P 需要
    point_snk=point_sink,    # V2P, P2P 需要
    ...
)

for t_src in range(Lt):
    VSV, PSV, VSP, PSP = gen.calc(t_src)
    # 保存...
```

## 加载和使用

```python
# 加载
vsv_data = vsv.load("1000")
psv_data = psv.load("1000")
vsp_data = vsp.load("1000")
psp_data = psp.load("1000")

# 在收缩计算中使用
from lattice.quark_diagram import compute_diagrams_multitime

result = compute_diagrams_multitime(
    diagrams,
    time_slices,
    [meson, current],
    [None, vsv, psv]  # 传入 propagator 对象
)
```

## 常见问题速查

**Q: 为什么有两个 Lt 维度？**
- 第一个: sink 时间 (测量时间)
- 第二个: source 时间 (源时间)

**Q: 为什么包含 Nc (颜色)？**
- Propagator 是 Dirac 传播子，包含完整颜色结构
- 收缩时需要对颜色求和

**Q: PSV 和 VSP 形状相同，有什么区别？**
- PSV: 点源 → 本征向量
- VSP: 本征向量 → 点源
- 传播方向不同，但数据结构相同

**Q: 应该用单文件还是时间片版本？**
- **强烈推荐时间片版本**
- 单文件太大（P2V/V2P ~148 GB）
- 时间片便于并行生成和加载

**Q: P2P 为什么这么大？**
- 包含所有点对点组合: Np × Np
- 216 × 216 = 46,656 个点对
- 考虑是否真的需要，或使用采样

## 推荐实践

1. ✅ **V2V**: 简化形状 `[Lt, Ne, Ne]` 单文件
2. ✅ **P2V**: 时间片版本
3. ✅ **V2P**: 时间片版本  
4. ⚠️ **P2P**: 谨慎使用，数据量极大

## 更多信息

- 完整文档: [`doc/All_Propagator_Classes.md`](All_Propagator_Classes.md)
- PSV 详细文档: [`doc/PropagatorPSV_Usage_CORRECTED.md`](PropagatorPSV_Usage_CORRECTED.md)
- 总结: [`PROPAGATOR_CLASSES_SUMMARY.md`](../PROPAGATOR_CLASSES_SUMMARY.md)

