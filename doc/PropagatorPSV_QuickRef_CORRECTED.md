# PropagatorPSV 快速参考 (更新版)

## 重要更新 ⚠️

**PSV propagator 的正确形状包含颜色指标 Nc！**

- 单文件版本: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` (7维)
- 时间片版本: `[Lt, Ns, Ns, Np, Nc, Ne]` (6维，每个文件)

## 快速导入

```python
from lattice import PropagatorPSVNpy, PropagatorPSVTimeslicesNpy
from lattice import Ns, Nc  # 常量: Ns=4, Nc=3
```

## 选择合适的类

| 数据存储方式 | 使用的类 | 推荐 |
|------------|---------|------|
| 单个 `.npy` 文件 (~148 GB) | `PropagatorPSVNpy` | ⚠️ 文件太大 |
| 按时间片分离的多个 `.npy` 文件 (~2.1 GB/文件) | `PropagatorPSVTimeslicesNpy` | ✅ **推荐** |

## 快速示例

### 时间片分离版本 (推荐)

```python
from lattice import PropagatorPSVTimeslicesNpy, Ns, Nc

# 形状 [Lt, Lt, Ns, Ns, Np, Nc, Ne]
psv = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 72, 4, 4, 216, 3, 70],  # 完整形状
    Np=216,
    Ne=70
)
# 自动加载 t000.npy ~ t071.npy (每个文件 [72, 4, 4, 216, 3, 70])
data = psv.load("1000")
print(data.shape)  # (72, 72, 4, 4, 216, 3, 70)
```

### 单文件版本 (仅小数据集)

```python
from lattice import PropagatorPSVNpy, Ns, Nc

# 完整形状 [Lt, Lt, Ns, Ns, Np, Nc, Ne]
psv = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psv.npy",
    shape=[72, 72, 4, 4, 216, 3, 70],
    Np=216,
    Ne=70
)
data = psv.load("1000")
```

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `prefix` | str | 文件路径前缀 |
| `suffix` | str | 文件后缀（默认 ".npy"） |
| `shape` | List[int] | 数据形状（见下方） |
| `Np` | int | 点源数量 |
| `Ne` | int | 本征向量数量 |
| `dtype` | str | 数据类型（默认 "<c16"） |

## 数据形状 (重要!)

### 单文件版本完整形状
```python
shape = [Lt, Lt, Ns, Ns, Np, Nc, Ne]
# [0] Lt: sink 时间
# [1] Lt: source 时间
# [2] Ns: sink 旋量 (=4)
# [3] Ns: source 旋量 (=4)
# [4] Np: 点源数量
# [5] Nc: 颜色 (=3)
# [6] Ne: 本征向量数量
```

### 时间片版本形状（每个文件）
```python
shape_per_file = [Lt, Ns, Ns, Np, Nc, Ne]
# 每个文件对应一个 t_src
# 加载后组成: [Lt, Lt, Ns, Ns, Np, Nc, Ne]
```

## 与 gen_propagator.py 对应

```python
# 生成时 (2.gen_propagator.py)
for t_src in range(Lt):
    VSV, PSV, VSP, PSP = perambulator.calc(t_src)
    # PSV.get() 形状: [Lt, Ns, Ns, Np, Nc, Ne]
    peramb_PSV = np.roll(PSV.get(), -t_src, 0)
    np.save(f"{save_dir}/cfg_{cfg}.t{t_src:03d}.npy", peramb_PSV)

# 加载时 (4.contraction.py)
psv = PropagatorPSVTimeslicesNpy(
    prefix=f"{save_dir}/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],  # 完整形状
    Np=Np, Ne=Ne
)
data = psv.load(cfg)  # 形状: [Lt, Lt, Ns, Ns, Np, Nc, Ne]
```

## 在收缩计算中使用

```python
from lattice import PropagatorPSVTimeslicesNpy, Ns, Nc
from lattice.quark_diagram import Current, compute_diagrams_multitime

# 加载 PSV propagator
psv = PropagatorPSVTimeslicesNpy(
    prefix=f"{data_dir}/cfg_",
    suffix=".npy",
    shape=[Lt, Lt, Ns, Ns, Np, Nc, Ne],
    Np=Np,
    Ne=Ne
)

# 在计算中使用
for cfg in cfg_list:
    psv_data = psv.load(cfg)
    result = compute_diagrams_multitime(
        diagrams,
        time_slices,
        operators,
        [None, vsv_propagator, psv]  # PSV 传入这里
    )
```

## 内存占用

| 配置 | 单文件大小 | 推荐 |
|------|-----------|------|
| Lt=72, Np=216, Ne=70 (complex128) | ~148 GB | ❌ 太大 |
| Lt=72, Np=216, Ne=70 (complex128, 时间片) | ~2.1 GB × 72 = ~150 GB total | ✅ 推荐（分片） |
| Lt=72, Np=64, Ne=70 (complex128, 时间片) | ~630 MB × 72 = ~45 GB total | ✅ 可行 |

## 常见问题

**Q: 为什么包含 Nc (颜色) 维度？**
- PSV 是 Dirac propagator，包含完整的颜色结构
- 在收缩时需要对颜色指标求和

**Q: 形状应该用哪个？**
- **单文件**: `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`
- **时间片**: 指定完整形状 `[Lt, Lt, Ns, Ns, Np, Nc, Ne]`，每个文件自动是 `[Lt, Ns, Ns, Np, Nc, Ne]`

**Q: 时间片文件命名格式？**
- 必须是: `{cfg}.t{XXX}.npy`，如 `cfg_1000.t000.npy`
- 时间片编号是 3 位数字

**Q: 数据类型选择？**
- `<c16`: complex128 (推荐，精度高)
- `<c8`: complex64 (节省50%空间，精度降低)

## 完整示例代码

```python
from lattice import (
    PropagatorPSVTimeslicesNpy,
    PerambulatorNpy,
    Ns, Nc
)
from lattice.quark_diagram import (
    Current, Meson,
    compute_diagrams_multitime
)

# 参数
L, T = 24, 72
Np, Ne = 216, 70

# 加载 propagators
vsv = PerambulatorNpy(
    prefix=f"{data_dir}/vsv/cfg_",
    suffix=".npy",
    shape=[T, Ne, Ne],
    Ne=Ne
)

psv = PropagatorPSVTimeslicesNpy(
    prefix=f"{data_dir}/psv/cfg_",
    suffix=".npy",
    shape=[T, T, Ns, Ns, Np, Nc, Ne],
    Np=Np, Ne=Ne
)

# 计算
for cfg in cfg_list:
    vsv_data = vsv.load(cfg)
    psv_data = psv.load(cfg)
    
    result = compute_diagrams_multitime(
        [[0,1],[2,0]],
        np.arange(T),
        [meson, current],
        [None, vsv, psv]
    )
```

## 更多信息

- 详细文档: `doc/PropagatorPSV_Usage_CORRECTED.md`
- 测试示例: `examples/test_propagator_psv.py`
- 类对比: `doc/Propagator_Classes_Comparison.md`

