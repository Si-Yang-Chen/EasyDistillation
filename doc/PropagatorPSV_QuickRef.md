# PropagatorPSV 快速参考

## 快速导入

```python
from lattice import PropagatorPSVNpy, PropagatorPSVTimeslicesNpy
```

## 选择合适的类

| 数据存储方式 | 使用的类 |
|------------|---------|
| 单个 `.npy` 文件 | `PropagatorPSVNpy` |
| 按时间片分离的多个 `.npy` 文件 | `PropagatorPSVTimeslicesNpy` |

## 快速示例

### 单文件版本

```python
# 简化形状 [Lt, Np, Ne]
psv = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 216, 70],
    Np=216,
    Ne=70
)
data = psv.load("1000")
```

### 时间片分离版本

```python
# 完整形状 [Lt, Ns, Ns, Np, Ne]
psv = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/data/cfg_",
    suffix=".npy",
    shape=[72, 4, 4, 216, 70],
    Np=216,
    Ne=70
)
data = psv.load("1000")  # 自动加载 t000.npy ~ t071.npy
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

## 数据形状

### 完整形状
```python
shape = [Lt, Ns, Ns, Np, Ne]
# Lt: 时间维度
# Ns: 旋量维度 (=4)
# Np: 点源数量
# Ne: 本征向量数量
```

### 简化形状
```python
shape = [Lt, Np, Ne]
# 已收缩或选择旋量分量
```

## 在收缩计算中使用

```python
from lattice import PropagatorPSVNpy
from lattice.quark_diagram import Current, compute_diagrams_multitime

# 加载 PSV propagator
psv = PropagatorPSVNpy(
    prefix=f"{data_dir}/cfg_",
    suffix=".npy",
    shape=[Lt, Np, Ne],
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

## 常见问题

**Q: 如何选择 dtype？**
- `<c16`: 复数双精度 (complex128, 推荐)
- `<c8`: 复数单精度 (complex64, 节省内存)

**Q: 形状应该选择完整还是简化？**
- 根据实际保存的数据格式选择
- 如果包含 Dirac 旋量结构 → 完整形状
- 如果已经收缩了旋量 → 简化形状

**Q: 时间片文件命名格式？**
- 必须是: `{cfg}.t{XXX}.npy`
- 时间片编号是 3 位数字，如 `t000`, `t001`, `t071`

## 更多信息

- 详细文档: `doc/PropagatorPSV_Usage.md`
- 测试示例: `examples/test_propagator_psv.py`
- 实现总结: `PROPAGATOR_PSV_SUMMARY.md`

