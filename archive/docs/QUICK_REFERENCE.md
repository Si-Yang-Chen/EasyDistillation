# EasyDistillation 快速参考指南

**最后更新**: 2026-06-01

---

## 🚀 5分钟快速开始

### 最小示例
```python
from lattice import (
    GaugeFieldIldg, EigenvectorNpy,
    PerambulatorGenerator, Meson,
    set_backend
)

# 1. 设置后端
set_backend("numpy")  # 或 "cupy"

# 2. 加载数据
gauge = GaugeFieldIldg(prefix="/data/cfg_", suffix=".lime",
                       shape=[64, 16, 16, 16, 4, 3, 3])
eigs = EigenvectorNpy(prefix="/data/eigs_", suffix=".npy",
                      shape=[70, 64, 16**3, 3], Ne=70)

# 3. 生成传播子
peram_gen = PerambulatorGenerator(
    latt_size=[16, 16, 16, 64],
    gauge_field=gauge,
    eigenvector_src=eigs,
    mass=0.09253
)
peram = peram_gen.calc()

# 4. 计算关联函数
meson = Meson(irrep_row, flavor)
corr = meson.gen_correlator(peram, peram, t_range=range(64))
```

---

## 📦 核心模块速查

### 数据加载
```python
# 规范场
gauge = GaugeFieldIldg(prefix, suffix, shape)

# 本征矢量
eigs = EigenvectorNpy(prefix, suffix, shape, Ne=70)

# 传播子
peram = PerambulatorNpy(prefix, suffix, shape, Ne=70)
psv = PropagatorPSVNpy(prefix, suffix, shape, Np, Ne)
```

### 数据生成
```python
# 本征矢量
gen = EigenvectorGenerator(latt_size, gauge_field, num_eigs)
eigs = gen.calc()

# 基元
gen = ElementalGenerator(eigenvector, noise_vector)
elem = gen.calc()

# 传播子
gen = PerambulatorGenerator(latt_size, gauge_field, eigs, mass, ...)
peram = gen.calc()
```

### 物理计算
```python
# 介子
meson = Meson(irrep_row, flavor_structure)
corr = meson.gen_correlator(peram_snk, peram_src, t_range)

# 流算符
current = Current(irrep_row, flavor, gamma)
corr_3pt = current.gen_3pt_function(props, t_sep)

# 夸克收缩
result = quark_contract(diagram, propagators, vertices)
```

---

## 📊 数据形状速查表

完整数据类型与文件命名规范见 [doc/README.md](doc/README.md)。以下为常用传播子形状摘要：

### 传播子类型

| 类型 | 单文件形状 | 时间片形状 |
|------|-----------|-----------|
| **V2V** | `[Lt, Lt, Ns, Ns, Ne, Ne]` | `[Lt, Ns, Ns, Ne, Ne]` |
| **P2V** | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| **V2P** | `[Lt, Lt, Ns, Ns, Np, Nc, Ne]` | `[Lt, Ns, Ns, Np, Nc, Ne]` |
| **P2P** | `[Lt, Lt, Ns, Ns, Np_snk, Nc, Np_src]` | `[Lt, Ns, Ns, Np_snk, Nc, Np_src]` |

### 常用维度
- `Lt = 64` (时间长度)
- `Ns = 4` (Dirac 旋量)
- `Nc = 3` (颜色)
- `Ne = 70` (本征矢量数)
- `Np = 216` (点源数)

---

## 🔧 常用操作

### 后端管理
```python
from lattice import set_backend, get_backend, check_QUDA

# 检查 GPU
if check_QUDA():
    set_backend("cupy")
else:
    set_backend("numpy")

backend = get_backend()
```

### 内存优化
```python
# ✅ 推荐：使用时间片格式
peram = PerambulatorTimeslicesNpy(...)

# ✅ 及时释放
del large_data
import gc; gc.collect()

# ✅ 检查内存
from lattice import log_gpu_memory
log_gpu_memory()
```

### 批处理
```python
# 处理多个配置
for cfg in range(1000, 1100):
    gauge.load(str(cfg))
    peram = peram_gen.calc()
    # 处理...
    del peram
```

---

## 🧪 测试与调试

### 运行测试
```bash
# 快速测试
pytest tests/ -v

# 跳过 GPU 测试
pytest tests/ -m "not gpu"

# 特定测试
pytest tests/test_perambulator.py -v
```

### 调试技巧
```python
# 检查数据
data = peram.load("1000")
print(f"Shape: {data.shape}")
print(f"Memory: {data.nbytes / 1024**3:.2f} GB")

# 性能分析
import time
start = time.time()
result = operation()
print(f"Time: {time.time() - start:.2f}s")
```

---

## 📁 项目结构速览

```
lattice/
├── backend.py          # 后端管理
├── preset.py           # 数据类
├── generator/          # 数据生成
│   ├── eigenvector.py
│   ├── perambulator.py
│   └── elemental.py
├── quark_diagram.py    # 夸克图
├── hadron.py           # 强子
└── insertion/          # 插入算符
    ├── gamma.py
    └── derivative.py
```

---

## 🆘 常见问题

### Q: 如何选择传播子类型？
- **V2V**: 两点函数、介子能谱
- **P2V**: 三点函数、流算符矩阵元
- **V2P**: 三点函数（反方向）
- **P2P**: 点源传播子

### Q: GPU 加速如何使用？
```python
from lattice import check_QUDA, set_backend

if check_QUDA():
    set_backend("cupy")
    print("GPU 加速已启用")
```

### Q: 内存不足怎么办？
1. 使用时间片格式
2. 减小 `Np` 或 `Ne`
3. 分批处理
4. 使用内存映射

### Q: 如何添加新的物理算符？
参考 `lattice/hadron.py` 和 `example/` 目录中的示例

---

## 📚 更多资源

- **详细文档**: [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
- **数据形状**: [doc/README.md](doc/README.md)
- **传播子详解**: [doc/propagator_theory_and_usage.md](doc/propagator_theory_and_usage.md)
- **示例代码**: [example/](example/)
- **测试用例**: [tests/](tests/)

---

## ⚡ 性能优化提示

| 优化项 | 方法 | 效果 |
|--------|------|------|
| **内存** | 使用时间片格式 | ↓ 90% 内存 |
| **计算** | GPU 加速 | ↑ 10-100x 速度 |
| **I/O** | SSD 存储 | ↑ 5-10x 读取 |
| **并行** | MPI 分布式 | 线性加速 |

---

**提示**: 将此文档加入书签，作为日常开发参考！
