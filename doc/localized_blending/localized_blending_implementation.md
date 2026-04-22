## 程序实现部分

### 执行顺序导引（与理论对应）

- 1 计算未投影传播子 → 在 `lattice/generator/perambulator.py` 生成基础矩阵元 `S_{i,j}`, `S_{xa,i}`, `S_{i,xa}`, `S_{xa,yb}`（支撑 4.4–4.5、5.1–5.3、6.1–6.4）
- 1.5 计算 current_elemental（元算符）→ 在 `lattice/generator/elemental.py` 生成算符矩阵元 `O_{i,j}`, `O_{i,xa}`, `O_{xa,i}`, `O_{xa,yb}`（支撑 3.1、4.2、4.4、4.5）
- 2 读取未投影传播子并按需投影至 highmode → 在读取处构造 $\tilde{S}$（对应 5.1–5.3）
- 2.5 读取 current_elemental 数据 → 在 `Current` 类中加载并缓存算符矩阵元（对应 3.1、4.2、4.4、4.5）
- 3 夸克图（`QuarkDiagram`）
  - 3a `expand_with_current` → 生成 4^n 个 LL/LH/HL/HH 场景（对应 6.1–6.4、6.5）
  - 3b 采样得到 `unify_vertex_point_color_indices` → 施加点位重合约束（对应 6.2-sampled、6.2-independent、7.1、7.2）
  - 3c 确认展开系数 → 计算并附着 `scene_weight`（对应 6.2-sampled、6.2-independent、7.1、7.2）
- 4 运行时缩并 → 在 `compute_diagrams_multitime` 将各 sub-diagrams 按 `scene_weight` 加权求和，得到 `Tr[O_{1,low} S O_2 S]`（对应 6.1–6.4）

## 程序实现：未投影传播子（汇总与形状）

- 对应步骤 1：计算未投影传播子。
- 代码位置：在 `lattice/generator/perambulator.py` 中生成四类未投影传播子；本文与代码命名与形状如下（timeslice 版本）：
- VSV（本征→本征）

  - 数学：`S_{i,j} = ⟨ξ_i | S | ξ_j⟩`
  - 形状：`(Lt, Ns, Ns, Ne_snk, Ne_src)`
- PSV（点→本征）

  - 数学：`S_{xa,i} = ⟨η_x,a | S | ξ_i⟩`
  - 形状：`(Lt, Ns, Ns, Np_snk, Nc, Ne_src)`
- VSP（本征→点）

  - 数学：`S_{i,xa} = ⟨ξ_i | S | η_x,a⟩`
  - 形状：`(Lt, Ns, Ns, Ne_snk, Np_src, Nc)`
- PSP（点→点）

  - 数学：`S_{xa,yb} = ⟨η_x,a | S | η_y,b⟩`
  - 形状：`(Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc)`

说明：以上形状以实现中的 timeslice 布局为准；若使用时间序列版本（`t_sink` 或 `t_source` 至少一端为时间序列），最前将出现额外的 `Lt` 维度。带投影的高模传播子记为 `\tilde{S}`（代码名 `S_highmode`），由未投影 `S` 与 `(I - I_{low})` 按前述公式构造：如
`\tilde{S}_{xa,i} = S_{xa,i} - \sum_{j=1}^{N_v} M_{xj,a} S_{j,i}`、
`\tilde{S}_{i,xa} = S_{i,xa} - \sum_{j=1}^{N_v} S_{i,j} M_{jx,a}^*`、
`\tilde{S}_{xa,yb} = S_{xa,yb} - \sum_i M_{xi,a} S_{yb,i}^* - \sum_j S_{xa,j} M_{jy,b}^* + \sum_{i,j} M_{xi,a} S_{i,j} M_{jy,b}^*`。

## 程序实现：current_elemental（元算符）计算

- 对应步骤 1.5：计算 current_elemental（元算符）数据。
- 代码位置：在 `lattice/generator/elemental.py` 中定义 `CurrentElementalGenerator` 类，通过 `calc_all()` 方法生成四类算符矩阵元；本文与代码命名与形状如下（timeslice 版本）：

**V2V（本征→本征）**：

- 数学：$(O)_{i,j} = \langle \xi_i| \mathcal{O} |\xi_j\rangle$（对应公式 3.1）
- 形状：`(num_disp, num_momentum, Lt, Ne, Ne)`
- 说明：依赖动量，需要计算所有动量组合

**V2P（本征→点）**：

- 数学：$(O)_{i,xa} = \langle \xi_i| \mathcal{O} |\eta_x,a\rangle$（对应公式 4.4）
- 形状：`(num_disp, Lt, Ne, Np, Nc)`
- 说明：不依赖动量（momentum-independent），仅依赖位移
- **对称性**：与 P2V 存在对称关系 $(O)_{i,xa}(\text{disp}) \approx (O)_{xa,i}(-\text{disp})^T$

**P2V（点→本征）**：

- 数学：$(O)_{xa,j} = \langle \eta_x,a| \mathcal{O} |\xi_j\rangle$（对应公式 4.5）
- 形状：`(num_disp, Lt, Np, Nc, Ne)`
- 说明：不依赖动量（momentum-independent），仅依赖位移
- **对称性**：与 V2P 存在对称关系，可通过反向位移和转置相互计算

**P2P（点→点）**：

- 数学：$(O)_{xa,yb} = \langle \eta_x,a| \mathcal{O} |\eta_y,b\rangle$（对应公式 4.2）
- 形状：稀疏存储（sparse dict），每个时间片为列表，每个元素对应一个 `disp_idx`
- 说明：稀疏存储格式，仅保存满足 `right = left + disp` 的有效点对

**计算流程**：

通过 `CurrentElementalGenerator.calc_all(t)` 方法统一计算四种类型，详见"算符矩阵元的计算程序"章节（在 `Current` 类说明中）。

**存储格式**：

- V2V：存储为 dense numpy 数组（`.npy` 文件）
- P2V：存储为 dense numpy 数组（`.npy` 文件）
- V2P：**可选存储**，推荐通过 P2V 对称性计算（节省约 50% 存储空间）
- P2P：存储为稀疏格式（`.npz` 文件），包含 `indices` 和 `values` 数组

**存储优化说明**：由于 V2P 与 P2V 的对称性，实际应用中可以只存储 P2V 数据，V2P 通过反向位移和转置操作从 P2V 计算得到，从而减少磁盘占用和 I/O 开销。

## 程序实现：传播子读取与高模投影

- 对应步骤 2：读取未投影传播子并按需投影至 highmode。
- 代码位置：在 `lattice/quark_diagram.py` 中定义 `PropagatorWithCurrent` 类，支持四类传播子及其高模投影。

### `PropagatorWithCurrent` 类

**类定义**：`lattice/quark_diagram.py::PropagatorWithCurrent`

```python
class PropagatorWithCurrent(Propagator):
    def __init__(
        self,
        vsv: Perambulator = None,
        vsp: PropagatorVSP = None,
        psv: PropagatorPSV = None,
        psp: PropagatorPSP = None,
        overlap_matrix: "OverlapMatrix" = None,
        Lt: int = None,
        debug: bool = False,
    ):
```

**初始化参数**：

- `vsv`：VSV 传播子数据对象（`Perambulator`）
- `vsp`：VSP 传播子数据对象（`PropagatorVSP`）
- `psv`：PSV 传播子数据对象（`PropagatorPSV`）
- `psp`：PSP 传播子数据对象（`PropagatorPSP`）
- `overlap_matrix`：OverlapMatrix 对象（用于高模投影，包含本征向量和点源信息）
- `Lt`：时间维度
- `debug`：调试开关

**加载数据**：`PropagatorWithCurrent.load(key, usedNe=Ne, usedNp=Np)`

加载指定 `key` 的传播子数据，`usedNe` 和 `usedNp` 控制使用的本征向量和点源数量。该方法会加载所有可用的传播子类型（VSV、VSP、PSV、PSP）以及 overlap matrix 数据。

### 未投影传播子读取接口

本文统一约定：**source = 右端（ket）**，**sink = 左端（bra）**，与 `S_{left_indices,right_indices}` 记号一致。

**VSV 传播子**：`get(t_source, t_sink)`

- 数学：$S_{i,j} = \langle \xi_i| S |\xi_j\rangle$
- 约定：source=右端 $j$（eigenvector），sink=左端 $i$（eigenvector）
  - 返回形状：单时间片 `(Ns, Ns, Ne_snk, Ne_src)`；时间序列 `(Lt, Ns, Ns, Ne_snk, Ne_src)`

**VSP 传播子**：`get_VSP(t_source, t_sink)`

- 数学：$S_{i,xa} = \langle \xi_i| S |\eta_x,a\rangle$
- 约定：source=右端 $xa$（point），sink=左端 $i$（eigenvector）
  - 返回形状：单时间片 `(Ns, Ns, Ne_snk, Np_src, Nc)`；时间序列 `(Lt, Ns, Ns, Ne_snk, Np_src, Nc)`

**PSV 传播子**：`get_PSV(t_source, t_sink, cache=True)`

- 数学：$S_{xa,i} = \langle \eta_x,a| S |\xi_i\rangle$
- 约定：source=右端 $i$（eigenvector），sink=左端 $xa$（point）
  - 返回形状：单时间片 `(Ns, Ns, Np_snk, Nc, Ne_src)`；时间序列 `(Lt, Ns, Ns, Np_snk, Nc, Ne_src)`
- `cache` 参数：控制是否缓存 timeslice（默认 `True`）；在高模投影方法内部调用时通常传 `False` 以节省显存

**PSP 传播子**：`get_PSP(t_source, t_sink, cache=True)`

- 数学：$S_{xa,yb} = \langle \eta_x,a| S |\eta_y,b\rangle$
- 约定：source=右端 $yb$（point），sink=左端 $xa$（point）
  - 返回形状：单时间片 `(Ns, Ns, Np_snk, Nc, Np_src, Nc)`；时间序列 `(Lt, Ns, Ns, Np_snk, Nc, Np_src, Nc)`
- `cache` 参数：同上

### 高模投影传播子接口

记号为 $\tilde{S}$（代码名 `*_highmode`），通过投影算符 $(I - I_{\text{low}})$ 作用于未投影传播子得到。

**VSP 高模投影**：`PropagatorWithCurrent.get_VSP_highmode(t_source, t_sink, usedNe_source=None)`

- 数学（对应公式 5.3）：$\tilde{S}_{i,xa} = S_{i,xa} - \sum_j S_{i,j} M_{jx,a}^*$
- 参数：
  - `t_source`：源时间（int 或 array）
  - `t_sink`：汇时间（int 或 array）
  - `usedNe_source`：源端本征向量数量（默认 `self.usedNe`）；为 0 时返回未投影结果
- 返回形状：同 `get_VSP`
- 缓存策略：当 `usedNe_source == self.usedNe` 且单时间片时，缓存整条 timeslice

**PSV 高模投影**：`PropagatorWithCurrent.get_PSV_highmode(t_source, t_sink, usedNe_sink=None)`

- 数学（对应公式 5.2）：$\tilde{S}_{xa,i} = S_{xa,i} - \sum_j M_{xj,a} S_{j,i}$
- 参数：
  - `t_source`：源时间（int 或 array）
  - `t_sink`：汇时间（int 或 array）
  - `usedNe_sink`：汇端本征向量数量（默认 `self.usedNe`）；为 0 时返回未投影结果
- 返回形状：同 `get_PSV`
- 缓存策略：当 `usedNe_sink == self.usedNe` 且单时间片时，缓存整条 timeslice
- 内部调用：调用 `get_PSV(..., cache=False)` 避免不必要缓存

**PSP 高模投影**：`PropagatorWithCurrent.get_PSP_highmode(t_source, t_sink, usedNe_sink=None, usedNe_source=None)`

- 数学（对应公式 5.1）：$\tilde{S}_{xa,yb} = S_{xa,yb} - \sum_i M_{xi,a} \tilde{S}_{i,yb} - \sum_j S_{xa,j} M_{jy,b}$
- 参数：
  - `t_source`：源时间（int 或 array）
  - `t_sink`：汇时间（int 或 array）
  - `usedNe_sink`：汇端本征向量数量（默认 `self.usedNe`）
  - `usedNe_source`：源端本征向量数量（默认 `self.usedNe`）
  - 当两者均为 0 时返回未投影结果
- 返回形状：同 `get_PSP`
- 内部调用：
  - `get_PSP(..., cache=False)` - 未投影 PSP
  - `get_PSV(..., cache=False)` - 用于计算第三项
  - `get_VSP_highmode(...)` - 用于计算第二项

**Overlap Matrix**：`PropagatorWithCurrent._get_overlap_matrix()`

- 数学：$M_{xi,a} = \langle \eta_x, a| \xi_i\rangle$（转移矩阵）
- 返回形状：`(Lt, Ne, Np, Nc)`
- 说明：从已加载的 `overlap_matrix_data` 中提取数据；`key` 改变时在 `load()` 中重新加载

**形状与索引约定**：

- 所有返回的最后若干维度顺序与其数学右端（source）/左端（sink）含义一致
- 锚定哪个时间（source/sink）为 int 决定内部缓存选择与切片方式，但不影响矩阵元左右的物理含义

**使用示例**：

```python
from lattice.quark_diagram import PropagatorWithCurrent

# Initialize propagator with all four types
propagator = PropagatorWithCurrent(
    vsv=perambulator_vsv,
    vsp=perambulator_vsp,
    psv=perambulator_psv,
    psp=perambulator_psp,
    overlap_matrix=overlap_matrix_obj,
    Lt=64,
    debug=False
)

# Load data for specific configuration
propagator.load(key="conf_0100", usedNe=64, usedNp=216)

# Get VSV (standard)
S_vsv = propagator.get(t_source=0, t_sink=10)  # shape: (Ns, Ns, Ne, Ne)

# Get VSP with high mode projection
S_vsp_tilde = propagator.get_VSP_highmode(t_source=0, t_sink=10, usedNe_source=64)  # shape: (Ns, Ns, Ne, Np, Nc)

# Get PSV with high mode projection
S_psv_tilde = propagator.get_PSV_highmode(t_source=0, t_sink=10, usedNe_sink=64)  # shape: (Ns, Ns, Np, Nc, Ne)

# Get PSP with high mode projection
S_psp_tilde = propagator.get_PSP_highmode(t_source=0, t_sink=10, usedNe_sink=64, usedNe_source=64)  # shape: (Ns, Ns, Np, Nc, Np, Nc)
```

### `Current` 类（顶点算符）

**类定义**：`lattice/quark_diagram.py::Current`

```python
class Current(Meson):
    def __init__(
        self,
        elemental,
        operator,
        source,
        v2p_data: "CurrentElementalV2P" = None,
        p2v_data: "CurrentElementalP2V" = None,
        p2p_data: "CurrentElementalP2P" = None,
        debug: bool = False,
    ):
```

**说明**：`Current` 继承自 `Meson`，扩展支持点采样（point）算符。提供四种算符矩阵元类型（V2V、V2P、P2V、P2P），对应公式 3.1、4.4、4.5、4.2。

**初始化参数**：

- `elemental`：elemental 数据对象（用于 V2V 算符）
- `operator`：算符定义（gamma 结构与导数）
- `source`：是否为 source（dagger）
- `v2p_data`：V2P 算符的 elemental 数据对象（可选）
- `p2v_data`：P2V 算符的 elemental 数据对象（可选）
- `p2p_data`：P2P 算符的 elemental 数据对象（可选）
- `debug`：调试开关

**加载数据**：`Current.load(key, usedNe=Ne, usedNp=Np)`

**主要方法与返回形状**：

**V2V 算符**：`Current.get(t)`

- 数学（对应公式 3.1）：$(O)_{i,j} = \langle \xi_i| \mathcal{O} |\xi_j\rangle$（Low mode 算符矩阵元）
- 参数：`t` 为时间（int 或 array）
- 返回形状：单时间片 `(Ns, Ns, Ne, Ne)`；时间序列 `(t_len, Ns, Ns, Ne, Ne)`

**V2P 算符**：`Current.get_v2p(t)`

- 数学（对应公式 4.4）：$(O)_{i,xa} = \langle \xi_i| \mathcal{O} |\eta_x,a\rangle$（Low-High 混合项）
- 参数：`t` 为时间（int 或 array）
- 返回形状：单时间片 `(Ns, Ns, Ne, Np, Nc)`；时间序列 `(t_len, Ns, Ns, Ne, Np, Nc)`

**P2V 算符**：`Current.get_p2v(t)`

- 数学（对应公式 4.5）：$(O)_{xa,j} = \langle \eta_x,a| \mathcal{O} |\xi_j\rangle$（High-Low 混合项）
- 参数：`t` 为时间（int 或 array）
- 返回形状：单时间片 `(Ns, Ns, Np, Nc, Ne)`；时间序列 `(t_len, Ns, Ns, Np, Nc, Ne)`

**P2P 算符**：`Current.get_p2p(t)`

- 数学（对应公式 4.2）：$(O)_{xa,yb} = \langle \eta_x,a| \mathcal{O} |\eta_y,b\rangle$（High mode 算符矩阵元）
- 参数：`t` 为时间（int，时间序列暂未实现）
- 返回形状：单时间片 `(Ns, Ns, Np, Nc, Np, Nc)`
- 说明：内部使用稀疏存储（sparse dict），对位移为零的项存储为 `{None: coeff}`，表示单位矩阵

**算符矩阵元的计算程序**：

上述四种算符矩阵元（V2V、V2P、P2V、P2P）通过 `CurrentElementalGenerator.calc_all()` 方法统一计算。

**类定义**：`lattice/generator/elemental.py::CurrentElementalGenerator`

**方法签名**：`CurrentElementalGenerator.calc_all(t: int) -> Dict[str, Array]`

**功能说明**：

`calc_all` 方法同时计算 v2v、v2p、p2v、p2p 四种类型的 elemental 数据，通过重用 `gauge_link_product` 提高计算效率。

**计算流程**：

1. **初始化结果数组**：

   - `result_v2v`：形状 `[num_disp, num_momentum, usedNe, usedNe]`（依赖动量）
   - `result_v2p`：形状 `[num_disp, usedNe, usedNp, Nc]`（不依赖动量）
   - `result_p2v`：形状 `[num_disp, usedNp, Nc, usedNe]`（不依赖动量）
   - `result_p2p`：列表，每个元素为稀疏字典，长度 `num_disp`
2. **对每个位移索引 `disp_idx` 循环**：

   **a) 计算 V2V（eigenvector → eigenvector）**：

   - 计算 `gauge_link_product`：在整个格点上计算规范场链乘积，形状 `[Lz, Ly, Lx, Nc, Nc]`
   - 对 V 应用位移：`shift_V = roll(V, -disp)`（周期边界条件）
   - 对每个动量，计算缩并：$\sum_{x,y,z} \text{phase}(x,y,z) \cdot V^*(x,y,z) \cdot \text{gauge_link_product}(x,y,z) \cdot \text{shift_V}(x,y,z)$
   - 结果形状：`[num_momentum, usedNe, usedNe]`

   **b) 计算 V2P（eigenvector → point）**：

   - 重用步骤 a) 中计算的 `gauge_link_product`
   - 应用反向位移：`point_shifted = point - disp`（周期边界条件）
   - 提取本征向量在位移后的点位置：`V_at_points`，形状 `[usedNe, usedNp, Nc]`
   - 提取 `gauge_link_product` 在 `point_shifted` 位置的值：`gauge_link_at_points`，形状 `[usedNp, Nc, Nc]`
   - 应用反向路径：`gauge_link_at_points = gauge_link_at_points.transpose(0, 2, 1).conj()`（对应 $U_N^\dagger \cdots U_2^\dagger U_1^\dagger$）
   - 计算：`result = contract("pab,epb->epa", gauge_link_at_points, V_at_points)`，形状 `[usedNe, usedNp, Nc]`

   **c) 计算 P2V（point → eigenvector）**：

   - 重用步骤 a) 中计算的 `gauge_link_product`
   - 应用正向位移：`point_shifted = point + disp`（周期边界条件）
   - 提取本征向量在位移后的点位置：`V_at_points`，形状 `[usedNe, usedNp, Nc]`
   - 提取 `gauge_link_product` 在原始点位置的值：`gauge_link_at_points`，形状 `[usedNp, Nc, Nc]`
   - 应用正向路径：直接使用 `gauge_link_at_points`（对应 $U_N \cdots U_2 U_1$）
   - 计算：`result = contract("pab,epb->pae", gauge_link_at_points, V_at_points)`，形状 `[usedNp, Nc, usedNe]`
   - 注意：当 `gauge_link_product` 为 `None` 时，直接转置 `V_at_points` 为 `[usedNp, Nc, usedNe]`

   **d) 计算 P2P（point → point）**：

   - 重用步骤 a) 中计算的 `gauge_link_product`
   - 特殊情况：位移为零时返回 `{"type": "identity"}`
   - 计算期望的终点位置：`expected_final = point_left + disp`（周期边界条件）
   - 查找满足 `point_right == expected_final` 的有效点对 `(l, r)`，即 `r = l + disp`
   - 对每个有效点对 `(l, r)`，提取 `gauge_link_product` 在 `point_left[l]` 位置的值：`gauge_link_val`，形状 `[Nc, Nc]`
   - 稀疏存储：`{"type": "sparse", "indices": [[l, r], ...], "values": [gauge_link_val, ...]}`
   - 注意：当 `gauge_link_product` 为 `None` 时，使用单位矩阵 `eye(Nc)`
3. **返回结果字典**：

   ```python
   {
       "v2v": result_v2v,  # [num_disp, num_momentum, usedNe, usedNe]
       "v2p": result_v2p,  # [num_disp, usedNe, usedNp, Nc]
       "p2v": result_p2v,  # [num_disp, usedNp, Nc, usedNe]
       "p2p": result_p2p,  # List of sparse dicts
   }
   ```

**优化策略**：

- 通过先计算 `gauge_link_product` 并重用，避免重复计算规范场链乘积
- 使用向量化操作提取本征向量在点位置的值
- P2P 使用稀疏存储，仅保存有效的点对，节省内存
- **V2P 与 P2V 对称性**：虽然 `calc_all` 同时计算 V2P 和 P2V，但在实际存储时可以只保存 P2V 数据。V2P 可以通过 P2V 的反向位移和转置计算得到，从而减少磁盘存储和 I/O 负担

**使用示例**：

```python
from lattice.generator.elemental import CurrentElementalGenerator

# Initialize generator
current_elemental = CurrentElementalGenerator(
    latt_size=[Lx, Ly, Lz, Lt],
    gauge_field=gauge_field_obj,
    eigenvector=eigenvector_obj,
    point=point_source_obj,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=Ne,
    usedNp=Np,
    debug=False
)

# Load data
current_elemental.load(key="conf_0100")
current_elemental.stout_smear(20, 0.12)

# Calculate all elemental types for time slice t
results = current_elemental.calc_all(t=10)
# results["v2v"]: [num_disp, num_momentum, usedNe, usedNe]
# results["v2p"]: [num_disp, usedNe, usedNp, Nc]
# results["p2v"]: [num_disp, usedNp, Nc, usedNe]
# results["p2p"]: List of sparse dicts
```

**使用示例**：

```python
from lattice.quark_diagram import Current

# Initialize current vertex
current = Current(
    elemental=elemental_obj,
    operator=operator_def,
    source=True,  # dagger
    v2p_data=v2p_data_obj,
    p2v_data=p2v_data_obj,
    p2p_data=p2p_data_obj,
    debug=False
)

# Load data
current.load(key="conf_0100", usedNe=64, usedNp=216)

# Get different operator types
O_vv = current.get(t=10)           # V2V: (Ns, Ns, Ne, Ne)
O_vp = current.get_v2p(t=10)       # V2P: (Ns, Ns, Ne, Np, Nc)
O_pv = current.get_p2v(t=10)       # P2V: (Ns, Ns, Np, Nc, Ne)
O_pp = current.get_p2p(t=10)       # P2P: (Ns, Ns, Np, Nc, Np, Nc)
```

## 程序实现：current_elemental（元算符）读取

- 对应步骤 2.5：读取 current_elemental 数据。
- 代码位置：在 `lattice/preset.py` 中定义数据加载类，在 `lattice/quark_diagram.py` 中定义 `Current` 类，支持四类算符矩阵元的读取。

### 数据加载类

**类定义**：`lattice/preset.py::CurrentElementalV2P`, `CurrentElementalP2V`, `CurrentElementalP2P`

**CurrentElementalV2P**：

```python
class CurrentElementalV2P(NdarrayTimeslicesFile):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],  # [Lt, num_disp, Ne, Np, Nc]
        Ne: int,
        Np: int,
    ):
```

- **功能**：加载 V2P elemental 数据
- **存储格式**：每个时间片单独存储为 `.t{t:03d}.v2p.npy`
- **数据形状**：`[Lt, num_disp, Ne, Np, Nc]`（不依赖动量）
- **加载方法**：`load(key: str)` → 返回形状 `[Lt, num_disp, Ne, Np, Nc]`

**CurrentElementalP2V**：

```python
class CurrentElementalP2V(NdarrayTimeslicesFile):
    def __init__(
        self,
        prefix: str,
        suffix: str,
        shape: List[int],  # [Lt, num_disp, Np, Nc, Ne]
        Ne: int,
        Np: int,
    ):
```

- **功能**：加载 P2V elemental 数据
- **存储格式**：每个时间片单独存储为 `.t{t:03d}.p2v.npy`
- **数据形状**：`[Lt, num_disp, Np, Nc, Ne]`（不依赖动量）
- **加载方法**：`load(key: str)` → 返回形状 `[Lt, num_disp, Np, Nc, Ne]`

**CurrentElementalP2P**：

```python
class CurrentElementalP2P:
    def __init__(self, prefix: str, suffix: str = None):
    def load(self, key: str, t: int):
```

- **功能**：加载 P2P elemental 数据（稀疏格式）
- **存储格式**：每个时间片单独存储为 `.t{t:03d}.p2p.npz`（numpy savez 格式）
- **数据格式**：列表，每个元素为字典：
  - `{"type": "identity"}`：表示单位矩阵（位移为零）
  - `{"type": "sparse", "indices": [N, 2], "values": [N, Nc, Nc]}`：稀疏存储的有效点对
- **加载方法**：`load(key: str, t: int)` → 返回列表，长度为 `num_disp * num_momentum`

### V2P 与 P2V 的对称性优化

**数学原理**：

V2P 和 P2V 算符矩阵元之间存在对称性关系：

$$
(O)_{i,xa}(\text{disp}) = (O)_{xa,i}(-\text{disp})^T
$$

其中：

- 左侧为 V2P 算符在位移 `disp` 下的矩阵元，形状 `[Ne, Np, Nc]`
- 右侧为 P2V 算符在反向位移 `-disp` 下的矩阵元转置，形状 `[Np, Nc, Ne]` → `[Ne, Np, Nc]`

**实现策略**：

为了减少存储空间和提高数据加载效率，代码实现中采用以下优化：

1. **仅存储 P2V 数据**：

   - 在磁盘上只存储 P2V elemental 数据文件（`.t???.p2v.npy`）
   - V2P 数据不单独存储（虽然定义了 `CurrentElementalV2P` 类，但实际可以不生成 v2p 文件）
2. **通过对称性计算 V2P**：

   - 在 `Current._make_cache()` 方法中，从 `p2v_data` 预加载数据
   - 使用反向位移索引和转置操作，通过 P2V 数据计算得到 V2P
   - 公式：`v2p[disp] = p2v[-disp].transpose(Ne, Np)`
   - 在 `_make_cache()` 阶段完成所有算符项的累积，与 v2v 和 p2v 采用相同的预加载模式
3. **预加载和预计算**：

   - 在 `Current._make_cache()` 中，构建位移反向映射表 `_disp_reversal_map`
   - 预加载 P2V 全时间片数据：`p2v_full = self.p2v_data.load(self.key)[:]`
   - 为 V2P 和 P2V 预先计算并累积所有算符项，存储为实际数据数组
   - cache 结构中存储预计算的数据，避免运行时重复计算和 I/O 开销

**cache 结构**：

```python
self.cache = (
    gamma_array,              # cache[0]: gamma 矩阵
    ret_elemental_v2v,        # cache[1]: v2v 预计算数据 [num_parts, Lt, Ne, Ne]
    ret_elemental_v2p,        # cache[2]: v2p 预计算数据 [num_parts, Lt, Ne, Np, Nc] (从p2v对称性计算)
    ret_elemental_p2v,        # cache[3]: p2v 预计算数据 [num_parts, Lt, Np, Nc, Ne]
    ret_elemental_p2p,        # cache[4]: p2p 懒加载指令 (sparse, 按需加载)
)
```

**性能优势**：

- **存储空间**：减少约 50% 的 elemental 数据存储（V2P 和 P2V 形状相同）
- **计算效率**：所有数据在 `_make_cache()` 阶段预加载和预计算，运行时直接从 cache 读取，无 I/O 开销
- **内存占用**：仅加载一份 P2V 数据，通过索引变换服务于 V2P 和 P2V 两种算符
- **一致性**：v2v、v2p、p2v 都遵循相同的预加载模式，提高代码一致性和可维护性

**使用注意**：

- 如果提供了 `v2p_data` 对象，代码仍会尝试加载 V2P 数据文件（向后兼容）
- 如果只提供 `p2v_data`，代码会自动通过对称性计算 V2P（推荐方式）
- 两种方式在数值上完全等价（除舍入误差），但后者更节省资源

### `Current` 类中的数据读取

**加载数据**：`Current.load(key, usedNe=Ne, usedNp=Np)`

该方法执行以下操作：

1. **加载 V2V elemental**：从 `self.elemental` 加载（继承自 `Meson`），形状 `[num_disp, num_momentum, Lt, Ne, Ne]`
2. **构建缓存**：调用 `_make_cache()` 将 elemental 数据与 gamma 矩阵组合，生成最终的算符矩阵元
   - 在 `_make_cache()` 中预加载 P2V 数据：`p2v_full = self.p2v_data.load(self.key)[:]`
   - 为 V2P 和 P2V 预计算并累积所有算符项，存储为实际数据数组（利用 P2V 对称性计算 V2P）
   - 为 P2P 构建懒加载指令（稀疏数据按需加载）

**说明**：V2P 和 P2V 数据都在 `_make_cache()` 阶段预加载和预计算，V2P 通过 P2V 数据和对称性关系计算得到（详见"V2P 与 P2V 的对称性优化"章节）。这与 v2v 采用相同的预加载模式，提高了性能和数据访问的一致性。

**读取接口**：

**V2V 算符**：`Current.get(t)`

- 从 `self.elemental_data` 读取，形状 `[num_disp, num_momentum, Lt, Ne, Ne]`
- 在 `_make_cache()` 中与 gamma 矩阵组合
- 返回形状：单时间片 `(Ns, Ns, Ne, Ne)`；时间序列 `(t_len, Ns, Ns, Ne, Ne)`

**V2P 算符**：`Current.get_v2p(t)`

- **通过 P2V 对称性预计算**：数据在 `_make_cache()` 阶段从 `self.p2v_data` 预加载并计算
- 计算过程（在 `_make_cache()` 中完成）：
  1. 预加载 P2V 全时间片数据：`p2v_full = self.p2v_data.load(self.key)[:]`
  2. 对每个算符项，使用反向位移索引访问 P2V 数据：`reverse_gaugelink_idx = _disp_reversal_map[gaugelink_idx]`
  3. 转置维度：`p2v[reverse_disp].transpose(Ne, Np)` → V2P 形状 `[Lt, Ne, Np, Nc]`
  4. 累积所有算符项，存储到 `cache[2]`：`[num_parts, Lt, Ne, Np, Nc]`
- 运行时：直接从 `cache[2]` 读取预计算数据，与 gamma 矩阵组合
- 返回形状：单时间片 `(Ns, Ns, Ne, Np, Nc)`；时间序列 `(t_len, Ns, Ns, Ne, Np, Nc)`
- **优势**：无需单独存储 V2P 数据，节省存储空间；预计算避免运行时 I/O 和计算开销

**P2V 算符**：`Current.get_p2v(t)`

- **预加载和预计算**：数据在 `_make_cache()` 阶段从 `self.p2v_data` 预加载并累积
- 计算过程（在 `_make_cache()` 中完成）：
  1. 预加载 P2V 全时间片数据：`p2v_full = self.p2v_data.load(self.key)[:]`
  2. 对每个算符项，直接切片并累积：`p2v_full[:, gaugelink_idx, :usedNp, :, :usedNe]`
  3. 累积所有算符项，存储到 `cache[3]`：`[num_parts, Lt, Np, Nc, Ne]`
- 运行时：直接从 `cache[3]` 读取预计算数据，与 gamma 矩阵组合
- 返回形状：单时间片 `(Ns, Ns, Np, Nc, Ne)`；时间序列 `(t_len, Ns, Ns, Np, Nc, Ne)`
- **优势**：预计算避免运行时 I/O 开销，与 v2v 和 v2p 采用相同的预加载模式

**P2P 算符**：`Current.get_p2p(t)`

- 从 `self.p2p_data.load(key, t)` 按需加载（稀疏格式）
- 在 `_make_cache()` 中与 gamma 矩阵组合
- 返回形状：单时间片 `(Ns, Ns, Np, Nc, Np, Nc)`
- 说明：内部使用稀疏存储，对位移为零的项存储为 `{None: coeff}`，表示单位矩阵

**使用示例**：

```python
from lattice.preset import CurrentElementalP2V, CurrentElementalP2P
from lattice.quark_diagram import Current

# Initialize data loaders
# Note: V2P data loader not needed - computed from P2V via symmetry
p2v_data = CurrentElementalP2V(
    prefix="/path/to/p2v/",
    suffix=".t???.p2v.npy",
    shape=[Lt, num_disp, Np, Nc, Ne],
    Ne=Ne,
    Np=Np
)

p2p_data = CurrentElementalP2P(
    prefix="/path/to/p2p/",
    suffix=".t???.p2p.npz"
)

# Initialize Current with data loaders
# v2p_data=None (or omitted) - V2P computed from P2V automatically
current = Current(
    elemental=elemental_obj,  # For V2V
    operator=operator_def,
    source=True,
    v2p_data=None,  # Not needed - computed via symmetry
    p2v_data=p2v_data,
    p2p_data=p2p_data,
    debug=False
)

# Load data (in _make_cache(): pre-loads P2V, pre-computes V2P via symmetry)
current.load(key="conf_0100", usedNe=64, usedNp=216)

# Get operator matrix elements (all pre-computed in _make_cache(), no runtime I/O)
O_vv = current.get(t=10)      # V2V: (Ns, Ns, Ne, Ne) - pre-loaded
O_vp = current.get_v2p(t=10)  # V2P: (Ns, Ns, Ne, Np, Nc) - pre-computed from P2V
O_pv = current.get_p2v(t=10)  # P2V: (Ns, Ns, Np, Nc, Ne) - pre-loaded
O_pp = current.get_p2p(t=10)  # P2P: (Ns, Ns, Np, Nc, Np, Nc) - lazy loaded (sparse)
```

## 程序实现：夸克图展开（`QuarkDiagram`）

- 对应步骤 3a：`expand_with_current` 展开算符分量并生成缩并模式
- 对应步骤 3b/3c：采样场景枚举与权重计算（详见附录 A）
- 代码位置：在 `lattice/quark_diagram.py` 中定义夸克图数据结构

### `QuarkDiagram` 类初始化

**类定义**：`lattice/quark_diagram.py::QuarkDiagram`

```python
class QuarkDiagram:
    def __init__(
        self,
        adjacency_matrix,
        vertex_list: List[int] = None,
        debug: bool = False,
        label_set: str = "ascii",
    ):
```

**初始化参数**：

- `adjacency_matrix`：邻接矩阵，`adjacency_matrix[i][j]` 表示从顶点 `i`（source）到顶点 `j`（sink）的传播子编号（0 表示无连接）
- `vertex_list`：标记哪些顶点是"流算符"（current vertex）；非零值表示 current，序号相同的 vertex 属于同一采样组，序号为 0 表示普通 meson
- `debug`：调试开关
- `label_set`：Einstein 求和标签集（`"ascii"`/`"greek"`/`"hebrew"`）

**内部字段**：

- `operands`：每个缩并组的 `[propagator_list, vertex_list]`
- `subscripts`：对应的 Einstein 求和字符串
- `propagator_types`：每个传播子的类型列表（`"VSV"`/`"VSP"`/`"PSV"`/`"PSP"`，**不带 `_highmode` 后缀**）
- `vertex_types`：每个顶点的类型列表（`"V2V"`/`"V2P"`/`"P2V"`/`"P2P"`）
- `sampling_groups`：采样组字典 `{group_id: [(vertex_idx, "left"|"right"), ...]}`
- `scene_weights`：场景权重列表
- `scene_constraints`：场景约束列表
- `expanded_diagrams`：展开后的 sub-diagram 列表

**自动分析**：`QuarkDiagram.analyse()`

初始化时自动调用 `analyse()` 方法：

- 若 `vertex_list` 包含 current vertex（非零值），调用 `expand_with_current()` 生成 $4^n$ 种组合
- 否则调用 `analyse_v2v()` 使用标准 V2V 分析

### `expand_with_current` 方法（步骤 3a）

**方法签名**：`QuarkDiagram.expand_with_current(current_vertices: List[int]) -> List["QuarkDiagram"]`

**功能说明**：

对 $n$ 个 current vertex，生成 $4^n$ 种状态组合，每个顶点独立选择左端（source）和右端（sink）的状态：

- `(v,v)`：对应 $\mathcal{O}_{\text{low}}$（公式 3.1），矩阵元 $(O)_{i,j}$
- `(p,p)`：对应 $\mathcal{O}_{\text{high}}$（公式 4.2），矩阵元 $(O)_{xa,yb}$ with $|\tilde{\eta}\rangle$
- `(v,p)`：对应 $\mathcal{O}_{\text{(low-high)}}$（公式 4.4），矩阵元 $(O)_{i,yb}$
- `(p,v)`：对应 $\mathcal{O}_{\text{(high-low)}}$（公式 4.5），矩阵元 $(O)_{xa,j}$

**关键步骤**：

1. 为每种状态组合创建独立的 `QuarkDiagram` 实例
2. 调用 `_analyse_with_states(state_dict, current_vertices)` 分析连通分量并构建下标
3. 调用 `_collect_sampling_groups(state_dict, current_vertices)` 收集采样组信息
4. 调用 `_generate_point_scenes()` 枚举点位重合场景并计算权重

**重要约定**：

- **`propagator_types` 存储格式**：不带 `_highmode` 后缀，仅存储 `"VSV"`、`"VSP"`、`"PSV"`、`"PSP"`
- **高模投影判断**：在 `compute_diagrams_multitime` 中根据类型自动调用 `get_*_highmode` 方法
- **采样参数**：`_generate_point_scenes()` 当前使用占位符 `M=1000, N=10`，实际运行时 **M 应等于 L^3，N 应等于 usedNp**（TODO：需在调用前传入或自动提取）

**存储结果**：

- `self.expanded_diagrams`：包含所有状态组合 × 点位重合场景的 sub-diagram 列表
- 每个 sub-diagram 携带 `scene_weight` 和 `scene_constraints`

3. **分析连通分量**（`QuarkDiagram._analyse_with_states`）：

   - 对邻接矩阵做广度优先搜索（BFS），找出所有连通的传播子分组（contraction group）
   - 每个分组对应 $Tr[\cdots]$ 中的一次张量缩并
4. **构建下标与类型**（`QuarkDiagram._build_subscripts_with_types`）：

   **a) 基本约定**：

   - 邻接矩阵 `adjacency_matrix[src][snk]` 表示从 `src`（source）到 `snk`（sink）的传播子
   - source（src）= 传播子右端（ket）
   - sink（snk）= 传播子左端（bra）
   - 传播子类型命名为 `S_{sink类型, source类型}` = `S_{左端, 右端}`

   **b) propagator_types 与 vertex_types 确定**（详见附录 B）：

   - 根据 `state_dict` 确定每条边的传播子类型（在 `_build_subscripts_with_types` 中实现）
   - 根据累积状态确定每个顶点的类型（在 `_build_subscripts_with_types` 中实现）
   - **重要**：`propagator_types` 仅存储基础类型（`"VSV"`/`"VSP"`/`"PSV"`/`"PSP"`），不含 `_highmode` 后缀
   - 高模投影判断延迟至 `compute_diagrams_multitime` 执行时

   **c) 下标构建**：

   - 按 `sink指标 + source指标` 顺序（对应数学记号的左右顺序）
   - 重排：将 spin 指标与其它指标分离（所有 spin 在前，其它在后）

**使用示例**：

```python
from lattice.quark_diagram import QuarkDiagram

# Define topology: O1 S O2 S (2 vertices, 2 propagators)
adjacency_matrix = [
    [0, 0],  # vertex 0 -> vertex 1: propagator 0
    [1, 0]   # vertex 1 -> vertex 0: propagator 1
]
# Mark both vertices as current vertices
vertex_list = [1, 1]  # Same group_id (1) means same sampling set

# Create diagram (automatically calls analyse -> expand_with_current)
diagram = QuarkDiagram(adjacency_matrix, vertex_list=vertex_list, debug=True)

# Result: diagram.expanded_diagrams contains 4^2 = 16 state combinations × point scenes
print(f"Total expanded diagrams: {len(diagram.expanded_diagrams)}")

# Each expanded diagram has:
# - operands, subscripts
# - propagator_types (e.g., ["VSP", "PSV"])
# - vertex_types (e.g., ["V2V", "P2P"])
# - scene_weights (e.g., [M(M-1)/N(N-1)])
# - scene_constraints (e.g., [[(None, None), (0, 1)]])
```

### 状态组合示例

假设邻接矩阵表示 $O_1 S O_2 S$ 拓扑（2 个顶点，2 条边），`vertex_list=[1, 1]` 标记两顶点为 current 且属于同一采样组。`expand_with_current` 生成 $4^2=16$ 种状态组合，典型示例：

- **组合 1**：`vertex[0]=(v,v), vertex[1]=(v,v)`

  - 算符：$O_1 = \mathcal{O}_{\text{low}}$，$O_2 = \mathcal{O}_{\text{low}}$
  - `propagator_types = ["VSV", "VSV"]`（**注意：不含 `_highmode` 后缀**）
  - `vertex_types = ["V2V", "V2V"]`
  - 对应公式 (6.1)
- **组合 2**：`vertex[0]=(v,v), vertex[1]=(p,p)`

  - 算符：$O_1 = \mathcal{O}_{\text{low}}$，$O_2 = \mathcal{O}_{\text{high}}$
  - `propagator_types = ["VSP", "PSV"]`（**注意：不含 `_highmode` 后缀**）
  - `vertex_types = ["V2V", "P2P"]`
  - 对应公式 (6.2)
- **组合 3**：`vertex[0]=(v,v), vertex[1]=(v,p)`

  - 算符：$O_1 = \mathcal{O}_{\text{low}}$，$O_2 = \mathcal{O}_{\text{(low-high)}}$
  - `propagator_types = ["VSV", "PSV"]`
  - `vertex_types = ["V2V", "P2V"]`
  - 对应公式 (6.3)
- **组合 4**：`vertex[0]=(v,v), vertex[1]=(p,v)`

  - 算符：$O_1 = \mathcal{O}_{\text{low}}$，$O_2 = \mathcal{O}_{\text{(high-low)}}$
  - `propagator_types = ["VSP", "VSV"]`
  - `vertex_types = ["V2V", "V2P"]`
  - 对应公式 (6.4)
- …（其余 12 种组合以此类推）

每个组合进一步按点位重合场景拆分（步骤 3b/3c，详见附录 A），最终结果为所有场景的加权和。

## 程序实现：缩并计算（`compute_diagrams_multitime`）

- 对应步骤 4：运行时将拆分得到的 sub-diagrams 按 `scene_weight` 加权相加，汇总为最终 Trace 结果
- 代码位置：在 `lattice/quark_diagram.py` 中实现 $Tr[O_{1,\text{low}} S O_2 S]$ 各项的自动缩并

### 函数签名

**函数定义**：`lattice/quark_diagram.py::compute_diagrams_multitime`

```python
def compute_diagrams_multitime(
    diagrams: List[QuarkDiagram],
    time_list,
    vertex_list: List[Meson],
    propagator_list: List[Propagator],
    multitime_shape: int = False,
    debug: bool = False,
):
```

### 输入参数

- `diagrams`：`QuarkDiagram` 对象列表（通常包含 `expanded_diagrams` 的原始 diagram）
- `time_list`：时间列表，每个元素对应一个顶点的时间（int 或 array）
- `vertex_list`：算符顶点对象列表（`Meson` 或 `Current` 实例）
- `propagator_list`：传播子对象列表（`Propagator` 或 `PropagatorWithCurrent` 实例）
- `multitime_shape`：是否返回时间序列形状（默认 `False`）
- `debug`：调试开关

### 核心流程

**1. 自动展开 diagrams**：

检测每个 `diagram` 是否有 `expanded_diagrams` 属性：

- 若有，将 `expanded_diagrams` 列表中的所有 sub-diagram 加入计算队列
- 若无，将 diagram 本身加入计算队列

```python
diagrams_to_compute = []
for diagram in diagrams:
    if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
        diagrams_to_compute.extend(diagram.expanded_diagrams)
    else:
        diagrams_to_compute.append(diagram)
```

**2. 类型分派与方法调用**：

根据 `propagator_types` 和 `vertex_types` 自动调用对应方法：

**Propagator 类型映射**：

- `"VSV"` → `propagator.get(t_source, t_sink)` → $S_{i,j}$
- `"VSP"` → `propagator.get_VSP_highmode(t_source, t_sink, usedNe_source)` → $\tilde{S}_{i,xa}$（**自动加高模投影**）
- `"PSV"` → `propagator.get_PSV_highmode(t_source, t_sink, usedNe_sink)` → $\tilde{S}_{xa,i}$（**自动加高模投影**）
- `"PSP"` → `propagator.get_PSP_highmode(t_source, t_sink, usedNe_sink, usedNe_source)` → $\tilde{S}_{xa,yb}$（**自动加高模投影**）

**Vertex 类型映射**：

- *`"V2V"` → `vertex.get(t)` → $(O)_{i,j}$*
  - 计算：通过 `CurrentElementalGenerator.calc_all()` 计算 `v2v` elemental，然后在 `Current._make_cache()` 中预加载并与 gamma 矩阵组合
- `"V2P"` → `vertex.get_v2p(t)` → $(O)_{i,xa}$
  - 计算：在 `Current._make_cache()` 中从 `p2v_data` 预加载数据，通过对称性计算 `v2p`（`v2p[disp] = p2v[-disp].transpose(Ne, Np)`），然后与 gamma 矩阵组合
- `"P2V"` → `vertex.get_p2v(t)` → $(O)_{xa,i}$
  - 计算：在 `Current._make_cache()` 中从 `p2v_data` 预加载数据并累积，然后与 gamma 矩阵组合
- `"P2P"` → `vertex.get_p2p(t)` → $(O)_{xa,yb}$
  - 计算：通过 `CurrentElementalGenerator.calc_all()` 计算 `p2p` elemental（稀疏存储），在 `Current.get_p2p(t)` 中按需加载，然后在 `Current._make_cache()` 中与 gamma 矩阵组合

**3. 张量缩并**：

- 拼接最终 Einstein 求和字符串（处理时间序列前缀）
- 调用 `contract(final_subscripts, *operands_data)`

**4. 应用 scene_weight**：

若 sub-diagram 携带 `scene_weights` 属性，将缩并结果乘以权重：

```python
if hasattr(diagram, "scene_weights") and diagram.scene_weights:
    scene_weight = diagram.scene_weights[0]
    diagram_value[-1] = diagram_value[-1] * scene_weight
```

**5. 返回结果**：

- 形状：`[num_diagrams]` 或 `[num_diagrams, Lt]`（取决于是否有时间序列）
- 最终结果为所有 sub-diagrams 的加权和

### 使用示例

```python
from lattice.quark_diagram import QuarkDiagram, compute_diagrams_multitime

# Create diagram
adjacency_matrix = [[0, 0], [1, 0]]
vertex_list = [1, 1]
diagram = QuarkDiagram(adjacency_matrix, vertex_list=vertex_list)

# Prepare vertices and propagators
vertices = [current0, current1]  # Current instances
propagators = [None, propagator0, propagator1]  # PropagatorWithCurrent instances
time_list = [10, 10]  # Both vertices at t=10

# Compute (automatically expands and weights)
results = compute_diagrams_multitime(
    diagrams=[diagram],
    time_list=time_list,
    vertex_list=vertices,
    propagator_list=propagators,
    multitime_shape=False,
    debug=False
)

# results.shape: [num_expanded_diagrams]
# Sum to get final trace value
final_trace = results.sum()
```

### 与公式的对应

- **Low-Low 项**：`propagator_types = ["VSV", "VSV"]`, `vertex_types = ["V2V", "V2V"]`

  - 对应公式 (6.1)：$\sum_{i,j,k,l} (O_1)_{i,j} (O_2)_{k,l} S_{j,k} S_{l,i}$
- **Low-High 项**：`propagator_types = ["VSP", "PSV"]`, `vertex_types = ["V2V", "P2P"]`

  - 对应公式 (6.2)：$\sum_{i,j,x,y,a,b} (O_1)_{i,j} (O_2)_{xa,yb} \tilde{S}_{j,xa} \tilde{S}_{yb,i}$
  - 说明：虽然 `propagator_types` 为 `["VSP", "PSV"]`，但 `compute_diagrams_multitime` 自动调用 `get_VSP_highmode` 和 `get_PSV_highmode`
- **混合项（low-high）**：`propagator_types = ["VSV", "PSV"]`, `vertex_types = ["V2V", "P2V"]`

  - 对应公式 (6.3)：$\sum_{i,j,k,y,b} (O_1)_{i,j} (O_2)_{k,yb} S_{j,k} \tilde{S}_{yb,i}$
- **混合项（high-low）**：`propagator_types = ["VSP", "VSV"]`, `vertex_types = ["V2V", "V2P"]`

  - 对应公式 (6.4)：$\sum_{i,j,x,a,l} (O_1)_{i,j} (O_2)_{xa,l} \tilde{S}_{j,xa} S_{l,i}$

### 实现要点

- **自动展开**：检测 `diagram.expanded_diagrams` 属性，自动展开为所有 sub-diagrams
- **高模投影**：根据 `propagator_types`（不含 `_highmode` 后缀）自动调用 `get_*_highmode` 方法
- **权重应用**：每个 sub-diagram 的结果乘以其 `scene_weight`，最终求和得到完整 Trace
- **时间处理**：时间序列与单时间片通过 `isinstance(time, int)` 自动适配

---
