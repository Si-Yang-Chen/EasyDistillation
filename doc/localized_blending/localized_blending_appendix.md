## 附录

### 附录 A：采样权重自动计算（步骤 3b/3c）

本附录详细说明采样权重的自动计算流程，包括整数分拆、权重计算、场景枚举、约束生成等辅助函数。这些函数在 `expand_with_current` 方法中被调用（第 431 行），用于生成点位重合场景并计算对应的采样补偿权重。

#### 理论背景：一般化补偿系数公式

对于从 $M$ 个点中抽取 $N$ 个点形成的采样集合，若一次计算涉及该集合中的 $r$ 个点位（$1 \leq r \leq N$），则根据这 $r$ 个点位中**互异点的数量** $k$（$1 \leq k \leq r$）确定补偿系数：

$$
w(k) = \frac{\binom{M}{k}}{\binom{N}{k}} = \frac{M!/(M-k)!}{N!/(N-k)!} = \frac{M(M-1)\cdots(M-k+1)}{N(N-1)\cdots(N-k+1)}
\tag{7.1}
$$

其中 $\binom{M}{k}$ 表示从 $M$ 个点中选取 $k$ 个互异点的组合数（下降阶乘形式）。

**特殊情况**：

- $k=1$（1 个点）: $w(1) = \frac{M}{N}$
- $k=2$（2 个互异点）: $w(2) = \frac{M(M-1)}{N(N-1)}$
- $k=r$（全部互异）: $w(r) = \frac{M(M-1)\cdots(M-r+1)}{N(N-1)\cdots(N-r+1)}$

#### 采样组的划分

**定义**：在 `QuarkDiagram` 中，`vertex_list` 的**序号不同**即代表点来自**独立的采样集合**。

- `vertex_list = [0, 1, 1, 2]`：vertex 0 为普通 meson（序号 0，无点采样），vertex 1 和 2 为 current（序号 1，共享采样集），vertex 3 为 current（序号 2，独立采样集）
- 序号相同的 vertex 共享同一采样集合，序号不同的属于独立集合，序号为 0 表示 meson

**实现约定**：

- 每个 vertex 的左端（source, "left"）和右端（sink, "right"）属于相同的采样组
- 当 vertex 状态为 `(p, p)` 时，左右端都需要点采样，且属于同一采样组
- 若 vertex 状态为 `(v, p)` 或 `(p, v)`，则只有一端需要点采样

#### 辅助函数详细说明

以下函数按执行顺序说明（在 `QuarkDiagram.expand_with_current` 方法中调用 `_generate_point_scenes()` 时被使用）。

**1. `integer_partitions(n)` - 整数分拆**

**函数定义**：`lattice/quark_diagram.py::integer_partitions`

```python
def integer_partitions(n: int) -> List[List[int]]:
```

- **功能**：生成 $n$ 的所有整数分拆，每个分拆为降序排列的正整数列表
- **参数**：`n` - 要分拆的正整数
- **返回**：分拆列表，如 `integer_partitions(3) = [[3], [2,1], [1,1,1]]`
- **用途**：枚举 $r$ 个点位的所有重合模式，分拆 `[2,1]` 表示"2个点位共享同一点，1个点位使用另一点"
- **示例**：

```python
>>> integer_partitions(4)
[[4], [3,1], [2,2], [2,1,1], [1,1,1,1]]
```

**2. `calculate_sampling_weight(M, N, k)` - 权重计算**

**函数定义**：`lattice/quark_diagram.py::calculate_sampling_weight`

```python
def calculate_sampling_weight(M: int, N: int, k: int) -> float:
```

- **功能**：计算从 $N$ 个采样点中选取 $k$ 个互异点的采样权重（补偿系数）
- **参数**：
  - `M`：总点数（通常为 $L^3$）
  - `N`：采样点数（`usedNp`）
  - `k`：互异点数量
- **返回**：权重 $w(k) = \frac{M(M-1)\cdots(M-k+1)}{N(N-1)\cdots(N-k+1)}$（对应公式 7.1）
- **特殊情况**：`k=0` 返回 1.0；`k > min(M,N)` 返回 0.0
- **示例**：

```python
>>> calculate_sampling_weight(M=1000, N=100, k=1)
10.0  # M/N
>>> calculate_sampling_weight(M=1000, N=100, k=2)
101.0  # M(M-1)/[N(N-1)]
```

**3. `enumerate_point_scenes(r, M, N)` - 场景枚举**

**函数定义**：`lattice/quark_diagram.py::enumerate_point_scenes`

```python
def enumerate_point_scenes(r: int, M: int, N: int) -> List[Tuple[List[int], float]]:
```

- **功能**：枚举 $r$ 个点位的所有重合场景及其权重
- **参数**：
  - `r`：点位数量
  - `M`：总点数
  - `N`：采样点数
- **返回**：`[(partition, weight), ...]` 列表，其中 `partition` 为整数分拆，`weight` 为对应权重
- **用途**：为同一采样组的 $r$ 个点位生成所有可能的重合模式
- **示例**：

```python
>>> enumerate_point_scenes(r=2, M=1000, N=100)
[([1, 1], 101.0),  # 2 points distinct, weight = M(M-1)/N(N-1)
 ([2], 10.0)]      # 2 points same, weight = M/N
```

**4. `partition_to_constraints(partition, point_positions)` - 约束转换**

**函数定义**：`lattice/quark_diagram.py::partition_to_constraints`

```python
def partition_to_constraints(
    partition: List[int], 
    point_positions: List[Tuple[int, str]]
) -> List[Tuple[int | None, int | None]]:
```

- **功能**：将整数分拆转换为 `unify_vertex_point_color_indices` 的约束格式
- **参数**：
  - `partition`：整数分拆，如 `[2, 1]` 表示前 2 个点位相同，第 3 个不同
  - `point_positions`：点位列表 `[(vertex_idx, "left"|"right"), ...]`
- **返回**：约束列表 `[(left_id, right_id), ...]`，`None` 表示 eigenvector 位置，相同数字表示使用相同点
- **示例**：

```python
>>> partition = [2, 1]  # First 2 positions same, 3rd different
>>> positions = [(1, "left"), (1, "right"), (2, "left")]
>>> partition_to_constraints(partition, positions)
[(None, None), (0, 0), (1, None)]  
# vertex 0: no constraint
# vertex 1: left=0, right=0 (same point)
# vertex 2: left=1 (different point), right=None
```

**5. `QuarkDiagram._collect_sampling_groups(state_dict, current_vertices)` - 采样组收集**

**方法定义**：`QuarkDiagram._collect_sampling_groups`

```python
def _collect_sampling_groups(self, state_dict: Dict, current_vertices: List[int]) -> None:
```

- **功能**：从 `state_dict` 收集需要点采样的 vertex 端点，按 `vertex_list` 分组
- **参数**：
  - `state_dict`：状态字典 `{(vertex_idx, "left"|"right"): "v"|"p"}`
  - `current_vertices`：current vertex 索引列表
- **副作用**：填充 `self.sampling_groups`，格式为 `{group_id: [(vertex_idx, side), ...]}`
- **用途**：在 `expand_with_current` 中调用，为后续场景枚举准备采样组信息
- **示例**：

```python
# Input:
state_dict = {(0, "left"): "v", (0, "right"): "v", 
              (1, "left"): "p", (1, "right"): "p"}
current_vertices = [0, 1]
self.vertex_list = [1, 1]  # Same group

# Output (stored in self.sampling_groups):
{1: [(1, "left"), (1, "right")]}  # Group 1 has 2 point positions from vertex 1
```

**6. `QuarkDiagram._generate_point_scenes(M, N)` - 场景生成主函数**

**方法定义**：`QuarkDiagram._generate_point_scenes`

```python
def _generate_point_scenes(self, M: int = None, N: int = None) -> None:
```

- **功能**：为所有 `expanded_diagrams` 枚举点位重合场景，生成 sub-diagrams 并计算权重
- **参数**：
  - `M`：总点数（默认 1000，**实际应为 $L^3$，TODO：需传入**）
  - `N`：采样点数（默认 10，**实际应为 `usedNp`，TODO：需传入**）
- **副作用**：将 `self.expanded_diagrams` 中的每个 diagram 替换为多个 scene-specific sub-diagrams
- **流程**：
  1. 对每个 diagram 的 `sampling_groups`，调用 `enumerate_point_scenes` 枚举场景
  2. 对多个 group 做笛卡尔积
  3. 为每个场景组合调用 `_build_scene_constraints` 生成约束
  4. 复制 diagram 并附加 `scene_weights` 和 `scene_constraints`
- **调用位置**：在 `expand_with_current` 中调用
- **示例**：

```python
# Before:
self.expanded_diagrams = [diagram_vv_pp]  # 1 diagram with sampling_groups

# After (假设 M=1000, N=100, 2 point positions):
self.expanded_diagrams = [
    diagram_vv_pp_scene_A,  # scene_weight = M(M-1)/N(N-1), constraints = [..., (0,1)]
    diagram_vv_pp_scene_B   # scene_weight = M/N, constraints = [..., (0,0)]
]
```

**7. `QuarkDiagram._build_scene_constraints(partitions_and_positions)` - 多组约束构建**

**方法定义**：`QuarkDiagram._build_scene_constraints`

```python
def _build_scene_constraints(
    self, 
    partitions_and_positions: List[Tuple[List[int], List[Tuple[int, str]]]]
) -> List[Tuple[int | None, int | None]]:
```

- **功能**：为多个采样组的分拆组合构建统一的约束列表
- **参数**：`partitions_and_positions` - `[(partition, positions), ...]` 每个 group 的分拆与点位
- **返回**：约束列表，格式同 `partition_to_constraints`
- **用途**：处理多个独立采样组（不同 `vertex_list` 序号）的约束合并，确保不同组使用不同的 point_id 范围
- **示例**：

```python
# Input: 2 groups
partitions_and_positions = [
    ([1, 1], [(1, "left"), (1, "right")]),  # Group 1: 2 positions, distinct
    ([1], [(2, "left")])                     # Group 2: 1 position
]

# Output:
[(None, None), (0, 1), (2, None)]
# vertex 0: no constraint
# vertex 1: left=0, right=1 (distinct within group 1)
# vertex 2: left=2 (independent group 2, uses different point_id)
```

### 附录 B：下标与约束处理

本附录说明 `QuarkDiagram` 中用于构建 Einstein 求和下标和处理约束的内部方法。

**8. `QuarkDiagram.unify_vertex_point_color_indices(constraints)` - 统一点/颜色下标**

**方法定义**：`QuarkDiagram.unify_vertex_point_color_indices`

```python
def unify_vertex_point_color_indices(
    self, 
    constraints: List[Tuple[int | None, int | None]]
) -> None:
```

- **功能**：根据约束统一指定 vertex 的点和颜色下标
- **参数**：`constraints` - 约束列表，`constraints[i] = (left_id, right_id)` 表示 vertex i 的左右端约束
  - `None`：该端为 eigenvector，无需约束
  - 相同数字：这些端使用相同的点/颜色下标
- **副作用**：修改 `self.subscripts`，将相同 `id` 的点/颜色下标替换为统一符号
- **用途**：在 `_generate_point_scenes` 生成 sub-diagram 后，应用约束以实现点位重合（可能被递归调用）
- **验证**：检查约束与 `vertex_types` 的一致性（eigenvector 端必须为 `None`）
- **示例**：

```python
# constraints = [(None, None), (0, 1)]
# vertex 0: V2V, no constraint
# vertex 1: P2P, left uses point_id=0, right uses point_id=1 (distinct)
# Effect: replace point/color subscripts in vertex 1 to use different symbols
```

**9. `QuarkDiagram._build_subscripts_with_types(...)` - 构建下标与类型**

**方法定义**：`QuarkDiagram._build_subscripts_with_types`

```python
def _build_subscripts_with_types(
    self,
    propagators: List,
    state_dict: Dict,
    current_vertices: List[int],
    contraction_group: int,
) -> None:
```

- **功能**：为一个 contraction group 构建 Einstein 求和下标，并确定 propagator/vertex 类型
- **参数**：
  - `propagators`：传播子列表 `[[prop_id, src, snk], ...]`
  - `state_dict`：状态字典
  - `current_vertices`：current vertex 索引列表
  - `contraction_group`：当前缩并组索引
- **副作用**：
  - 填充 `self.operands[contraction_group]`
  - 填充 `self.subscripts[contraction_group]`
  - 填充 `self.propagator_types[contraction_group]`
  - 填充 `self.vertex_types[contraction_group]`
- **流程**：
  1. 根据 `state_dict` 确定每个传播子的类型（VSV/VSP/PSV/PSP）
  2. 累积每个 vertex 的左右端状态
  3. 推导 vertex 类型（V2V/V2P/P2V/P2P）
  4. 构建下标字符串（spin + eigen/point/color）
  5. 重排下标（spin 在前）
- **调用位置**：在 `_analyse_with_states` 中调用

**10. `QuarkDiagram._analyse_with_states(state_dict, current_vertices)` - 分析连通分量**

**方法定义**：`QuarkDiagram._analyse_with_states`

```python
def _analyse_with_states(
    self, 
    state_dict: Dict, 
    current_vertices: List[int]
) -> None:
```

- **功能**：对给定状态组合分析邻接矩阵的连通分量，生成缩并模式
- **参数**：
  - `state_dict`：状态字典
  - `current_vertices`：current vertex 索引列表
- **副作用**：填充 `self.operands`, `self.subscripts`, `self.propagator_types`, `self.vertex_types`
- **流程**：
  1. 广度优先搜索（BFS）找出所有连通分量
  2. 对每个分量调用 `_build_subscripts_with_types` 构建下标
- **调用位置**：在 `expand_with_current` 中调用

### 附录 C：PropagatorWithCurrent 内部方法

本附录说明 `PropagatorWithCurrent` 中用于高模投影的内部辅助方法。

**11. `PropagatorWithCurrent._get_overlap_matrix()` - 获取转移矩阵**

**方法定义**：`PropagatorWithCurrent._get_overlap_matrix`

```python
def _get_overlap_matrix(self):
```

- **功能**：从已加载的 `overlap_matrix_data` 中提取 overlap matrix $M_{xi,a} = \langle \eta_x, a| \xi_i\rangle$
- **参数**：无（使用 `self.overlap_matrix_data`）
- **返回**：形状 `(Lt, Np, Ne, Nc)`，其中 `Ne = self.usedNe`，`Np = self.usedNp`
- **说明**：数据在 `load()` 方法中从文件加载，存储在 `self.overlap_matrix_data`
- **用途**：在 `get_VSP_highmode`、`get_PSV_highmode`、`get_PSP_highmode` 中调用，用于构造投影算符 $(I - I_{\text{low}})$

**12. `PropagatorWithCurrent._apply_gamma_on_spin(array)` - Gamma 矩阵变换**

**方法定义**：`PropagatorWithCurrent._apply_gamma_on_spin`

```python
def _apply_gamma_on_spin(self, array_with_spin_first_two_axes):
```

- **功能**：对数组的前两个 spin 轴应用 `gamma(15)` 变换并共轭（dagger 操作）
- **参数**：形状为 `[Lt, Ns_left, Ns_right, ...]` 的数组
- **返回**：相同形状的变换后数组
- **用途**：在 `_dagger_vsp/psv/psp` 中调用，实现传播子的 dagger 操作
- **实现**：`gamma(15) @ array.conj() @ gamma(15)`

**13. `PropagatorWithCurrent._dagger_vsp/psv/psp(block)` - 传播子 Dagger 变换**

**方法定义**：`PropagatorWithCurrent._dagger_vsp`, `PropagatorWithCurrent._dagger_psv`, `PropagatorWithCurrent._dagger_psp`

```python
def _dagger_vsp(self, vsp_block):  # VSP -> PSV ordering
def _dagger_psv(self, psv_block):  # PSV -> VSP ordering  
def _dagger_psp(self, psp_block):  # PSP swap left/right
```

- **功能**：对传播子数据应用 dagger 变换（gamma + conjugate + transpose）
- **输入/输出形状**：
  - `_dagger_vsp`：`[..., Ne, Np, Nc]` → `[..., Np, Nc, Ne]`（VSP → PSV 顺序）
  - `_dagger_psv`：`[..., Np, Nc, Ne]` → `[..., Ne, Np, Nc]`（PSV → VSP 顺序）
  - `_dagger_psp`：`[..., Np_snk, Nc, Np_src, Nc]` → `[..., Np_src, Nc, Np_snk, Nc]`（交换左右）
- **用途**：在 two-time 接口中，当锚定时间与数据存储时间不同时，需要调用 dagger 变换
- **实现**：先调用 `_apply_gamma_on_spin`，再 transpose 尾部轴

### 附录 D：Current 类内部实现

本附录说明 `Current` 类中算符矩阵元（V2V/V2P/P2V/P2P）的缓存构建逻辑。

**14. `Current._make_cache()` - 算符缓存构建**

**方法定义**：`Current._make_cache`

```python
def _make_cache(self):
```

- **功能**：为 `operator.parts` 中的所有 gamma 结构和 elemental 部分构建缓存，支持四种矩阵元类型
- **参数**：无（使用 `self.elemental_data`、`self.v2p_data`、`self.p2v_data`、`self.p2p_data` 等字段）
- **缓存结构**：
  - `self.cache`：tuple of 5 元素 `(ret_gamma, ret_elemental_v2v, ret_elemental_v2p, ret_elemental_p2v, ret_elemental_p2p)`
  - `ret_gamma`：gamma 矩阵列表，形状 `[num_parts, Ns, Ns]`
  - `ret_elemental_v2v`：V2V elemental，形状 `[num_parts, Lt, Ne, Ne]`
  - `ret_elemental_v2p`：V2P elemental，形状 `[num_parts, Lt, Ne, Np, Nc]`
  - `ret_elemental_p2v`：P2V elemental，形状 `[num_parts, Lt, Np, Nc, Ne]`
  - `ret_elemental_p2p`：P2P elemental，**稀疏存储**为 dict 列表（见下文）
- **稀疏存储说明**（P2P）：
  - 位移非零时：`{(t, l, r): matrix[Nc, Nc]}` 仅存储有效的 $(t, l, r)$ 对（满足 `right = left + displacement`）
  - 位移为零时：`{None: coeff}` 表示单位矩阵 $\delta_{lr} \delta_{ab} \times$ coeff
- **调用位置**：在 `load()` 方法内部调用

**Gauge Link 处理逻辑**：

算符包含导数项时需要沿规范场路径传播：

- **V2P 方向**：
  - 起点：从 `v2p_data` 加载的位移后的点数据
  - 沿 gauge_list 反向应用 $U^\dagger$，回到原始点
  - 最终：`V[t, e, p, a]` at original point position
- **P2V 方向**：
  - 起点：从 `p2v_data` 加载的位移后的点数据
  - 沿 gauge_list 正向应用 $U$，回到原始点
  - 最终：`V[t, p, e, a]` at original point position
- **P2P 方向**：
  - 预计算满足 `right = left + displacement` 的 $(l, r)$ 对
  - 对每个有效对计算 gauge link product
  - 稀疏存储：仅保存非零项，显著节省显存

**Displacement 处理**：

- `displacement = [dx, dy, dz]`：算符作用的空间位移
- 通过 `GaugeLink(gaugelink_idx).displacement` 获取
- 周期边界条件：所有坐标计算使用模运算 `% lattice_size`
