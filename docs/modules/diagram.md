# 图收缩层：`lattice/quark_diagram.py` 与 `lattice/quark_draw.py`

> 定位说明：本文档描述 EasyDistillation 的 quark 收缩（contraction）层——`lattice/quark_diagram.py` 把 hadron 间 quark 线拓扑表示为"邻接矩阵 + einsum 下标"，并提供 sympy 符号化简与 `opt_einsum.contract` 数值求值两条通路；`lattice/quark_draw.py` 将这些收缩图画为 Feynman 风格示意图。

## ① 模块概览（含依赖关系）

### 1.1 模块职责

| 模块 | 行数 | 职责 |
|---|---|---|
| `lattice/quark_diagram.py` | 3387 | 收缩拓扑表示（`QuarkDiagram` 及状态/场景展开子类）、einsum 下标构造、current 顶点适配、sympy 符号包装（`Diagram`）、表达式级化简/替换工具、数值求值流水线（prepare/bind/eval）、`quark_contract` 味空间收缩 |
| `lattice/quark_draw.py` | 473 | 把邻接矩阵画成 Feynman 风格图（依赖第三方 `feynman` 与 `matplotlib`） |

### 1.2 依赖关系

`quark_diagram.py` 的依赖（`quark_diagram.py:1-21`）：

- 第三方：`numpy`、`opt_einsum.contract`、`sympy`、`hashlib`；
- 仓库内顶层：`lattice.spatial_structure.HadronIrrepRow`（`quark_diagram.py:19`）、`.backend.get_backend`（`quark_diagram.py:21`）；
- 函数体内按需 import：`.flavor_structure` 的 `HadronFlavorStructure/Qurak/Propagator`（`quark_diagram.py:3248-3251`）、`.base_types.Tag`、`.symmetry.sympy_utils.convert_pow_to_mul`（`quark_diagram.py:3357-3358`）；
- 从 `.propagators` re-export：`Particle, Meson, Current, Propagator, PropagatorLocal, PropagatorWithCurrent`（`quark_diagram.py:19-26`）——这六个类属于 propagator 句柄域，详见《propagators.md》。

被谁使用（grep 确认）：

| 使用方 | 用途 |
|---|---|
| `lattice/__init__.py:42-52` | 顶层 API 导出 |
| `lattice/hadron.py:19,214-222` | `diagram_simplify, diagram_vertice_replace, quark_contract` |
| `lattice/group_projection.py:7,78` | `Diagram`（群投影分派） |
| `lattice/insertion/current.py:772` | `CurrentVertexAdapter` |
| `lattice/quark_draw.py:6` | `vertex_type_from_matrix` |

`quark_draw.py` 依赖第三方 `feynman.diagrams.Diagram`、`feynman.Operator/Vertex`、`matplotlib.pyplot`（`quark_draw.py:1-3`），仓库内仅依赖 `vertex_type_from_matrix`（`quark_draw.py:6`）；使用方为 `example/gen_multi_draw_diagrams.py:72` 与 `test/test_quark_draw.py`。

### 1.3 固定 einsum 字母表（`quark_diagram.py:52-56`）

| 常量 | 字母 | 语义 |
|---|---|---|
| `_SUB_VECTOR` | `NOPQRSTUVWXYZ` | eigenvector（低模）索引 |
| `_SUB_SPIN` | `ABCDEFGHIJKLM` | spin 索引 |
| `_SUB_POINT` | `nopqrstuvwxyz` | point（点源）索引 |
| `_SUB_COLOR` | `abcdefghijklm` | color 索引 |

13 个 spin 字母意味着：**每个 connected contraction group 最多 13 条 propagator**（每条占 2 个 node 编号）。

## ② 公开 API 参考

`quark_diagram.py` 的 `__all__`（`quark_diagram.py:28-46`）列出全部公开符号；其中 re-export 的 6 个 propagator 句柄类不在本篇展开。

### 2.1 采样组合数学工具

| 函数 | 签名 | 参数 | 返回值 |
|---|---|---|---|
| `integer_partitions` | `integer_partitions(n: int) -> List[List[int]]`（`quark_diagram.py:71`） | `n`：被分拆的非负整数 | n 的全部整数分拆，降序正整数列表；`integer_partitions(0) == [[]]`、`integer_partitions(1) == [[1]]`（`quark_diagram.py:84-86`） |
| `calculate_sampling_weight` | `calculate_sampling_weight(L: int, usedNp: int, k: int) -> float`（`quark_diagram.py:99`） | `L`：空间格长（总点数 L³）；`usedNp`：采样点数；`k`：所需互异点数 | 补偿权重 `w(k) = L³(L³−1)…(L³−k+1) / [usedNp(usedNp−1)…(usedNp−k+1)]`，即 C(L³,k)/C(usedNp,k)（`quark_diagram.py:103-132`）；边界：`k==0 → 1.0`，`k > min(M,N) → 0.0`（`quark_diagram.py:114-118`） |
| `enumerate_point_scenes` | `enumerate_point_scenes(r: int, L: int, usedNp: int) -> List[Tuple[List[int], float]]`（`quark_diagram.py:134`） | `r`：point 位置数；`L`、`usedNp` 同上 | `[(partition, weight), …]`：每个分拆描述哪些位置共点（如 `[2,1]` = 前 2 个共点、第 3 个独立），`k=len(partition)` 为互异点数，权重为 `calculate_sampling_weight(L, usedNp, k)`（`quark_diagram.py:160-172`） |
| `partition_to_constraints` | `partition_to_constraints(partition: List[int], point_positions: List[Tuple[int, str]])`（`quark_diagram.py:174`） | `partition`：分拆；`point_positions`：位置列表 | `constraints[vertex_idx] = (left_point_id, right_point_id)`：None 表示该侧是 eigenvector 侧，同数字表示共享格点；按 `point_positions` 顺序把每组 group_size 个位置分配同一 point_id（`quark_diagram.py:174-219`）。仓库内测试无直接引用；生产路径由 `StateExpandedDiagram._build_scene_constraints`（`quark_diagram.py:1453-1505`）的独立同构实现替代 |

### 2.2 `CurrentVertexAdapter`（`quark_diagram.py:223-348`）

把"等时（equal-time）组装完成的 V2V current 顶点"适配成 `get(t)` 接口供收缩使用；显式拒绝 temporal point-split 项（`quark_diagram.py:226-228, 306-310`，point-split 抛 `ValueError`）。

```python
def __init__(self, vertices_by_time) -> None   # quark_diagram.py:230
```

- **参数** `vertices_by_time`：`dict[int_time, adapter_dict]`，逐条目校验（`quark_diagram.py:231-326`）：
  - key 为非 bool 整数；
  - entry 为 dict 且 `entry["schema"] == "lattice.current.assembler/v1"`；
  - `entry["axes"] == ("sink_spin","source_spin","sink_ne","source_ne")`；
  - `term_count` 为正整数、`terms` 长度一致；
  - 每个 term 的 `endpoints` 恰含 `{"bar_time","field_time","link_origin_time","temporal_point_split","boundary"}`；三个时间为整数；`boundary ∈ {"periodic","open","unbounded"}`；`temporal_point_split` 为 bool 且为 False、`bar_time == field_time`；
  - `entry["vertex"]` 为有限复数 4 维数组，前两轴 `(4,4)`（即 `(4,4,sink_ne,source_ne)`），后两轴与 `entry["ne"] = {"source":{available,used},"sink":{available,used}}` 的 used 一致；
  - 所有条目共享同一 `(source_ne, sink_ne)`（`quark_diagram.py:324-326`）。
- **数据变换**：缓存前转置 `value.transpose(1, 0, 3, 2)`，把 assembler 的 `(sink_spin, source_spin, sink_ne, source_ne)` 重排为 `(source_spin, sink_spin, source_ne, sink_ne)`（`quark_diagram.py:327`）。
- **属性**：`used_source_ne`、`used_sink_ne`、`usedNe`（source==sink 时为该值否则 None）、`smeared = False`（`quark_diagram.py:329-332`）。
- **`get(time)`**（`quark_diagram.py:334-348`）：整数时间返回单个数组，缺时间抛 `IndexError`；1 维整数数组时间沿 axis 0 `np.stack` 返回；其他类型抛 `TypeError`。

### 2.3 `validate_adjacency_matrix`

```python
def validate_adjacency_matrix(adjacency_matrix) -> None   # quark_diagram.py:366
```

校验 quark-line 度数不变量：要求方阵（`quark_diagram.py:385-388`）；对每个非零条目中的每个 propagator 标签，`degree[i][0] += 1`（离开 i）、`degree[j][1] += 1`（到达 j）（`quark_diagram.py:382-397`）。允许的（出度， 入度）恰三种（`quark_diagram.py:398-416`）：

| (出度， 入度) | 含义 |
|---|---|
| `(1,1)` | meson（允许自环，即非零对角元） |
| `(3,0)` | source 端 baryon（三条 quark 全离开） |
| `(0,3)` | sink 端 baryon（三条 quark 全到达） |

`(0,0)` 被刻意禁止并提示"不连任何东西的 hadron 必须自环"（`quark_diagram.py:404-416`）；违反抛 `ValueError` 指出第一个违规顶点。

### 2.4 `QuarkDiagramOriginal`（`quark_diagram.py:419-505`）

旧版实现：`__init__(self, adjacency_matrix)`（`quark_diagram.py:419-425`）仅设 `adjacency_matrix/operands/subscripts/operands_data` 并调 `analyse()`（`quark_diagram.py:427-505`）。`analyse()` 与新版 `analyse_v2v` 相同的 BFS 分组 + 下标构造，但不校验、不产出 `propagator_types/vertex_types/vertex_point_info`。仓库内无调用方（仅 `__all__` 导出）。

### 2.5 `QuarkDiagram`（`quark_diagram.py:507-848`）——图表示核心

```python
def __init__(self, adjacency_matrix, vertex_list: List[int] = None,
             L: int = None, usedNp: int = None,
             debug: bool = False, validate: bool = True) -> None   # quark_diagram.py:508-514
```

| 参数 | 含义 |
|---|---|
| `adjacency_matrix` | n×n 嵌套列表，见 §3.1 的边语义 |
| `vertex_list` | 长度 n 的顶点对象列表；元素为 0/None 表示普通顶点，非零表示 current 顶点（`quark_diagram.py:575-597`）；数值求值时元素是 `Meson/Current` 句柄或 `HadronIrrepRow` |
| `L` | 空间格长（总点数 L³），用于采样权重（`quark_diagram.py:522-523`） |
| `usedNp` | 采样点数 |
| `debug` | 调试打印开关 |
| `validate` | `True` 时调 `validate_adjacency_matrix`；`False` 用于"机械片段而非完整图"（`quark_diagram.py:526-546`） |

实例字段（`quark_diagram.py:520-543`）：

| 字段 | 含义 |
|---|---|
| `operands` | 每个 connected contraction group 一项 `[propagator_operands, vertex_operands]`；propagator 项为 `[prop_id, src_vertex_idx, snk_vertex_idx]` |
| `subscripts` | 每组一个 `",".join(propagator_subscripts) + "," + ",".join(vertex_subscripts)` |
| `operands_data` | 每组对应的 numpy/backend 数组 |
| `propagator_types` | 每组每线的 `"VSV"/"VSP"/"PSV"/"PSP"` |
| `sampling_groups` | `Dict[int, List[Tuple[int,str]]]`：group_id → [(vertex_idx, "left"\|"right")] |
| `scene_weights` / `scene_constraints` | 场景权重与约束（`quark_diagram.py:535-542`） |

方法：

| 方法 | 位置 | 说明 |
|---|---|---|
| `analyse()` | `quark_diagram.py:555-580` | 调度器：`vertex_list` 存在且含非零项 → 初始化 `expanded_diagrams/expanded_diagrams_weights` 并调 `expand_with_current()`；否则 `analyse_v2v()` |
| `expand_with_current()` | `quark_diagram.py:582-721` | current 顶点状态展开，见 §4.3 |
| `analyse_v2v()` | `quark_diagram.py:724-848` | V2V einsum 下标构造，见 §4.2；幂等保护 `if not self.subscripts == []: return`（`quark_diagram.py:725-726`） |

### 2.6 `StateExpandedDiagram(QuarkDiagram)`（`quark_diagram.py:851-1505`）

```python
def __init__(self, adjacency_matrix, vertex_list: List[int] = None,
             vertex_state: tuple = None, L: int = None,
             usedNp: int = None, debug: bool = False) -> None   # quark_diagram.py:864-872
```

- `vertex_state`：每顶点的 `(left_state, right_state)` 元组，每个状态取值 `"v"`（eigenvector）或 `"p"`（point）。
- 绕过父类 `__init__`（不 validate），额外初始化 `vertex_types/vertex_infos/sampling_groups/scene_weights/scene_constraints/scene_diagrams` 与 `_vertex_state`，直接执行 `_analyse_with_states(vertex_state)`（`quark_diagram.py:906-940`）。

| 方法 | 位置 | 说明 |
|---|---|---|
| `_analyse_with_states` | `quark_diagram.py:909-973` | 与 `analyse_v2v` 相同的 BFS 连通分量划分，但每分量调 `_build_subscripts_with_types` 生成混合 V/P 类型下标 |
| `_build_subscripts_with_types` | `quark_diagram.py:975-1303` | 见 §4.4 |
| `expand_scenes()` | `quark_diagram.py:1305-1414` | 见 §4.5 |
| `_collect_sampling_groups` | `quark_diagram.py:1416-1451` | 按 `vertex_list[vertex_idx]` 的值（group_id）把所有 "p" 状态的 (顶点， 侧) 归组 |
| `_build_scene_constraints` | `quark_diagram.py:1453-1505` | 分拆→约束，与 `partition_to_constraints` 同构；各组用独立 point_id 区间（`point_id_offset` 递增）避免跨组冲突（`quark_diagram.py:1477-1505`） |

### 2.7 `SceneExpandedDiagram(QuarkDiagram)`（`quark_diagram.py:1507-1643`）

```python
def __init__(self, adjacency_matrix, vertex_list=None, operands=None,
             subscripts=None, operands_data=None, propagator_types=None,
             vertex_types=None, vertex_infos=None, scene_constraints=None,
             L=None, usedNp=None, debug=False) -> None   # quark_diagram.py:1520-1536
```

完全绕过父类初始化与 BFS：直接继承 `StateExpandedDiagram` 已算好的元数据，只附加自己的 `scene_constraints`（`quark_diagram.py:1553-1570`）。

`unify_vertex_point_color_indices() -> None`（`quark_diagram.py:1570-1643`）：按约束格式 `[(left_id, right_id), …]`（`constraints[i]` 对应顶点 i；同数字表示须用相同 point/color 下标；None 表示该侧是 eigenvector 位置）构建"约束数字→位置"映射并做 debug 打印。**当前函数体没有任何修改 `self.subscripts`/`self.vertex_infos` 的写操作，也无返回值**（经阅读源码 `quark_diagram.py:1610-1643` 确认）。

### 2.8 数值求值入口

#### `compute_diagrams_multitime`（`quark_diagram.py:1646-1936`）——主求值入口

```python
def compute_diagrams_multitime(
    diagrams: List[QuarkDiagram], time_list,
    vertex_list: List[Meson], propagator_list: List[Propagator],
    multitime_shape: int = False, debug: bool = False,
)
```

- `multitime_shape` 的注解 `int` 与默认值 `False` 类型不符，代码中仅作 truthy 使用（`quark_diagram.py:1652, 1890-1896` 附近实现）。
- 流程见 §4.6。返回 `backend.asarray(diagram_value)`（`quark_diagram.py:1936`），第 i 项为第 i 个 diagram 的值。

#### `compute_diagrams`（`quark_diagram.py:1938-2065`）——旧版求值

与 multitime 版的差别：

| 差异点 | `compute_diagrams` | `compute_diagrams_multitime` |
|---|---|---|
| 自动展开 current 状态 | 无 | 有（`quark_diagram.py:1656-1672`） |
| 数组时间（multitime） | 不支持 | 支持（至多一个数组时间对象，`quark_diagram.py:1675-1685`） |
| VSP/PSV/PSP 取数 | 非高模接口 `get_v2p/get_p2v/get_p2p`，返回 `(data, effective_type)` 二元组且 effective_type 覆写 prop_type（`quark_diagram.py:1998-2022`） | `get_VSP_highmode/get_PSV_highmode/get_PSP_highmode`（`quark_diagram.py:1730-1820`） |
| 顶点取数 | 一律 `vertex.get(t)`（V2V） | 按 `vertex_types` 分派 `get/get_v2p/get_p2v/get_p2p` |
| 场景权重 | 无 | 乘 `scene_weights[0]`（`quark_diagram.py:1926-1935`） |

注意：`get_v2p/get_p2v/get_p2p` 在 `PropagatorWithCurrent` 上未定义（会 AttributeError→RuntimeError），只在 `Current`（vertex 句柄）上存在——即 `PropagatorWithCurrent` 只能配 `compute_diagrams_multitime`。

### 2.9 `Diagram(Symbol)`（`quark_diagram.py:2068-2452`）——sympy 符号包装

sympy `Symbol` 子类；符号名为 `f"{diagram.adjacency_matrix},{time_list},{vertex_list},{propagator_list}"`（`quark_diagram.py:2069-2080`）。属性：`diagram/time_list/vertex_list/propagator_list/value/value_pointer`（`quark_diagram.py:2081-2102`）。

| 方法 | 位置 | 说明 |
|---|---|---|
| `calc()` | `quark_diagram.py:2105-2112` | 惰性求值 `compute_diagrams_multitime([self.diagram], …)`；首次调用前把 `self.value` 短暂设为 `self.__hash__()` 再覆盖（`quark_diagram.py:2107-2110`） |
| `__str__/__repr__/__eq__/__hash__` | `quark_diagram.py:2113-2131` | 相等 = 四元组逐项相等；hash = sha256(str) mod 2³¹；`_collect_diagrams` 去重依赖 `__eq__` |
| `transform(group_element, time=None)` | `quark_diagram.py:2132-2164` | `assert isinstance(vertex, HadronIrrepRow)`（`quark_diagram.py:2139`）；`time is None` 或时间匹配的顶点做 `vertex.transform(group_element)`；`Add/Mul.make_args` 拆 (系数, HadronIrrepRow) 对；对全部顶点做笛卡尔积，得 `Σ diagram_coeff * Diagram(self.diagram, self.time_list, new_vertex_list, self.propagator_list)`——图与 propagator 不变，只换顶点 |
| `conjugate()` | `quark_diagram.py:2166-2175` | 每顶点换 `vertex.conjugate()`；同样断言顶点是 `HadronIrrepRow` |
| `simplify()` | `quark_diagram.py:2176-2427` | 三步合一，见 §4.7 |
| `replace_propagator(propagator_map: Dict)` / `replace_vertex(vertex_map: Callable)` / `replace_time(time_map: Dict)` | `quark_diagram.py:2429-2452` | 原地替换（`vertex_map` 返回 None = 不替换） |

### 2.10 表达式级工具

| 函数 | 位置 | 说明 |
|---|---|---|
| `diagram_vertice_replace(expr, indice_map: Dict)` | `quark_diagram.py:2455-2494` | 对每个 Diagram 用 `indice_map[v]` 重建 vertex_list；递归处理 list/tuple/dict/Add/Mul/带 args 的 sympy 节点。调用方 `lattice/hadron.py:222` |
| `diagram_simplify(expr)` | `quark_diagram.py:2496-2596` | 递归对每个 Diagram 调 `.simplify()` 再 `sp.simplify`；异常时打日志并原样返回（`quark_diagram.py:2523-2534`）；支持 list/ndarray（flatten→逐元素→reshape）/Add/Mul/Pow/dict/tuple。调用方 `lattice/hadron.py:214` |
| `vertex_type_from_matrix(adjacency_matrix, vertex_idx)` | `quark_diagram.py:2598-2641` | 从邻接矩阵推断 meson/baryon，见 §4.8 |
| `remove_unexpected_diagram(expr, condition: Callable)` | `quark_diagram.py:2643-2717` | "存在任一 propagator 不满足 condition" 的 Diagram 替换为 `S(0)`（`quark_diagram.py:2672-2677`）。docstring 残留 `propagator_list` 参数描述与实际签名不符（`quark_diagram.py:2644-2665`） |
| `remove_disconneted_diagram(expr, propagator_list)` | `quark_diagram.py:2719-2799`（函数名拼写如此） | "包含 propagator_list 中任一 propagator" 的 Diagram 替换为 `S(0)`（`quark_diagram.py:2743-2747`）；调用方 `test/test_group_projection.py:8`，配合 simplify 的连通分量拆分剔除含指定 propagator 的因子 |

### 2.11 求值流水线（prepare / bind / eval）

| 函数 | 位置 | 签名与说明 |
|---|---|---|
| `calc_diagram_prepare` | `quark_diagram.py:2971-3043` | `calc_diagram_prepare(expr, propagator_map=None, vertex_map=None, save_dir=None, debug=False, timing: Dict=None) -> _CalcDiagramPrepared`。显式调用 `_collect_diagrams` 两次（`quark_diagram.py:2998-3005`），然后 `_build_combined`，打包为 `_CalcDiagramPrepared`（`quark_diagram.py:3045-3061`，`__slots__` 持有 expr/diagram_list/combined_diagrams/all_vertices/all_propagators/all_times/irrep_vertices/save_dir/debug/backend/timing）。空 `diagram_list` 时返回空 prepared（`quark_diagram.py:3007-3011`）。`timing` 为可选 dict，填充各阶段耗时 |
| `calc_diagram_bind` | `quark_diagram.py:3064-3103` | `calc_diagram_bind(prepared, vertex_map, timing=None)`。对 `prepared.irrep_vertices` 逐个调 `vertex_map`（典型为 `Meson.load` 包装），返回 None 则保留原顶点；结果写回 `prepared.all_vertices`；`irrep_vertices` 保持不变（结构上可重复 bind；`test/test_calc_diagram_bind.py:43` 验证） |
| `calc_diagram_eval` | `quark_diagram.py:3105-3143` | `calc_diagram_eval(prepared, time_map: Dict=None)`。把 `time_map` 应用到 `all_times` 副本；`debug=False` 时一次调 `compute_diagrams_multitime(combined_diagrams, …, multitime_shape=True)` 算出所有合并图的值并按序写入 `diagram_list[i].value`；`debug=True` 时以 `Symbol("result_i")` 代替数值。`_replace_diagrams`（`quark_diagram.py:3145-3206`）把表达式中的 Diagram 按 `value_pointer`/`value` 替换为数值；无值时返回 1 不抛错（`quark_diagram.py:3166-3174`） |
| `calc_diagram` | `quark_diagram.py:3208-3223` | `calc_diagram(expr, time_map=None, propagator_map=None, vertex_map=None, save_dir=None, debug=False)`。便捷入口 = prepare + eval；docstring 明示"对 time_map 循环时用 prepare + eval 分离" |

内部分工：`_collect_diagrams(expr, diagram_list, save_dir, backend)`（`quark_diagram.py:2801-2878`）递归收集表达式中所有互不相等的 Diagram，用 `__eq__` 查重、`value_pointer` 指向既有索引；`save_dir` 磁盘缓存分支被 `and False` 短路禁用（`quark_diagram.py:2827-2834`）。`_build_combined(diagram_list, vertex_map, propagator_map, debug, timing=None)`（`quark_diagram.py:2880-2969`）把多个 Diagram 合并到统一骨架（唯一 (time, vertex) 对 + 唯一 propagator，重映射后 `QuarkDiagram(new_adjacency, validate=False)`）；**ndarray 邻接条目不被重映射**（只有 int 与 list 两个分支，`quark_diagram.py:2948-2954`）。

典型生产循环：prepare（每表达式一次）→ propagator.load → bind（每 config 一次）→ eval（每 time_map 一次）。

### 2.12 `quark_contract`（`quark_diagram.py:3226-3353`）与 `_quark_contract`（`quark_diagram.py:3355-3387`）

```python
def quark_contract(expr, particles, degenerate=True)   # quark_diagram.py:3226
```

| 参数 | 含义 |
|---|---|
| `expr` | 含 `HadronFlavorStructure` 的 sympy 表达式（`HadronFlavorStructure(flavor_str, time=0)`，携带 `baryon_num/quark_list/anti_quark_list/time`） |
| `particles` | 粒子（irrep row）列表，作为输出 Diagram 的 vertex_list |
| `degenerate` | `True` 且 flavor ∈ {"u","d"} 时 propagator 统一记为 `Propagator("q", …)`（u/d 简并共用 "q"），否则用原 flavor（`quark_diagram.py:3377-3380`） |

返回 `diagram_expr = Σ_i coeffs[i] * Diagram(QuarkDiagram(diagrams[i]), time_list, particles, propagators)`（`quark_diagram.py:3344-3353`）。算法见 §4.9。

`_quark_contract(symbol_list, result_list, result, degenerate)`（`quark_diagram.py:3355-3387`）为递归回溯收缩：终止条件 symbol_list 空 ⇒ `result_list.append(Mul(*result))`；取第一个 antiquark 位置 i，对每个与 src.flavor 相同的非 antiquark（snk，位置 j）配对：费米符号因子为 `i > j` 时 `(-1)^(i-j-1)`、否则 `(-1)^(j-i)`（`quark_diagram.py:3371-3376`）；回溯时 `result.pop()` 并把两符号插回原位（`quark_diagram.py:3381-3387`）。

### 2.13 `quark_draw.py` 公开符号

| 符号 | 位置 | 说明 |
|---|---|---|
| `Meson`（NamedTuple） | `quark_draw.py:126-131` | 字段 `vertex_out: List[Vertex]; vertex_in: List[Vertex]; operator: Operator; xy: tuple` |
| `Baryon`（NamedTuple） | `quark_draw.py:133-138` | 字段同 `Meson` |
| `meson_source(diagram, xy, size, tag)` | `quark_draw.py:140-145` | source 侧 vertex_out 在 xy 上方（dy=+size）、vertex_in 在下方；`diagram.operator([...], c=2)` 居中并 `operator.text(tag)` 标注 |
| `meson_sink(diagram, xy, size, tag)` | `quark_draw.py:147-152` | 与 source 上下颠倒 |
| `baryon_source(diagram, xy, size, tag)` | `quark_draw.py:154-161` | 三个 vertex_out（上、右 dx=size/2、下），vertex_in 空 |
| `baryon_sink(diagram, xy, size, tag)` | `quark_draw.py:163-170` | 三个 vertex_in（上、左 dx=-size/2、下） |
| `make_operator(hadron, pos, **kwargs)` | `quark_draw.py:172-185` | 分派上述四种；`pos` 非 src/snk 抛 `ValueError` |
| `draw_diagram(diagram, adjacency_matrix, operator_list, line_color_list)` | `quark_draw.py:76-124` | 旧版逐图绘制：BFS 收集线，按两端相对位置选线型，`style["color"] = line_color_list[propagator[0]]`；`outward_idx/inward_idx` 递增保证同一 operator 多线接到不同 vertex 槽位（`quark_draw.py:114-123`）。**副作用：原地清零传入的 adjacency_matrix**（`quark_draw.py:97`） |
| `draw_quark_diagram(diagram, line_color_list=None, save_path=None, quark_types=None)` | `quark_draw.py:252-283` | 高层入口：图、顶点侧、标签全部从 `lattice.quark_diagram.Diagram` 对象读取，只有颜色由调用者给。`line_color_list=None` 时按邻接矩阵最高标签生成 `[None]*(highest+1)`（标签从 1 起）；`quark_types: {label: flavour}` 给定时按 `QUARK_COLORS` 上色并在中点标注味名（垂直偏移 0.04，`quark_draw.py:433-447`）；委托 `draw_single_diagram` |
| `draw_multi_diagrams(adjacency_matrix_list, vertex_attribute_list, line_color_list, save_path=None)` | `quark_draw.py:286-294` | 多个邻接矩阵复用同一 vertex_attribute_list 逐个调 `draw_single_diagram`；`save_path` 缺省 `[None]*n` |
| `draw_single_diagram(adjacency_matrix, vertex_attribute_list, line_color_list, save_path=None, quark_types=None)` | `quark_draw.py:307-473` | 核心绘制流程，见 §4.10 |
| `QUARK_COLORS` | `quark_draw.py:297-305` | 味→颜色映射：u=r, d=b, s=g, c=m, b=k, t=c |
| `_flatten_paths(path)` | `quark_draw.py:53-74` | 与 `quark_diagram._flatten_paths_matrix` 相同的递归展平（int 丢弃 0；list 递归；其他 ValueError） |
| `_vertex_attributes_from_diagram(diagram)` | `quark_draw.py:231-250` | 从 `Diagram` 提取绘图属性：邻接矩阵取 `diagram.diagram.adjacency_matrix`；每个 vertex_list 元素生成 `dict(pos="src" if vertex.dagger else "snk", type=vertex_type_from_matrix(...), name=rf"${vertex.hadron_name}$")`。即 **dagger=True ⇒ source 侧** |

模块级副作用：import 即创建 `fig = plt.figure(figsize=(6,6)); ax = fig.add_subplot(111)`（`quark_draw.py:8-9`）与一组椭圆风格线型 dict（`quark_draw.py:11-62`），并执行一段演示绘图脚本（`quark_draw.py:192-228`）。即 **import `quark_draw` 有副作用（建图并 print 一次 propagators）**。

## ③ 数据结构与数组形状约定

### 3.1 邻接矩阵

拓扑由三要素完全描述：**邻接矩阵**、**vertex_list**（长度 n 的顶点对象列表，`vertex_list[i] == 0` 或 None = 普通（非 current）顶点）、**time_list**（不在 `QuarkDiagram` 内，在 `Diagram` 包装里，与顶点一一对应）。

`adjacency_matrix[i][j]` 编码"从顶点 i 流向顶点 j 的 quark 线"（`quark_diagram.py:3288-3323`；两种书写约定的说明见 `vertex_type_from_matrix` docstring，`quark_diagram.py:2598-2641`）：

| 条目形式 | 含义 |
|---|---|
| `0` | 无边 |
| 正整数 `k` | 一条 propagator，标签 k |
| `list`（单层，如手写 baryon 的 `[1,1,1]`） | 多条并列 quark 线 |
| 两层嵌套 `matrix[i][j][source_quark][sink_quark]` | `quark_contract` 生成的 baryon 收缩约定 |
| `numpy.ndarray` | 仅在 `Diagram.simplify` / `_build_combined` / `vertex_type_from_matrix` 层被部分容忍（`quark_diagram.py:2272-2279, 2898-2922`）；`_build_combined` 的索引重写不处理 ndarray 条目（`quark_diagram.py:2948-2954`） |

边语义：propagator 记 `[id, src, snk]`；`_build_subscripts_with_types` 明确 "src (source) 对应 propagator 的 right end (ket)、snk (sink) 对应 left end (bra)"（`quark_diagram.py:1023-1026`）；propagator 类型命名 `S_{sink_type, source_type}`（PSV = sink 是 point、source 是 eigenvector，`quark_diagram.py:1027-1039`）。

### 3.2 einsum 下标布局

#### V2V 路径（`analyse_v2v`）

- propagator 下标初值 `sink_spin + sink_eigen + source_spin + source_eigen`（实现顺序 `_SUB_SPIN[node+1]+_SUB_VECTOR[node+1]+_SUB_SPIN[node]+_SUB_VECTOR[node]`，`quark_diagram.py:831-838`）；
- 下标重排 `s[0::2] + s[1::2]`（`quark_diagram.py:810-818`）：把 `S_snk S_src V_snk V_src` 重排为 `(S_snk S_src)(V_snk V_src)`，使 propagator 数组轴序为 (spin_snk, spin_src, vec_snk, vec_src)；
- `vertex_point_info` 记录每顶点 left/right 侧的 spin/eigen 字母（`quark_diagram.py:820-840`）。

#### 状态展开路径（`_build_subscripts_with_types`，`quark_diagram.py:1064-1267`）

propagator 下标（每条线 `node += 2`）：

| propagator 类型 | 语义 | 下标字母序列 |
|---|---|---|
| VSV | eigen→eigen | `spin_snk + spin_src + vector_snk + vector_src` |
| VSP | sink=eigen, source=point | `spin_snk + spin_src + vector_snk + point_src + color_src` |
| PSV | sink=point, source=eigen | `spin_snk + spin_src + point_snk + color_snk + vector_src` |
| PSP | point→point | `spin_snk + spin_src + point_snk + color_snk + point_src + color_src` |

顶点下标按 vertex_type 组装（`quark_diagram.py:1239-1267`）：

| vertex 类型 | 字母数 | 序列 |
|---|---|---|
| V2V | 4 | `L.spin R.spin L.eigen R.eigen` |
| V2P | 5 | `L.spin R.spin L.eigen R.point R.color` |
| P2V | 5 | `L.spin R.spin L.point L.color R.eigen` |
| P2P | 6 | `L.spin R.spin L.point L.color R.point R.color` |

顶点状态命名交叉：`expand_with_current` 里顶点字典 key `"left"` 在 `_build_subscripts_with_types` 中实际扮演 propagator 的 right/ket 端——代码写 `src_state = vertex_state[src]["left"]`（注释 "right end state"，`quark_diagram.py:1023-1027`）。即顶点侧命名与 propagator 端命名相反。

### 3.3 current 顶点取数形状

| 接口 | 输出形状（int t） | 数组 t |
|---|---|---|
| `vertex.get(t)`（V2V） | `[Ns,Ns,Ne,Ne]` | `[t_len,Ns,Ns,Ne,Ne]` |
| `vertex.get_v2p(t)` | `[Ns,Ns,Ne,Np,Nc]` | `[t_len,…]` |
| `vertex.get_p2v(t)` | `[Ns,Ns,Np,Nc,Ne]` | `[t_len,…]` |
| `vertex.get_p2p(t)` | `[Ns,Ns,Np,Nc,Np,Nc]` | 抛 `NotImplementedError`（`quark_diagram.py` 消费端见 `Current.get_p2p`） |

### 3.4 `quark_contract` 的 Tag 编码

`tag = hadron_id*3 + quark_id`，后续由 `source_tag.tag // 3` 反解 hadron_id、`% 3` 反解 quark_id（`quark_diagram.py:3313-3319`）。meson（`baryon_num == 0`）的 quark 与 antiquark 共享同一 Tag（`quark_diagram.py:3269-3280`）；baryon（±1）的三 quark 分别 Tag `hadron_id*3 + quark_id`（`quark_diagram.py:3281-3297`）。propagator 标签从 1 开始（`propagators` 从 `[None]` 起头，`quark_diagram.py:3284`），0 为"无边"。

## ④ 算法与流程

### 4.1 连通分量 = contraction group

每个连通分量是独立 contraction group：`compute_diagrams_*` 逐 group 做 einsum 后把各组结果相乘（`quark_diagram.py:1718-1719, 2050-2055`）——"图 = 连通分量之积"在求值层面成立。连通分量由 BFS 划分，遍历过的边置 0。

### 4.2 `analyse_v2v`：V2V 下标构造（`quark_diagram.py:724-848`）

1. 幂等保护（`quark_diagram.py:725-726`）；
2. 对每个连通分量收集 propagator `[id, i, j]`；list 条目逐个展开，嵌套 list 递归提取（内嵌 `extract_propagators`，`quark_diagram.py:758-777`）——baryon 两层嵌套在 v2v 路径被拍平；
3. 逐条 propagator（`quark_diagram.py:783-809`）：下标构造见 §3.2；顶点首次出现登记 2 字母下标，重复出现则拼接；
4. 下标重排 `s[0::2] + s[1::2]`（`quark_diagram.py:810-818`）；
5. 记录 `vertex_point_info`、追加 operands/subscripts；`propagator_types.append(["VSV"]*n)`、`vertex_types.append(["V2V"]*n)`（`quark_diagram.py:842-848`）。

### 4.3 `expand_with_current`：current 顶点状态展开（`quark_diagram.py:582-721`）

1. 收集与 current 顶点相连的全部边（`quark_diagram.py:605-622`）；
2. 每顶点状态集合：非 current 只有 `{"left":"v","right":"v"}`；current 有全部 4 种 `{"left","right"} ∈ {"v","p"}²`（`quark_diagram.py:633-650`）；
3. 顶点状态笛卡尔积 → `4^k`（k=current 顶点数）个组合（`quark_diagram.py:653-654`；测试断言见 `test/test_current_contraction.py:74-77`）；
4. 每组合构造 `StateExpandedDiagram` 并 `expand_scenes()`，追加 `expanded_diagrams`；`expanded_diagrams_weights.append(1.0)` 恒为 1.0（`quark_diagram.py:684-716`）——真正的采样权重挂在 scene 层（`quark_diagram.py:1388-1410`）。

### 4.4 `_build_subscripts_with_types`（`quark_diagram.py:975-1303`）

1. 所有顶点都是 operand：`vertex_operands = list(range(len(vertex_state)))`（与 v2v 的按需登记不同，`quark_diagram.py:992-997`）；按 (left,right) 状态赋 `vertex_types ∈ {"V2V","V2P","P2V","P2P"}` 与空 `vertex_infos` 槽（`quark_diagram.py:995-1042`）；
2. 逐 propagator 定类型：`src_state = vertex_state[src]["left"]`（右端/ket）、`snk_state = vertex_state[snk]["right"]`（左端/bra）→ VSV/PSV/VSP/PSP（`quark_diagram.py:1023-1039`）；
3. 下标布局见 §3.2；
4. 追加 operands/subscripts/propagator_types/vertex_types/vertex_infos；`operands_data.append(None)` 占位（`quark_diagram.py:1289-1296`）。

### 4.5 `expand_scenes`：点重合场景枚举（`quark_diagram.py:1305-1414`）

1. `_collect_sampling_groups`：group_id 相同的 "p" 位置归入同一采样组（`quark_diagram.py:1416-1451`）；
2. 无采样组：构造一个 `SceneExpandedDiagram`（空约束）+ `unify_vertex_point_color_indices()`，`scene_weights=[1.0]`、`scene_constraints=[[]]`（`quark_diagram.py:1333-1357`）；
3. 有采样组：每组 `enumerate_point_scenes(r, M, N)`，组间场景做笛卡尔积；总权重 = 各组权重乘积（公式注释 `w(k) = C(L³,k)/C(usedNp,k)`，`quark_diagram.py:1319-1321`）；约束由 `_build_scene_constraints` 生成；每组合一个 `SceneExpandedDiagram` + `unify_vertex_point_color_indices()`（`quark_diagram.py:1360-1413`）。

**当前实现现状（阅读源码确认）**：`unify_vertex_point_color_indices` 目前不修改 einsum 下标（`quark_diagram.py:1570-1643`）；`compute_diagrams_multitime` 只取 `scene_weights[0]` 缩放（`quark_diagram.py:1926-1935`），且 `scene_diagrams` 列表在 `lattice/` 与 `example/` 内没有逐场景求值的消费者（grep 确认）。因此"多场景加权求和"的求值闭环在当前代码中不可见——重合场景间数值未区分，仅权重不同。

### 4.6 `compute_diagrams_multitime` 求值流程（`quark_diagram.py:1646-1936`）

1. **自动展开**（`quark_diagram.py:1656-1672`）：输入 diagram 若带非空 `expanded_diagrams`，用展开后的 `StateExpandedDiagram` 列表替换原图；
2. **multitime 约定**（`quark_diagram.py:1675-1685`）：time_list 元素可为 int 或"数组时间"；至多允许一个唯一数组时间对象（按 `id()` 判等），否则 `NotImplementedError("only support one multitime yet")`；
3. **逐 diagram、逐 contraction group**：
   - propagator 取数（`quark_diagram.py:1725-1870`）按 `propagator_types[contraction_idx][prop_idx]` 分派：

     | 类型 | 调用 | 取数后切片 |
     |---|---|---|
     | VSV | `propagator.get(t_source, t_sink)` | `[..., :usedNe_sink, :]` / `[..., :usedNe_source]` |
     | VSP | `get_VSP_highmode(t_src, t_snk, usedNe_source=…, usedNe_sink=…)` | sink 侧 `[..., :usedNe_sink, :, :]`，source 侧 `[..., :usedNp_source, :]` |
     | PSV | `get_PSV_highmode(...)` | `[..., :usedNp_sink, :, :]` 与 `[..., :usedNe_source]` |
     | PSP | `get_PSP_highmode(usedNe_sink, usedNe_source)` | `[..., :usedNp_sink, :, :, :]` 与 `[..., :usedNp_source, :]` |

     失败包装 RuntimeError 附 propagator 可用方法清单（`quark_diagram.py:1828-1845`）；`usedNe/usedNp` 从 vertex 句柄取：`getattr(src_vertex, "usedNe", None)` 等（`quark_diagram.py:1743-1751`）。
   - multitime 下标：时间非 int 的 operand 下标加前缀 `"t"`，标记 `have_multitime`（`quark_diagram.py:1821-1827, 1879-1896`）；
   - vertex 取数（`quark_diagram.py:1872-1906`）按 `vertex_types` 分派 `vertex.get(t)` / `get_v2p(t)` / `get_p2v(t)` / `get_p2p(t)`；
   - **输出形状对齐**（`quark_diagram.py:1890-1896`）：整组无 multitime 但 `multitime_shape` truthy 时追加哑维 `subscripts.append("t")`、`operands_data.append([1]*len(multi_time))`，输出端改 `"->t"`；有 multitime 时末段加 `"->t"`。即每组收缩输出至少带一个 `t` 轴；
   - `result = contract(final_subscripts, *operands_data)`；各组结果相乘（`quark_diagram.py:1898-1905`）；
4. **场景权重**（`quark_diagram.py:1926-1935`）：若 diagram 有非空 `scene_weights`，乘 `scene_weights[0]`；
5. 返回 `backend.asarray(diagram_value)`（`quark_diagram.py:1936`）。

### 4.7 `Diagram.simplify` 三步（`quark_diagram.py:2176-2427`）

1. BFS 求连通分量（容忍 int/list/ndarray 边条目，`quark_diagram.py:2204-2259`）；
2. 顶点排序 `sort(key=(time, vertex))`；有完全相同 (time, vertex) 顶点时在各相同组内枚举 `permutations`、跨组笛卡尔积，取 sha256 最小者为规范形（`quark_diagram.py:2264-2292`）；
3. 每分量裁出新矩阵 `component_matrix[i][j] = adjacency_matrix[vertices[j]][vertices[i]]`（注意 `[vertices[j]][vertices[i]]` 的取法，`quark_diagram.py:2296-2300`）、`QuarkDiagram(component_matrix, validate=False)`、propagator 重编号（`used_propagators` 排序后前插 0 占位，`quark_diagram.py:2321-2360`），结果 `Mul` 后 `sp.simplify`（`quark_diagram.py:2423-2427`）。

### 4.8 `vertex_type_from_matrix` 判定规则（`quark_diagram.py:2598-2641`）

依据 `quark_contract` 的构造约定：`matrix[i][j]` 形状 `(M,N)` 中**列维 N 属于顶点 i、行维 M 属于顶点 j**（baryon-baryon→(3,3)、baryon-meson→(1,3)、meson-baryon→(3,1)、meson-meson→标量，`quark_diagram.py:2603-2612`）。判定：顶点 i 是 baryon ⇔ 其行中存在"列宽 3"的条目（两层嵌套看 `len(path[0])==3`；单层手写看 `len(path)==3`，`quark_diagram.py:2624-2641`）。docstring 明示读错维度会把混合图里每个 meson 误分类（`quark_diagram.py:2613-2622`）。

### 4.9 `quark_contract` 算法（`quark_diagram.py:3226-3353`）

1. `expr = convert_pow_to_mul(expr.expand())`，`Add.make_args` 拆项（`quark_diagram.py:3252-3255`）；
2. 对每个 term（`quark_diagram.py:3257-3313`）：`Mul.make_args` 拆因子；`HadronFlavorStructure` 因子按 `baryon_num` 生成 `Qurak` 符号（Tag 编码见 §3.4）；非 `HadronFlavorStructure` 因子乘入 `coeff`（`quark_diagram.py:3298-3300`）；`_quark_contract(symbol_list, result_list, result, degenerate)` 做收缩；项级结果 = `coeff * Add(*result_list)`；
3. `simplify(Add(*result_terms)).expand()` 合并同类项（`quark_diagram.py:3315`）；
4. 对每个合项（`quark_diagram.py:3316-3341`）：建 `num_particles × num_particles` 的 `diagram` 矩阵；按两端 baryon_num 预置嵌套形状 (3,3)/(1,3)/(3,1) 或标量（`quark_diagram.py:3288-3293`）；遍历因子，`Propagator`（`Propagator(flavor, source_tag, sink_tag)`）按 `propagators.index(factor.tag)` 写入矩阵——meson-meson 写标量位、含 baryon 写 `[quark_id_source][quark_id_sink]`；`propagators` 从 `[None]` 起头故标签从 1 开始（`quark_diagram.py:3284`）；
5. 系数（非 Propagator 因子）入 `coeffs`；最终逐项 `Diagram(...)` 求和（`quark_diagram.py:3344-3353`）。

输出邻接矩阵采用"两层嵌套 baryon"书写约定；`QuarkDiagram(diagrams[i])` 默认 `validate=True` 会查度数不变量。

### 4.10 `draw_single_diagram` 流程（`quark_draw.py:307-473`）

1. 新建 figure/axes（0..1 坐标），`mpl.rc("text", usetex=True)`（`quark_draw.py:325-336`）；
2. BFS 收集 propagators（同样原地清零邻接矩阵，`quark_draw.py:353`）；
3. **每个顶点都画**，包括没连线的 hadron：docstring 说明"quark 必有连线但 hadron 未必——disconnected pieces 是关联函数的合法组成；顶点永不删除或重编号"（`quark_draw.py:308-318`）。孤立 hadron 须写成非零对角元（自环）——与 `validate_adjacency_matrix` 的 (0,0) 禁令呼应；
4. 布局：source 顶点放 x=0.2 列、sink 放 x=0.8 列；y 按 `n_src/n_snk` 与序号 `0.5 + y_tmp/2*0.6` 计算（`quark_draw.py:387-412`）；size=0.1；
5. 逐线选线型（同 `draw_diagram` 规则）、上色、可选味标注、`diagram.line(...)`；
6. `diagram.plot()`；`save_path` 给定时 `diagram.savefig(save_path)`；然后 `diagram.show()`（`quark_draw.py:468-473`）——无 save_path 也会 show。

## ⑤ 不变量与校验

| # | 不变量 | 出处 |
|---|---|---|
| 1 | 每顶点 (出度， 入度) ∈ {(1,1), (3,0), (0,3)}，(0,0) 禁止；`QuarkDiagram(validate=True)` 强制 | `quark_diagram.py:398-416, 544-546` |
| 2 | 邻接矩阵必须方阵；条目仅 int/list/嵌套 list（ndarray 只在 simplify/_build_combined 被部分容忍，`_build_combined` 不映射 ndarray 条目） | `quark_diagram.py:385-388, 2948-2954` |
| 3 | einsum 字母表 26 字母 / 每 contraction group 至多 13 条 propagator | `quark_diagram.py:52-56` |
| 4 | multitime：time_list 中至多一个数组时间对象，按 `id()` 判等——同一数组对象复用是硬要求 | `quark_diagram.py:1675-1685` |
| 5 | propagator 标签从 1 开始、0 为"无边"，`propagators=[None]` 占位 | `quark_diagram.py:3284` |
| 6 | `analyse_v2v` 幂等（`subscripts` 非空即返回） | `quark_diagram.py:725-726` |
| 7 | `Diagram.transform/conjugate` 断言顶点是 `HadronIrrepRow`——符号代数阶段顶点不能是 Meson 句柄 | `quark_diagram.py:2139, 2168` |
| 8 | `CurrentVertexAdapter` 拒绝 point-split 与不等时条目；schema/axes/endpoints/ne 全面校验 | `quark_diagram.py:240-326` |
| 9 | quark_contract 的 Tag 编码 `hadron_id*3 + quark_id`——隐含"每 hadron 至多 3 quark"，与度数不变量呼应 | `quark_diagram.py:3269-3297, 3313-3319` |
| 10 | `draw_diagram`/`draw_single_diagram` 原地清零调用者传入的邻接矩阵；`draw_single_diagram` 无条件 `show()`；import `quark_draw` 有副作用 | `quark_draw.py:97, 353, 473, 8-228` |
| 11 | `compute_diagrams_multitime.multitime_shape` 注解 `int` 与默认值 `False` 类型不符；`remove_unexpected_diagram` docstring 参数描述与签名不符；`Diagram.calc` 先设 `self.value = hash` 再覆盖 | `quark_diagram.py:1652, 2644-2665, 2107-2110` |
| 12 | 场景展开求值现状：`unify_vertex_point_color_indices` 无下标写操作；求值只乘 `scene_weights[0]`；`scene_diagrams` 在 lattice/、example/ 内无逐场景消费者 | `quark_diagram.py:1570-1643, 1926-1935`（grep 验证） |

## ⑥ 相关测试

| 测试文件 | 被测对象 | 确认的行为（出处均为 test/ 下文件：行号） |
|---|---|---|
| `test/test_quark_diagram_contraction.py` | `QuarkDiagram`、`compute_diagrams_multitime`、采样工具 | 两点图 `[[0,1],[1,0]]` + `vertex_list=[0,0]` 不触发 current 展开 (:31-50)；current 状态展开 (:77-106)；propagator type 分派 (:108-127)；scene 权重公式 (:151-178)；multitime 输出形状与 rolling (:316-347)；同子图合并 (:350-368)；展开数 = `4**k`、类型覆盖 VSV/VSP/PSV/PSP (:51-150) |
| `test/test_quark_contract.py` | `quark_contract`、`_quark_contract` | meson-meson / baryon-antibaryon 收缩可运行 (:151-171)；`degenerate=True` 时 `[[1,0],[0,1]]` 自环矩阵 (:173-183)；空符号表得 `[S(1)]` (:188-194)；单 q-q̄ 对产生一个含 Propagator 的项 (:197-215)；Tag/Qurak/Propagator 符号行为 (:13-133) |
| `test/test_diagram.py` | `Diagram.simplify/transform` | 顶点排序、同顶点置换、冗余顶点移除、连通分量拆分 (:71-280)；表达式/嵌套结构递归 (:281-443)；little group 旋转下 `Σ_j M[i,j]·diagram_j` 不变性 (:444-460, TestDiagram2 :534+) |
| `test/test_simplify.py` | 同 test_diagram 的简化子集（独立副本，:67-277） | 同上 |
| `test/test_vertex_type.py` | `vertex_type_from_matrix` | 四种 (meson/baryon) 组合；"baryon 从列读而非行读"；空/标量行按 meson (:57-84) |
| `test/test_quark_draw.py` | `quark_draw`（需 matplotlib+feynman，CI skip） | `_vertex_attributes_from_diagram`：dagger⇒src、hadron_name⇒标签、矩阵⇒类型 (:51-88)；`draw_quark_diagram` 直接吃 Diagram (:89-100)；孤立 hadron 自环且索引对齐 (:102-141)；meson 自环 (:142-162)；quark_types 味色标注 (:163-188)；两层嵌套 baryon 矩阵 (:189-200) |
| `test/test_sampling_weight.py` | `integer_partitions` / `calculate_sampling_weight` / `enumerate_point_scenes` / highmode 投影 | 分拆性质 (:28-95)；权重边界与公式 (:112-243)；场景枚举 (:243-344)；VSP/PSV/PSP 投影公式数值验证 (:382-633) |
| `test/test_calc_diagram_bind.py` | `_CalcDiagramPrepared`、`calc_diagram_bind` | bind 把 irrep 顶点替换为 Meson 句柄、保留 irrep_vertices、timing 字段 (:8-41)；幂等性 (:43+) |
| `test/test_current_contraction.py` | `QuarkDiagram` + `compute_diagrams(_multitime)`（integration） | current 收缩端到端：展开数、类型覆盖、multitime 形状 (:15-260) |
| `test/test_current_vertex.py` | `QuarkDiagram` 展开行为 | 单/双 current 顶点、无 vertex_list 不展开 (:14-130) |
| `test/test_flavor_structure.py` | `quark_contract`、`_quark_contract` | 味结构→图的上层集成 (:9) |
| `test/test_group_projection.py` | `remove_disconneted_diagram` | 群投影表达式中剔除含指定 propagator 的 Diagram (:8) |
| `test/test_ne_provenance.py` | `Meson/Propagator/PropagatorLocal/PropagatorWithCurrent` | 经 quark_diagram re-export 的句柄的 Ne/usedNe 出身校验 (:12) |
