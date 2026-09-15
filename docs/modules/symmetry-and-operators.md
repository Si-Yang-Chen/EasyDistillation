# 对称性与算符构造层（symmetry & operators）

> 本文档描述 EasyDistillation 的立方群（cubic group）表示、动量小群 irrep 投影、以及强子插值算符（interpolating operator）的符号构造层，覆盖 `lattice/symmetry/` 子包与 `lattice/group_projection.py`、`lattice/spatial_structure.py`、`lattice/flavor_structure.py`、`lattice/hadron.py`。所有代码出处均为仓库源码实际行号。

---

## ① 模块概览

### 1.1 各模块职责

| 模块 | 行数 | 一句话职责 |
|---|---|---|
| `lattice/symmetry/group_generator.py` | 315 | 纯数据模块：集中定义 cubic 群 OhD 双群（double group，96 阶）及各 little group 在所有 irrep 下、以少数生成元写出的矩阵表示种子，以及参考动量 → 群元的旋转字典 |
| `lattice/symmetry/utils.py` | 211 | 群表生成辅助（矩阵群 → 乘法表、代码序列化）与共线判定/归一化工具（numpy 版与 sympy 版） |
| `lattice/symmetry/sympy_utils.py` | 172 | 对含自定义 sympy Symbol（`HadronIrrepRow`/`Operator` 等）的表达式做积式基底收集、精确系数矩阵与 RREF 求线性无关组，以及 `Pow` → `Mul` 展开 |
| `lattice/symmetry/gen_hardcoded_rep.py` | 487 | irrep 的"在线生成器"：从生成元构造 OhD 双群与各 little group 的 irrep、Wigner 旋转、行间 connection、OhD→little group 的 subduction（约化）矩阵 |
| `lattice/symmetry/hardcoded_rep.py` | 15735 | 生成代码（冻结产物）：96 元双群的群元名表与乘法表、Fermion 4×4 表示、全部 irrep、refRotateDict、行连接表、约化表、`gauge_link` 表，均由 `lazy_object_proxy.Proxy` 惰性构建 |
| `lattice/symmetry/two_particle.py` | 122 | 在连续 J^M(LS) 分类下构造两粒子（PV、VV 等）插值算符的 sympy 公式（CG 系数 + 球谐函数 → 圆基 → Cartesian 基） |
| `lattice/group_projection.py` | 319 | 把含 `HadronIrrepRow`/`Diagram` 的 sympy 表达式按动量小群 irrep 的指定行做群投影，并把多强子算符乘积投影为小群 irrep 本征算符 |
| `lattice/spatial_structure.py` | 337 | 定义"动量 + irrep + 行 + 宇称 + tag"意义上的强子空间结构符号类型 `HadronIrrep` / `HadronIrrepRow`，后者自带小群变换法则 |
| `lattice/flavor_structure.py` | 142 | 定义味空间符号类型：单 quark/antiquark `Qurak`、传播子占位 `Propagator`、强子味结构 `HadronFlavorStructure` |
| `lattice/hadron.py` | 229 | 把空间结构表达式与味结构表达式配对成 `Hadron` 插值算符，提供时间设定、共轭，及批量生成关联函数表达式矩阵 `gen_correlator` |

### 1.2 依赖关系

内部依赖（箭头表示 "import"）：

```
group_generator.py  ←── gen_hardcoded_rep.py  ←──┬── hardcoded_rep.py（生成器读冻结表）
       ↑              ↑        ↑                 │
       │              │        └─ hardcoded_rep 的 OhD_mul/OhD_inv/各 irrep 表
       │              │
       │              ├── group_projection.py ──→ spatial_structure.py（HadronIrrepRow）
       │              ├── spatial_structure.py ──→ base_types.Tag
       │              ├── hadron.py ──→ flavor_structure.py ──→ quark_diagram.py
       │              │      └────→ quark_diagram.diagram_simplify / diagram_vertice_replace / quark_contract
       │              ├── lattice/insertion/__init__.py、lattice/insertion/gamma.py、lattice/insertion/gauge_link.py
       │              └── symmetry/__init__.py（再导出枢纽）
       └── symmetry/__init__.py
```

要点：

- `group_generator.py` 只 import `sympy` 与 `numpy`（`lattice/symmetry/group_generator.py:1-2`），不依赖仓库其它模块。
- `hardcoded_rep.py` 与 `gen_hardcoded_rep.py` 互相咬合：生成器 import 冻结表的 `OD_irreps`、`Dic*_irreps`、`OhD_inv`、`OhD_mul`、`refRotateDict`、`little_group_reduction_map`、`irrep_row_connection_dict`（`lattice/symmetry/gen_hardcoded_rep.py:4-10`）。
- `group_projection.py` import `lattice.quark_diagram.Diagram`（`lattice/group_projection.py:8`）、`gen_hardcoded_rep`（`lattice/group_projection.py:9-12`）、`hardcoded_rep`（`lattice/group_projection.py:13-14`）、`sympy_utils.find_linear_independent_exprs`（`lattice/group_projection.py:15`）、`spatial_structure.HadronIrrepRow`（`lattice/group_projection.py:16`），并在函数内局部 import `HadronIrrep`（`lattice/group_projection.py:98`、`lattice/group_projection.py:190`，避免循环依赖）。
- `spatial_structure.py` import `gen_hardcoded_rep`（`lattice/spatial_structure.py:10-14`）与 `base_types.Tag`（`lattice/spatial_structure.py:15`）；其中 `reductionToLittleGroup` 在 `lattice/spatial_structure.py:12` 被 import 但模块内无调用点。
- `hadron.py` import `gen_hardcoded_rep`（`lattice/hadron.py:10-14`，其中 `reductionToLittleGroup`/`wignerRotate` 未被使用）、`sympy_utils.convert_pow_to_mul`（`lattice/hadron.py:15`）、`base_types.Tag`（`lattice/hadron.py:16`）、`flavor_structure.HadronFlavorStructure`（`lattice/hadron.py:17`）、`spatial_structure.HadronIrrepRow, HadronIrrep`（`lattice/hadron.py:18`）、`quark_diagram.diagram_simplify, diagram_vertice_replace, quark_contract`（`lattice/hadron.py:19`）。
- `lattice/symmetry/__init__.py` 是再导出枢纽（`lattice/symmetry/__init__.py:7-66`），文件头注释声明按历史 import 顺序排列以保持 `import *` 时代的名字优先级（`lattice/symmetry/__init__.py:1-6`）。注意 `split_first_term/split_mul/split_expression` 导出的是 utils 版本（`lattice/symmetry/__init__.py:21-33`），sympy_utils 的同名变体不在此导出。

### 1.3 总体数据流

1. **群的静态层**：`group_generator.py`（生成元种子）→ `gen_hardcoded_rep.py`（在线构造/查表）→ `hardcoded_rep.py`（冻结的 96 元双群与全部 irrep/连接/约化表）。群代数一律以"群元名字符串 + 乘法表"进行（`OhD_mul/OhD_inv`，`lattice/symmetry/hardcoded_rep.py:9533-9542`）。
2. **动量分派**：任意整数动量经 `momentunSymplify` 归一 → 按 |p|²∈{0,1,2,3,5,6} 分派 little group（Oh/Dic4/Dic2/Dic3/C4₂/C4₁，`lattice/symmetry/gen_hardcoded_rep.py:186-206`）；`refRotateDict` + `wignerRotate` 把任意动量的小群 irrep 挂接到参考动量的表上。
3. **算符符号层**：`HadronIrrep`/`HadronIrrepRow`（`lattice/spatial_structure.py`）是 irrep 行算符的 sympy 符号，`transform` 实现小群协变变换；`HadronFlavorStructure` 是味内容符号。
4. **投影层**：`group_projection.py` 用 d_Γ/|G| · Σ_g D_00(g) U(g) 把任意表达式投到小群 irrep 的行 0，再用 `irrep_row_connection_dict` 换到任意行；`hadron_little_group_projection` 处理多强子乘积算符。
5. **组装层**：`Hadron`（`lattice/hadron.py`）= irrep 行表达式 × 味结构；`gen_correlator` 把 Hadron 列表的直积展开成关联函数对象 ndarray，味收缩交给 `quark_diagram.quark_contract` 生成 `Diagram`。
6. **旁路**：`two_particle.py` 独立产出连续 J^LS 两粒子算符公式（仅 example 使用，`example/gen_two_particle_operators.py:2,5,9,13`）；`utils.antisymmetric_tensor`、`falvor2irrep`/`falvor3irrep`、`group_projection.diagonalize_Cij`、`hardcoded_rep.gauge_link` 当前均无运行时调用者。

---

## ② 公开 API 参考

### 2.1 `lattice/symmetry/group_generator.py`

本文件无任何 `def`/`class`，全部为模块级 dict 数据。

| 名称 | 类型 | 出处 | 内容 |
|---|---|---|---|
| `Fermion_generator` | `dict[str, Matrix]` | `lattice/symmetry/group_generator.py:5-24` | 键 `"c4y"`、`"c4z"`、`"inviden"`。`c4y` 为两个 2×2 旋转块的块对角矩阵；`c4z` 是对角相位矩阵，元素含 √2(1−i)/2 = e^(−iπ/4)（`lattice/symmetry/group_generator.py:13-18`）；`inviden = diag(1,1,-1,-1)`（`lattice/symmetry/group_generator.py:19-24`） |
| `OhD_generator` | `dict[irrep, dict[gen, Matrix]]` | `lattice/symmetry/group_generator.py:26-95` | 转动子群 O 的 8 个 irrep：`A_1, A_2, E, T_1, T_2, G_1, G_2, H`，各给生成元 `c4y`、`c4z`。维数：A_1/A_2 为 1×1，E/G_1/G_2 为 2×2，T_1/T_2 为 3×3，H 为 4×4 |
| `Dic4_generator` | 同上 | `lattice/symmetry/group_generator.py:98-133` | little group of [0,0,1]，生成元 `c4z`、`invc2y`；irrep：`A_1, A_2, B_1, B_2, E, G_1, G_2` |
| `Dic2_generator` | 同上 | `lattice/symmetry/group_generator.py:136-157` | little group of [0,1,1]，生成元 `c2e`、`invc2f`；irrep：`A_1, A_2, B_1, B_2, G` |
| `Dic3_generator` | 同上 | `lattice/symmetry/group_generator.py:160-190` | little group of [1,1,1]，生成元 `c3delta`、`invc2b`；irrep：`A_1, A_2, F_1, F_2, E, G` |
| `C4_generator1` | 同上 | `lattice/symmetry/group_generator.py:193-204` | 单一生成元 `invc2f`；irrep `A_1, A_2, F_1, F_2`（对应 p²=6 参考点 [2,1,1]） |
| `C4_generator2` | 同上 | `lattice/symmetry/group_generator.py:206-215` | 单一生成元 `invc2x`；同上 irrep（对应 p²=5 参考点 [0,1,2]） |
| `irrep_generators` | `dict[str, dict]` | `lattice/symmetry/group_generator.py:218-226` | 动量字符串 → 生成元表：`"0,0,0"→OhD_generator`、`"0,0,1"→Dic4_generator`、`"0,1,1"→Dic2_generator`、`"1,1,1"→Dic3_generator`、`"2,1,1"→C4_generator1`、`"0,1,2"→C4_generator2` |
| `refRotateDict` | `dict[str, dict[str, str]]` | `lattice/symmetry/group_generator.py:229-291` | 外层键为参考动量字符串，内层把该动量 star（等价类）里每个动量映射到群元名，如 `"0,0,1": {"0,0,1":"iden", "1,0,0":"c4y", "-1,0,0":"c4y^-1", ...}`（`lattice/symmetry/group_generator.py:235-242`）；覆盖 6 个参考动量类（`"0,0,0"` 仅 1 条，`"2,1,1"` 与 `"0,1,2"` 各 24 条，`lattice/symmetry/group_generator.py:254-291`）。`test/test_symmetry.py:317-327` 验证 matrix_group[rotation] @ p_ref == p（T_1 三维表示下） |
| `falvor2irrep` | `dict[int, list[ndarray]]` | `lattice/symmetry/group_generator.py:294-302` | 键 1、3，值为 numpy object dtype 矩阵列表：1→[I/√2]，3→[Pauli 型矩阵 + diag(1,-1)/√2]。键名拼写为 "falvor"（仓库原样拼写） |
| `falvor3irrep` | `dict[int, list[ndarray]]` | `lattice/symmetry/group_generator.py:305-315` | 键 1、8，值为 3×3 矩阵列表，8 维部分即 Gell-Mann 型生成元组合 |

说明：由 e^(−iπ/4) 的四次方 = −1 可得 `c4z^4 = -1`（对角相位自乘 4 次），即该 4×4 表示中 2π 旋转给出 −1，这是 spinor/双群行为；`inviden` 形如 Dirac 表象的 γ⁰。`falvor2irrep`/`falvor3irrep` 除被 `lattice/symmetry/__init__.py:17-18` 再导出外，全仓库无使用者（grep "falvor" 仅命中 `__init__.py` 与定义处）。

### 2.2 `lattice/symmetry/utils.py`

| 函数 | 签名 | 出处 | 参数与返回值 |
|---|---|---|---|
| `antisymmetric_tensor` | `(n) -> np.ndarray` | `lattice/symmetry/utils.py:10-27` | n：张量阶数。返回 shape `(n,)*n`、`dtype=object` 的全反对称张量（Levi-Civita：按每个置换的逆序对数填 ±1）。本目录内无调用者，属预留工具 |
| `generate_hardcoded_code` | `(vals, indent=4) -> str` | `lattice/symmetry/utils.py:30-50` | 递归把嵌套 dict / `sympy.Matrix`（`Matrix({value.tolist()})`，`lattice/symmetry/utils.py:43`）/其它 repr 值序列化为 Python 源码文本。这是 `hardcoded_rep.py` 这类生成代码的序列化器；当前仓库内无调用点 |
| `multiplicationTable` | `(matrix_group) -> list[list[int]]` | `lattice/symmetry/utils.py:53-78` | matrix_group：`dict[name, Matrix]`。计算 `group[keys[i]] @ group[keys[j]]`，存匹配到的元素下标；匹配判据 `abs((result - candidate).norm()).evalf() < 0.1`（`lattice/symmetry/utils.py:71`）；找不到则 `print` 并 `exit()`（`lattice/symmetry/utils.py:76`）。容差松、失败路径直接 exit，仅适合离线生成脚本 |
| `are_collinear` | `(arrays) -> bool` | `lattice/symmetry/utils.py:84-98` | 以第 0 个为基准逐个 `np.cross` 判共线；少于 2 个默认 True |
| `select_nonzero_vector` | `(arrays) -> Optional[np.ndarray]` | `lattice/symmetry/utils.py:101-110` | 返回首个范数 > 0 的向量 |
| `normalize_array` | `(arr) -> np.ndarray` | `lattice/symmetry/utils.py:113-121` | 按首个非零分量的符号归一化（规范到"首分量取正"） |
| `check_and_normalize_arrays` | `(arrays) -> Optional[np.ndarray]` | `lattice/symmetry/utils.py:124-143` | 不共线则 `raise ValueError("Arrays are not collinear.")`（`lattice/symmetry/utils.py:134`）；全零返回 None |
| `are_collinear_and_normalize` | `(expressions: List[Expr]) -> Optional[Expr]` | `lattice/symmetry/utils.py:148-190` | 过滤零表达式；对每个非零项检查 `simplify(expr/base)` 是否 `is_constant()`（`lattice/symmetry/utils.py:159-161`）；共线则用 `Poly.LC()` 或 `as_coeff_mul()` 提取首系数归一化；全零返回 0（`lattice/symmetry/utils.py:155`） |
| `split_first_term` | `(expr: Expr) -> Expr` | `lattice/symmetry/utils.py:193-199` | 取 `Add` 的第一项 |
| `split_mul` | `(expr: Expr) -> list` | `lattice/symmetry/utils.py:202-209` | 按 `Mul` 的 args 拆因子 |
| `split_expression` | `(expr)` | `lattice/symmetry/utils.py:211` | 拆表达式（Add/Mul 两级） |

注意：`split_first_term/split_mul/split_expression` 在 `lattice/symmetry/sympy_utils.py:6-44` 有同名实现（sympy_utils 的 `split_mul` 额外把平方展开成两个因子，`lattice/symmetry/sympy_utils.py:31-32`）；`lattice/symmetry/__init__.py:21-33` 从 utils 导入这三个名字，故 `from lattice.symmetry import split_mul` 得到的是 utils 版本。

### 2.3 `lattice/symmetry/sympy_utils.py`

| 函数 | 签名 | 出处 | 参数与返回值 |
|---|---|---|---|
| `split_first_term` | `(expr: Expr) -> Expr` | `lattice/symmetry/sympy_utils.py:6-17` | `Add` → `expr.args[0]`；非 `Add` 原样返回 |
| `split_mul` | `(expr: Expr) -> list` | `lattice/symmetry/sympy_utils.py:21-33` | `Mul` → args；`expr.is_Pow` 且指数为 2 → `[base, base]`（`lattice/symmetry/sympy_utils.py:31-32`）；否则 `[expr]` |
| `split_expression` | `(expr)` | `lattice/symmetry/sympy_utils.py:35-44` | 先 `split_first_term` 再 `split_mul`，返回因子列表 |
| `collect_product_basis` | `(exprs: list[Expr]) -> list[Expr]` | `lattice/symmetry/sympy_utils.py:51-69` | 把每个表达式按 `Add.make_args`/`Mul.make_args` 拆项拆因子，将 `Symbol` 类因子的乘积作为基、数值系数另计，去重收集基列表（`lattice/symmetry/sympy_utils.py:54-68`）。返回基底表达式列表 |
| `build_coefficient_matrix` | `(exprs: list[Expr], basis_map: dict) -> list[list]` | `lattice/symmetry/sympy_utils.py:71-96` | 逐表达式逐项，用 `str(basis)` 查 `basis_map` 定位列，精确 sympy 系数累加（`lattice/symmetry/sympy_utils.py:90-92`）。返回系数矩阵（list of list） |
| `find_linear_independent_exprs` | `(exprs: list[Expr]) -> list[Expr]` | `lattice/symmetry/sympy_utils.py:98-136` | 流程＝过滤零表达式（`expr != 0` 直接比较，异常则保守当非零，`lattice/symmetry/sympy_utils.py:103-112`）→ 收集积式基底 → 建系数矩阵 → `Matrix(...).rref()`（`lattice/symmetry/sympy_utils.py:124`）→ 用 pivot 行重组出线性无关表达式（`lattice/symmetry/sympy_utils.py:125-133`）。返回线性无关表达式列表（全被过滤时返回空列表，`lattice/symmetry/sympy_utils.py:114-116`） |
| `expr_simplify` | `(expr)` | `lattice/symmetry/sympy_utils.py:139-151` | 只 print 系数矩阵和基、无 return 值——调试残留，不宜作为 API 使用 |
| `convert_pow_to_mul` | `(expr)` | `lattice/symmetry/sympy_utils.py:152-172` | 递归把正整数幂 `x**n` 展开为 `Mul(x, x, ..., evaluate=False)`（`lattice/symmetry/sympy_utils.py:163-166`）；Atom 与 `sympy.physics.quantum.Operator` 原样返回（`lattice/symmetry/sympy_utils.py:168-169`）；其余节点递归重建 `expr.func(*args)`（`lattice/symmetry/sympy_utils.py:171-172`）。返回转换后的表达式 |

使用者：`lattice/group_projection.py:15`（`find_linear_independent_exprs`）、`lattice/hadron.py:15`（`convert_pow_to_mul`）、`lattice/insertion/gauge_link.py:8`、test 系列文件。`lattice/flavor_structure.py:6` 也 import 了 `convert_pow_to_mul` 但在模块内无调用点（未使用 import）。

### 2.4 `lattice/symmetry/gen_hardcoded_rep.py`

| 函数 | 完整签名 | 出处 |
|---|---|---|
| `genMatrixGroupOhD` | `def genMatrixGroupOhD(c4y: Matrix, c4z: Matrix, inv: Matrix = None)` | `lattice/symmetry/gen_hardcoded_rep.py:23` |
| `genIrrepOhD` | `def genIrrepOhD(irrep_name: Literal["A_1","A_2","E","T_1","T_2","G_1","G_2","H"], parity: Literal[1,-1,None] = None, is_hardcoded: bool = True)` | `lattice/symmetry/gen_hardcoded_rep.py:73` |
| `littleGroup` | `def littleGroup(fixed_point=[0,0,0], group=None, elem=True)` | `lattice/symmetry/gen_hardcoded_rep.py:100` |
| `momentunSymplify` | `def momentunSymplify(p)` | `lattice/symmetry/gen_hardcoded_rep.py:116` |
| `genR_ref` | `def genR_ref(p_ref, group=None, all=False)` | `lattice/symmetry/gen_hardcoded_rep.py:140` |
| `wignerRotate` | `def wignerRotate(p_i: list, ele: str)` | `lattice/symmetry/gen_hardcoded_rep.py:160` |
| `genLittleGroupIrrep` | `def genLittleGroupIrrep(p, irrep_name, parity=None, p_ref=None, is_hardcoded=True, p_ref_irrep=False)` | `lattice/symmetry/gen_hardcoded_rep.py:182` |
| `gen_connection` | `def gen_connection(momentum, irrep_name)` | `lattice/symmetry/gen_hardcoded_rep.py:250` |
| `reductionToLittleGroup` | `def reductionToLittleGroup(momentum, OhD_irep_name, parity, little_group_irrep_name, is_hardcoded=True)` | `lattice/symmetry/gen_hardcoded_rep.py:388` |

（均无返回类型注解。）

| 函数 | 参数含义 | 返回值 |
|---|---|---|
| `genMatrixGroupOhD` | `c4y`/`c4z`：两个 C4 生成元在某表示下的矩阵；`inv`：宇称元矩阵（可选） | `dict[str, Matrix]`：群元名 → 矩阵；纯转动 24 元，加 `r_2pi @ g` 的 r-前缀副本，`inv` 非 None 时再加 `inv @ g` 的 inv-前缀副本，扩展至 96 元 |
| `genIrrepOhD` | `irrep_name`：O 群 8 个 irrep 之一；`parity`：1/-1/None（决定 g/u）；`is_hardcoded`：True 查冻结表 | `dict[str, Matrix]`：该 irrep 的群元 → 矩阵表示 |
| `littleGroup` | `fixed_point`：固定点动量；`group`：用于判定固定点的群表示（默认 `genIrrepOhD("T_1", -1)`，即三维极向量表示的宇称版）；`elem`：False 时只留键集、值存 None | `dict[str, Matrix|None]`：保持 fixed_point 不动（`(g @ p - p).norm() < 0.01`，`lattice/symmetry/gen_hardcoded_rep.py:107`）的群元集合 |
| `momentunSymplify` | `p`：长度 3 的整数动量 | 长度 3 的整数 list：同一 little group 类的规范代表 |
| `genR_ref` | `p_ref`：参考动量；`group`：枚举用群表示；`all`：True 时每个像动量记录所有到达群元列表 | `dict[str, str|list[str]]`：像动量字符串 → 群元名 |
| `wignerRotate` | `p_i`：初动量；`ele`：把 p_i 转到 p_f 的群元名 | str：小群元素 W = R_{p_f}^{-1} g R_{p_i} 的群元名；查不到参考类则 `raise NotImplementedError`（`lattice/symmetry/gen_hardcoded_rep.py:178`） |
| `genLittleGroupIrrep` | `p`：实际动量；`irrep_name`：小群 irrep 名；`parity`：宇称；`is_hardcoded`：是否查冻结表（p²=5/6 强制在线）；`p_ref_irrep`：True 时返回参考动量的小群 irrep 本身 | `dict[str, Matrix]`：动量小群 irrep 的群元 → 矩阵 |
| `gen_connection` | `momentum`：参考动量；`irrep_name`：小群 irrep 名 | `list[list[(coeff, group_element)]]`：`irrep_row_connection_dict` 兼容格式——第 j 项是 (系数, 群元) 元组列表，表示第 j 行标准基向量在该 irrep 下的展开。该函数含大量 `print`，是离线生成工具；仓库内唯一调用处是 `example/hardcoding_OhD.py:99`（整段被注释），运行时无人调用 |
| `reductionToLittleGroup` | `momentum`：动量；`OhD_irep_name`：OhD irrep 名；`parity`：宇称（1/-1）；`little_group_irrep_name`：小群 irrep 名；`is_hardcoded`：是否查冻结表 | subduction 矩阵（把 OhD irrep 行向量组合映射为小群 irrep 行向量）；硬编码路径查不到时 print 并返回 None（`lattice/symmetry/gen_hardcoded_rep.py:411-414`） |

运行时使用者：`reductionToLittleGroup` 被 `lattice/insertion/__init__.py:288`（`Insertion.little_group_projection`）调用；`lattice/spatial_structure.py:12` import 但模块内未见调用。

### 2.5 `lattice/symmetry/hardcoded_rep.py`（概述）

该文件是纯查找表的"冻结产物"，无待逐一罗列的 API；关键顶层符号如下：

| 名称 | 出处 | 内容 |
|---|---|---|
| `_create_group_element()` / `group_element` | `lattice/symmetry/hardcoded_rep.py:8-114` | 96 个群元名字符串：24 个转动元（`iden, c4x, c2x, c4x^-1, ..., c2b`）+ 24 个 r-前缀 + 48 个 inv-前缀；`group_element = Proxy(...)` |
| `OhD_mul(ele1, ele2) -> str` / `OhD_inv(ele) -> str` | `lattice/symmetry/hardcoded_rep.py:9533-9542` | `OhD_mul` 查 `OhD_Multiply_table[i][j]` 得群元名；`OhD_inv` 在 `OhD_Multiply_table[i]` 中找值为 0（iden 下标）的位置即逆元。全代码库对群做代数运算均以"字符串 + 乘法表"进行 |
| `_create_Fermion_rep` / `Fermion_rep` | `lattice/symmetry/hardcoded_rep.py:9545-10206` | 4×4 spinor 表示（开头矩阵与 `group_generator.Fermion_generator` 的相位结构一致，`lattice/symmetry/hardcoded_rep.py:9548-9560` 处 `c4x` 含 e^(−iπ/4) 相位） |
| `_create_OD_irreps` / `OD_irreps` | `lattice/symmetry/hardcoded_rep.py:10209-10894` | O 子群 8 个 irrep：`dict[群元名, Matrix]` |
| `_create_OhD_irreps` / `OhD_irreps` | `lattice/symmetry/hardcoded_rep.py:10897-11628` | 24 个条目：8 个纯转动 irrep + 16 个带 `g`/`u` 后缀的宇称 irrep（如 `"T_1u": Proxy(_create_OhD_irreps("T_1", -1))`，`lattice/symmetry/hardcoded_rep.py:11616-11617`） |
| `_create_Dic4_irreps` / `_create_Dic2_irreps` / `_create_Dic3_irreps` / `_create_C4_irreps1` / `_create_C4_irreps2` / `little_group_irreps` | `lattice/symmetry/hardcoded_rep.py:11630-11996` | 6 个动量类 little group 的全部 irrep；`little_group_irreps` 把 6 个动量类键映射到相应 irrep dict（`lattice/symmetry/hardcoded_rep.py:11989-11996`） |
| `refRotateDict` | `lattice/symmetry/hardcoded_rep.py:11998-12086` | 与 `group_generator.py:229-291` 相同数据的副本 |
| `irrep_row_connection_dict` | `lattice/symmetry/hardcoded_rep.py:12088-12140` | `dict[p_str][irrep_name] -> list[list[(coeff, group_element)]]`，例如 `"0,0,0"/"T_1" = [[(1,"iden")],[(1,"c4z")],[(-1,"c4y")]]`（`lattice/symmetry/hardcoded_rep.py:12092`）。语义（由 `test/test_symmetry.py:342-360` 的验证反推）：第 j 行标准基向量可写作 Σ c·D(g)·e_0 |
| `little_group_reduction_map`（含 `_Dic4/_Dic2/_Dic3`） | `lattice/symmetry/hardcoded_rep.py:12142-12269` | `dict[p_str][OhD irrep±parity][little group irrep] -> list[list[list[int]]]`：若干个矩阵，每个矩阵把 OhD irrep 的行向量组合映射为小群 irrep 的行向量（如 `"T_1g": {"A_2": [[[0,0,1]]], "E": [[[0,1,0],[-1,0,0]]]}`，`lattice/symmetry/hardcoded_rep.py:12142-12158`）。只覆盖 Dic4/Dic2/Dic3 三类动量 |
| `_create_gauge_link` / `gauge_link` | `lattice/symmetry/hardcoded_rep.py:12271-15735` | `dict[str, list]`，键形如 `'A_1g+'`、`'A_1g-'`（`lattice/symmetry/hardcoded_rep.py:12272-12274`），值最深为 `[系数, 群元下标]` 对的嵌套列表。全仓库 grep 显示该 Proxy 名未被任何其它模块 import（`lattice/insertion/gauge_link.py:7` 只 import `OD_irreps, irrep_row_connection_dict, group_element`）；其物理语义无法从代码确认 |

所有大表都由 `lazy_object_proxy.Proxy` 包裹、首次属性访问时才构建（`lattice/symmetry/hardcoded_rep.py:1`；`group_element = Proxy(_create_group_element)` 在 `lattice/symmetry/hardcoded_rep.py:114`；`OhD_Multiply_table = Proxy(...)` 在 `lattice/symmetry/hardcoded_rep.py:9530`；`Fermion_rep = Proxy(...)` 在 `lattice/symmetry/hardcoded_rep.py:10206`；`OD_irreps = Proxy(...)` 在 `lattice/symmetry/hardcoded_rep.py:10894`；`gauge_link = Proxy(_create_gauge_link)` 在 `lattice/symmetry/hardcoded_rep.py:15735`）。

表的再生成配方：`utils.generate_hardcoded_code`（`lattice/symmetry/utils.py:30-50`）是序列化器，每个大表对应 `gen_hardcoded_rep.py` 一个函数的产物（乘法表↔`multiplicationTable`；irrep↔`genIrrepOhD`/`genLittleGroupIrrep`；connection↔`gen_connection`；约化表↔`reductionToLittleGroup`）；`example/hardcoding_OhD.py:39-125`（整个文件为注释掉的配方）按顺序记录了再生成流程。生成脚本本体未入库。

### 2.6 `lattice/symmetry/two_particle.py`

| 函数 | 签名 | 出处 | 参数与返回值 |
|---|---|---|---|
| `list_from_mom2_max` | `(n)` | `lattice/symmetry/two_particle.py:19` | n：目标 \|p\|²。返回所有 \|p\|²=n 的整数动量 `[S(k), S(j), S(i)]`（two_particle.py:23-29；顺序是 z,y,x）。枚举范围 `product(range(-imax, imax+1), repeat=3)`，`imax=int(sqrt(n))` |
| `make_operator` | `(form, mom="p")` | `lattice/symmetry/two_particle.py:33` | form：`"P*"` 标量或 `"V*"` 矢量算符名；mom：动量标签。`form[0]=="P"` 返回 `[Operator(f"{form}({mom})")]`；`form[0]=="V"` 返回矢量算符的球分量（`lattice/symmetry/two_particle.py:38-44`）：V_0=iV_z、V_{+1}=−i(V_x+iV_y)/√2、V_{−1}=i(V_x−iV_y)/√2 |
| `rotation` | `(vec)` | `lattice/symmetry/two_particle.py:47` | 返回动量的 (θ, φ) = (arccos(z/r), atan2(y,x)) |
| `two_particle_circle_basis_JM` | `(op1, op2, mom2, J, M, L, Spin)` | `lattice/symmetry/two_particle.py:60` | 对固定 M 构造 Σ_{s1,s2,mL,p} ⟨L mL, Spin mS|J M⟩⟨S1 s1, S2 s2|Spin mS⟩ Y_{L mL}(θ,φ) op1(p)_{s1} op2(−p)_{s2}；`S1 = 0 if op1[0]=="P" else 1`（`lattice/symmetry/two_particle.py:61-62`），op2 取反冲动量 −p（`lattice/symmetry/two_particle.py:71`）。返回 sympy 表达式 |
| `two_particle_circle_basis` | `(op1, op2, mom2, J, L, Spin)` | `lattice/symmetry/two_particle.py:76` | 对 M=−J..J 逐个求和，返回长度 2J+1 的列表（`basis[M+J]`，`lattice/symmetry/two_particle.py:104`） |
| `two_particle_Cartesian_basis` | `(op1, op2, mom2, J, L, Spin)` | `lattice/symmetry/two_particle.py:110` | J=0 直接返回圆基；J=1 用 (B_0−B_{−1})(−i)/√2、(B_0+B_{−1})/√2、(−i)B_{+1} 把圆基换成实 Cartesian 基（`lattice/symmetry/two_particle.py:114-119`）；J>1 `raise NotImplementedError("TODO: J>1")`（`lattice/symmetry/two_particle.py:120-121`） |

依赖与使用者：import numpy、sympy（含 `sympy.physics.quantum.cg.CG`、`Ynm`、`Operator`，`lattice/symmetry/two_particle.py:7-14`）；不 import 仓库任何模块，也不被 `lattice/` 内任何模块 import。唯一使用者是 `example/gen_two_particle_operators.py:2,5,9,13`。test/ 下无 `test_two_particle.py`。

### 2.7 `lattice/group_projection.py`

| 函数 | 完整签名 | 出处 | 参数含义与返回值 |
|---|---|---|---|
| `operator_transform` | `def operator_transform(expr: Expr, group_element, time=None) -> Expr` | `lattice/group_projection.py:36` | expr：任意嵌套结构（list/ndarray/dict/tuple/`sympy.Matrix`/`Add`/`Mul`/`Pow`）；group_element：群元名（字符串）；time：可选时间参数，传给 `HadronIrrepRow.transform`。深度优先遍历，把 `HadronIrrepRow` 节点替换为其 `.transform(group_element, time)`、`Diagram` 节点替换为其 `.transform(group_element)`（`lattice/group_projection.py:77-82`），其余叶子原样返回（`lattice/group_projection.py:83-84`）。返回变换后的表达式 |
| `expr_little_group_projection` | `def expr_little_group_projection(expr, irrep_name, row_idx, parity=None, p_ref=False)` | `lattice/group_projection.py:89` | expr：含 `HadronIrrepRow` 的表达式；irrep_name/row_idx/parity：目标小群 irrep 与行；p_ref：True 时额外返回动量信息。返回投影后的表达式；`p_ref=True` 时返回 `(projected, p_str, ref_p_str)`（`lattice/group_projection.py:125-127`） |
| `multi_exprs_little_group_projection` | `def multi_exprs_little_group_projection(expr_list, irrep_name, row_idx, parity=None, single_result=False)` | `lattice/group_projection.py:147` | expr_list：多个表达式；single_result：True 时拿到第一个非零投影即返回。逐个投影，跳过 0（`lattice/group_projection.py:158-160`）；否则 `find_linear_independent_exprs` 去掉线性相关（`lattice/group_projection.py:166`），并对每条无关表达再执行行-0→row_idx 的连接组合与转回旋转（`lattice/group_projection.py:167-172`）。返回投影表达式列表 |
| `hadron_little_group_projection` | `def hadron_little_group_projection(hadron_irreps: List, irrep_name, row_idx, parity=None, single_result=False)` | `lattice/group_projection.py:189` | hadron_irreps：多个 `HadronIrrep`（多强子）。求总动量并取 `littleGroup(momentum_total)`（`lattice/group_projection.py:196-198`）；用三维 T_1 表示枚举小群作用下的所有动量组态像（dict 去重，`lattice/group_projection.py:199-205`）；对每个像组态，为每个强子按其 irrep 维数（`hadron_irrep.lenth`）生成 `HadronIrrep[j]` 行符号，做笛卡尔积 `list(product(*hadrons_list))` 得 `Mul(*rows)` 表达式集合（`lattice/group_projection.py:206-221`）；交给 `multi_exprs_little_group_projection`（`lattice/group_projection.py:236-238`）。返回投影表达式列表 |
| `diagonalize_Cij` | `def diagonalize_Cij(expr)` | `lattice/group_projection.py:249` | 输入形如 Σ_{ij} C_{ij} A_i B_j 的表达式：按 symbol 名前缀（下划线/数字前截断，`lattice/group_projection.py:274-279`）自动分成 A、B 两族（`lattice/group_projection.py:281-296`）；用 `expr.coeff(A_i*B_j)` 提取 C（`lattice/group_projection.py:301-306`）；C 对称且方阵时 `C.diagonalize()` 得 P、D，输出 Σ_i λ_i A'_i B'_i（A'=P^{-1}A，`lattice/group_projection.py:308-315`）；否则用 `singular_value_decomposition()`，新基用 U/V 的转置或逆（`lattice/group_projection.py:316-319`）。grep 显示该函数在仓库内（除定义处）无任何调用者 |

### 2.8 `lattice/spatial_structure.py`

#### `HadronIrrep(Symbol)`

```python
class HadronIrrep(Symbol):
    def __new__(cls, hadron_name: str, momentum: List[int], irrep_name: str,
                parity: int, tag: Tag, dagger: bool = False)   # lattice/spatial_structure.py:22
    def __init__(cls, 同参)                                     # lattice/spatial_structure.py:45
```

| 参数 | 含义 |
|---|---|
| `hadron_name` | 强子名（进入符号名字符串） |
| `momentum` | 长度 3 的整数 list（`__init__` 内做拷贝） |
| `irrep_name` | cubic irrep 名，如 `"T_1"` |
| `parity` | `None`（不写宇称）/ `-1`（irrep 名带 `u` 后缀）/ 其它（带 `g` 后缀），见 `lattice/spatial_structure.py:23-42` |
| `tag` | `base_types.Tag`，含 `tag.tag` 与 `tag.time` |
| `dagger` | 是否 dagger 侧 |

属性：`hadron_name, momentum(List[int] 拷贝), irrep_name, parity, tag, dagger, lenth`。`lenth` 由 irrep 名首字母推断（`lattice/spatial_structure.py:66-72`）：`T*→3`、`G*/E*→2`、`H*→4`、其余→1；该规则与 group_generator.py 中各 irrep 矩阵维数一致（T_1/T_2 3×3、E/G_1/G_2 2×2、H 4×4、A_1/A_2 1×1）。其它方法：`__getitem__(row_idx) -> HadronIrrepRow`（`lattice/spatial_structure.py:119`）、`__eq__/__hash__/copy`（`lattice/spatial_structure.py:128-135`；`__eq__` 比较 hadron_name/momentum/irrep_name/parity/tag/dagger）。

#### `HadronIrrepRow(Symbol)`

```python
class HadronIrrepRow(Symbol):
    def __new__(cls, hadron_name: str, momentum: List[int], irrep_name: str,
                row_idx: int, parity: int, tag: Tag, dagger: bool = False)  # lattice/spatial_structure.py:139
    def __init__(cls, 同参)                                                  # lattice/spatial_structure.py:159
```

属性：`hadron_name, tag, momentum, irrep_name, row_idx, parity, dagger`，以及只读的 `rotate` 与 `little_group_matrix`。`__init__` 急切构建两个群数据（`lattice/spatial_structure.py:217-218`）：`self._rotate = genLittleGroupIrrep([0,0,0],"T_1",-1)`（完整 OhD 三维矩阵群，用于旋转动量）与 `self._little_group_matrix = genLittleGroupIrrep(momentum, irrep_name, parity, p_ref_irrep=True)`（该动量小群 irrep，参考系版）——每个 Row 符号构造都要跑一次 irrep 生成/查表。

方法：

| 方法 | 签名 | 出处 | 语义 |
|---|---|---|---|
| `transform` | `(group_element, time=None) -> Expr` | `lattice/spatial_structure.py:283` | 见 §4.4 |
| `__lt__/__gt__/__le__/__ge__` | `(other) -> bool` | `lattice/spatial_structure.py:297-306` | 排序约定：依次比较 `(tag.time, irrep_name, row_idx, tag.tag)` |
| `copy` | `() -> HadronIrrepRow` | `lattice/spatial_structure.py` | 重建同参数对象 |
| `conjugate` | `() -> Expr` | `lattice/spatial_structure.py:333-337` | 仅翻转 dagger |

### 2.9 `lattice/flavor_structure.py`

```python
Flavor = Literal["u", "d", "s", "c", "t", "b"]   # lattice/flavor_structure.py:7
```

| 类 | 完整签名 | 出处 | 语义 |
|---|---|---|---|
| `Qurak(Symbol)` | `__new__/__init__(cls, flavor: Flavor, tag: Tag, anti: bool, **assumptions)` | `lattice/flavor_structure.py:12-33` | 非交换 Symbol；字符串为 `f"{flavor}({tag.tag})"` 或 `f"\bar{{{flavor}}}({tag.tag})"`（`lattice/flavor_structure.py:14-17`）；属性 `flavor/tag/anti`。类名 `Qurak` 为仓库原样拼写；返回注解写作 `-> None` 但实际返回对象（签名原文如此） |
| `Propagator(Symbol)` | `__new__/__init__(cls, flavor: Flavor, source_tag: Tag, sink_tag: Tag, **assumptions)` | `lattice/flavor_structure.py:36-67` | 非交换 Symbol `f"S^{flavor}({sink_tag.tag}, {source_tag.tag})"`（`lattice/flavor_structure.py:43-45`）；属性 `tag` 在 source/sink 同 time 时为 `"S^f_\mathrm{local}"` 否则 `"S^f"`（`lattice/flavor_structure.py:60-63`）——把"local 传播子"编码进符号名，供下游按名字筛选（如 `test/test_group_projection.py:143`） |
| `HadronFlavorStructure(Operator)` | `__new__/__init__(cls, flavor_str: str, time: int = 0)`；属性 `flavor_str, time, baryon_num, quark_list, anti_quark_list`；`conjugate() -> "HadronFlavorStructure"` | `lattice/flavor_structure.py:71-142` | 按 `flavor_str` 解析：`"bar{uds}"` 型 → `baryon_num=-1`、`quark_list=[]`、`anti_quark_list=[u,d,s]`（`lattice/flavor_structure.py:92-95`）；长度 3（如 `"uds"`）→ 重子，`baryon_num=1`（`lattice/flavor_structure.py:96-99`）；长度 2（如 `"ud"`）→ 介子，`baryon_num=0`、`quark_list=[flavor_str[1]]`（第 2 个字符是 quark）、`anti_quark_list=[flavor_str[0]]`（`lattice/flavor_structure.py:100-103`）。因此 `"ud"` 语义是 ūd——字符串首个字符是反夸克（`test/test_flavor_structure.py:84-90` 断言 `quark_list==["d"], anti_quark_list==["u"]` 佐证）。`conjugate()`（`lattice/flavor_structure.py:110-124`）：介子 `XY`→`YX`；重子→`bar{uds}`；反重子→`uds` |

`HadronFlavorStructure` 还定义 `_eval_is_zero/_eval_is_infinite/_eval_is_extended_real/_eval_is_finite`（`lattice/flavor_structure.py:126-142`）：sympy 假设钩子，声明该算符型 Symbol 恒非零、有限、实——防止 sympy 自动把含它的表达式化简为 0。

### 2.10 `lattice/hadron.py`

#### `Hadron`

```python
class Hadron:
    def __init__(self, irep_row: Expr, flavor_structure: Expr)   # lattice/hadron.py:23
```

| 方法 | 签名 | 出处 | 语义 |
|---|---|---|---|
| `irrep_row` / `flavor_structure` | property | `lattice/hadron.py:23-37` | 两个字段：irrep 行表达式与味结构表达式，均为 sympy 表达式 |
| `set_time` | `(time: int) -> "Hadron"` | `lattice/hadron.py:39` | 返回新对象、不改原对象 |
| `conjugate` | `() -> "Hadron"` | `lattice/hadron.py:55` | 返回新对象、不改原对象 |
| `copy` | `() -> "Hadron"` | `lattice/hadron.py:64` | 返回新对象、不改原对象 |

#### 模块级函数

| 函数 | 完整签名 | 出处 | 语义 |
|---|---|---|---|
| `set_time_in_expr` | `def set_time_in_expr(expr: Expr, time: int) -> Expr` | `lattice/hadron.py:73` | `sp.preorder_traversal` 找到 `HadronIrrepRow`（重建同参数、`Tag(tag.tag, time)`）与 `HadronFlavorStructure`（重建同 flavor_str、新 time），`expr.xreplace` 替换。返回新表达式 |
| `set_time_in_list` | `def set_time_in_list(hadron_list: List[Hadron], time: int) -> List[Hadron]` | `lattice/hadron.py:103` | 对每个 Hadron 调 `set_time`，返回新列表 |
| `operator_conjugate` | `def operator_conjugate(expr: Expr) -> Expr` | `lattice/hadron.py:117` | `Add` 逐项；`Mul` 因子**逆序**共轭（`factors[::-1]`，`lattice/hadron.py:122`）——算符顺序反转的（反）厄米共轭；`Pow` 只共轭底；`HadronIrrepRow`/`HadronFlavorStructure` 走各自 `.conjugate()`；其它叶子 `expr.conjugate()` |
| `set_dagger_in_list` | `def set_dagger_in_list(hadron_list: List[Hadron], dagger: bool) -> List[Hadron]` | `lattice/hadron.py:134` | dagger=True 时对每个 Hadron 调 `conjugate()`，否则 `copy()` |
| `gen_correlator` | `def gen_correlator(hadrons: List[List[Hadron]], time_slice_list=None, dagger_list: List[bool] = None) -> np.ndarray` | `lattice/hadron.py:149` | 见 §4.5 |

---

## ③ 数据结构与数组形状约定

| 数据结构 | 形状/格式 | 出处 |
|---|---|---|
| 群表示 dict | `dict[str, Matrix]`：群元名字符串 → 该表示下的矩阵；群元名形如 `"c4x"`、`"c3delta^-1"`、前缀 `r`/`inv` 的合成元 | `lattice/symmetry/gen_hardcoded_rep.py` 各函数返回值 |
| 群元名 | 字符串；96 个 = 24 个纯转动 + 24 个 r-前缀 + 48 个 inv-前缀 | `lattice/symmetry/hardcoded_rep.py:10-113` |
| 乘法表 | `OhD_Multiply_table`：96×96 整数下标表，`OhD_mul(ele1, ele2)` 查表得群元名 | `lattice/symmetry/hardcoded_rep.py:9530-9542` |
| 动量 | 长度 3 的整数 list / sympy `Matrix`；动量字符串 `",".join(str)` 作为各类字典的键 | `lattice/symmetry/gen_hardcoded_rep.py`（核心数据结构约定） |
| irrep 行连接表 | `irrep_row_connection_dict[p_str][irrep_name] -> list[list[(coeff, group_element)]]`：外层第 j 项对应 irrep 第 j 行标准基向量，内层是 (系数, 群元名) 元组列表 | `lattice/symmetry/hardcoded_rep.py:12088-12140` |
| 约化表 | `little_group_reduction_map[p_str][OhD irrep±g/u][little group irrep] -> list[list[list[int]]]`：若干个整数矩阵，每个把 OhD irrep 的行向量组合映射为小群 irrep 的行向量 | `lattice/symmetry/hardcoded_rep.py:12142-12269` |
| irrep 维数约定 | A_1/A_2→1，E/G_1/G_2→2，T_1/T_2→3，H→4；`HadronIrrep.lenth` 按同名规则从 irrep 名首字母推断 | `lattice/symmetry/group_generator.py:26-95`、`lattice/spatial_structure.py:66-72` |
| `antisymmetric_tensor(n)` | shape `(n,)*n`、`dtype=object` | `lattice/symmetry/utils.py:10-27` |
| `list_from_mom2_max(n)` | 长度不定的动量列表，每项 `[S(k), S(j), S(i)]`（z,y,x 顺序） | `lattice/symmetry/two_particle.py:23-29` |
| `two_particle_circle_basis` 返回 | 长度 2J+1 的 sympy 表达式列表（`basis[M+J]`） | `lattice/symmetry/two_particle.py:104` |
| `gen_correlator` 返回 | `np.ndarray(shape=tuple(len(hadrons[i])...), dtype=object)`：N 点函数广义矩阵，每元素是 sympy 表达式 | `lattice/hadron.py:163-165` |
| 符号类型 | `HadronIrrep`/`HadronIrrepRow`/`Qurak`/`Propagator` 均为非交换 sympy Symbol（`commutative=False`，`lattice/spatial_structure.py:33`、`lattice/spatial_structure.py:151`、`lattice/flavor_structure.py:14-17`、`lattice/flavor_structure.py:43-45`），可直接参与 `Add/Mul/Pow` 并被 `operator_transform` 遍历 | — |
| 符号名字符串 | parity 为 None：`f"{hadron}({irrep}({tag.tag}),t={tag.time},{tuple(momentum)},{dagger})"`；parity==-1 时 irrep 名带 `u` 后缀、否则带 `g` 后缀；Row 版本再带 dagger 与 `[{row_idx}]` | `lattice/spatial_structure.py:23-42`、`lattice/spatial_structure.py:140-157` |

---

## ④ 算法与流程

### 4.1 `genMatrixGroupOhD`：从生成元构造 96 元双群

由 `c4y, c4z` 两个生成元构造转动子群 24 元：`c4x = c4z^{-1} c4y c4z`、`c3 = c4x c4y`、各 `c3x/c3y/c3z = (c4? c4?)^{-1}`、6 个 `c2*` 为两生成元积（`lattice/symmetry/gen_hardcoded_rep.py:26-44`）；全部 `sp.simplify`（`lattice/symmetry/gen_hardcoded_rep.py:45`）。随后 `r_2pi = c4x^4`（`lattice/symmetry/gen_hardcoded_rep.py:50`），对每个元素加 r-前缀副本 `r_2pi @ g`（`lattice/symmetry/gen_hardcoded_rep.py:51-53`）；若 `inv` 不为 None，再加 inv-前缀副本 `inv @ g`（`lattice/symmetry/gen_hardcoded_rep.py:55-58`）——即从 24 个纯转动扩展到 96 个双群元素。

### 4.2 `genIrrepOhD`：宇称 irrep 的实现

`is_hardcoded=True` 直接取 `OD_irreps[irrep_name]` 并按 parity 补 `inv{key} = parity * key`（`lattice/symmetry/gen_hardcoded_rep.py:77-81`）——宇称 irrep 中宇称作用是 ±1 乘纯转动矩阵，这正是 g/u 的实现；否则从 `OhD_generator[irrep_name]` 取生成元、`parity` 变成 `parity*I` 放入 `genMatrixGroupOhD`（`lattice/symmetry/gen_hardcoded_rep.py:83-94`）。

### 4.3 `momentunSymplify`：动量归一

把任意整数动量映射到"同一 little group 类"的规范代表（`lattice/symmetry/gen_hardcoded_rep.py:116-137`）。算法：按分量绝对值分类（`lattice/symmetry/gen_hardcoded_rep.py:120-124`），取符号（`lattice/symmetry/gen_hardcoded_rep.py:125`），若存在某绝对值出现 ≥2 次（`doublekey=1`，`lattice/symmetry/gen_hardcoded_rep.py:127-133`），再按不同绝对值排序依次乘以 `i+doublekey` 作为系数（`lattice/symmetry/gen_hardcoded_rep.py:135-137`）。目的：把 [1,1,2]、[2,2,4]、[−2,−1,1] 等都归一到同一代表，使 refRotateDict 等静态表可用有限键覆盖。（注意：对形如 [2,2,0] 的行为无单元测试覆盖，见 §5。）

### 4.4 `HadronIrrepRow.transform`：小群协变变换

```python
momentum_final = self._rotate[group_element] @ Matrix(self._momentum)
transform_matrix = self._little_group_matrix[wignerRotate(self._momentum, group_element)]
result = Σ_i transform_matrix[i, self._row_idx] * HadronIrrepRow(..., row=i, momentum=momentum_final, ...)
```

（`lattice/spatial_structure.py:283-295`。）语义：群元 g 把动量 p 转到 p' = R_g p；用 Wigner 共轭 W = R_{p'}^{-1} g R_p（`wignerRotate`，`lattice/symmetry/gen_hardcoded_rep.py:160-178`）把 g 换算成小群元素；算符行按小群 irrep 矩阵的**第 row_idx 列**混合：O_row(p) ↦ Σ_i D_Γ(W)_{i,row} O_i(p')。这是把算符（而非态）变换的约定——体现在取列而非取行；与 `test/test_group_projection.py:104-109` 的检验（rotated == Σ_i little_group[ele][i,j] * rows[i]）一致。

`wignerRotate` 的实现（`lattice/symmetry/gen_hardcoded_rep.py:160-178`）：把 p_i symplify 后在 `refRotateDict` 各参考类里查到 p_i 所在类与 ref_rotate(p_i)；返回 `OhD_mul(OhD_mul(OhD_inv(ref_rotate_p_f), ele), ref_rotate_p_i)`，即 W = R_{p_f}^{-1} g R_{p_i}（把 g: p_i ↦ p_f 共轭成固定参考点的群元）。

### 4.5 `gen_correlator`：从 Hadron 到关联函数矩阵

1. 默认 `time_slice_list = range(len(hadrons))`（`lattice/hadron.py:153-154`）；默认 `dagger_list` 全 False、但长度 ≥2 时把最后一个设 True（`lattice/hadron.py:156-159`）——即最后一段视为 sink（dagger 侧）。
2. 先对每段做 `set_time_in_list` + `set_dagger_in_list`（`lattice/hadron.py:160-162`）。
3. 结果矩阵 `np.ndarray(shape=tuple(len(hadrons[i])...), dtype=object)`（`lattice/hadron.py:163-165`）。
4. 对每个下标组合（`lattice/hadron.py:168-214`）：
   - `position_wavefnc = convert_pow_to_mul(Mul(各 hadron.irrep_row).expand())`（`lattice/hadron.py:192-195`）——空间/群结构部分；
   - `flavor_wavefnc = convert_pow_to_mul(Mul(各 hadron.flavor_structure).expand())`（`lattice/hadron.py:196-199`）；
   - 对 position_wavefnc 的每个 `Add` 项：把因子拆成 `HadronIrrepRow`（插入点列表，源码变量名 `insersion_list`，仓库原样拼写）与数值系数（`lattice/hadron.py:203-210`）；
   - 以 `str(flavor_wavefnc.expand())` 为 key 缓存 `diagram_simplify(quark_contract(flavor_wavefnc, np.arange(len(insersion_list)), degenerate=True))`（`lattice/hadron.py:211-221`）——味收缩结果只依赖味表达式，故可跨空间项复用；
   - `diagram_vertice_replace(term_of_result, {i: v ...})` 把 Diagram 顶点替换为具体的 `HadronIrrepRow`（`lattice/hadron.py:223-224`），按系数累加（`lattice/hadron.py:225`）；
   - 最后 `result_matrix[indices] = sp.simplify(result)`（`lattice/hadron.py:214`）。

其中 `quark_contract(expr, particles, degenerate=True)`（`lattice/quark_diagram.py:3226`）按味结构做收缩并返回 diagram 结构；`diagram_vertice_replace(expr, indice_map)`（`lattice/quark_diagram.py:2455`）对 expr 中每个 `Diagram` 用 `indice_map` 重排顶点。

### 4.6 `genLittleGroupIrrep` 与约化

`genLittleGroupIrrep` 按 `sum(p²)` 分派（`lattice/symmetry/gen_hardcoded_rep.py:186-206`）：0→Oh（`genIrrepOhD`）；1→Dic4（参考 [0,0,1]）；2→Dic2（[0,1,1]）；3→Dic3（[1,1,1]）；5→C4_generator2（[0,1,2]，**强制 `is_hardcoded=False`**，`lattice/symmetry/gen_hardcoded_rep.py:202-204`）；6→C4_generator1（[2,1,1]，同上，`lattice/symmetry/gen_hardcoded_rep.py:205-206`）；否则 `raise NotImplementedError`（`lattice/symmetry/gen_hardcoded_rep.py:207-208`）。

非 hardcode 路径（`lattice/symmetry/gen_hardcoded_rep.py:209-234`）：`littleGroup(p_ref, elem=False)` 得键集，把生成元矩阵填入，然后循环"用已知矩阵元素 × 生成元，经 `OhD_mul` 算出键名，填空缺槽"直至填满（`lattice/symmetry/gen_hardcoded_rep.py:224-233`）——按乘法表做闭包填充。硬编码路径直接取 `Dic*_irreps[irrep_name]`（`lattice/symmetry/gen_hardcoded_rep.py:211`）。最后（`lattice/symmetry/gen_hardcoded_rep.py:235-247`）：`p_ref_irrep=True` 返回参考动量的小群 irrep 本身；否则对实际动量 p 的小群每个群元 key 做 `little_group_p[key] = little_group_pref[wignerRotate(p, key)]`——用 Wigner 共轭把参考系 irrep 搬到实际动量。

`reductionToLittleGroup`（`lattice/symmetry/gen_hardcoded_rep.py:388-487`）把 OhD irrep 限制（branch）到动量小群再分解为小群 irrep：

- 由 `refRotateDict` 找到动量所属参考类与 `ref_rotate_p`（`lattice/symmetry/gen_hardcoded_rep.py:393-401`）；
- 硬编码路径：读 `little_group_reduction_map[pref_str][f"{irrep}{'g' if parity==1 else 'u'}"][little_group_irrep_name]`（`lattice/symmetry/gen_hardcoded_rep.py:406-414`），查不到则 print 并返回 None；
- 在线路径（`lattice/symmetry/gen_hardcoded_rep.py:415-455`）：构造共线判据矩阵 M_ij = Σ_{g∈H} D_OhD(g)_{ji} · D_H(g)_{00}（`lattice/symmetry/gen_hardcoded_rep.py:424-430`，对每个小群群元累加 OhD irrep 矩阵元乘小群 irrep 的 (0,0) 元）——这是 group-theory projector：M 的每行是按小群 irrep 行 0 变换的 OhD 空间向量；对 M 做 `rref()` + `GramSchmidt(vlist, True)` 得到一组正交归一子空间基（`lattice/symmetry/gen_hardcoded_rep.py:436-439`）；对每行再用 `irrep_row_connection_dict[pref_str][name][j]`（每行 j 的 (系数, 群元) 连接）合成第 j 行 subduction 矩阵（`lattice/symmetry/gen_hardcoded_rep.py:441-454`）；
- 收尾（`lattice/symmetry/gen_hardcoded_rep.py:456-462`）：每个 subduction 行左乘 `Matrix(OhD irrep[ref_rotate_p])` 并转置——把参考动量的行组合旋转到实际动量。

### 4.7 投影流程（`expr_little_group_projection`）

1. 从表达式第一项的因子中收集所有 `HadronIrrepRow` 的动量求和得总动量（`lattice/group_projection.py:96-101`），`momentunSymplify` 后在 `refRotateDict` 中找到参考类 `ref_p`（`lattice/group_projection.py:102-107`）；
2. 用 `operator_transform(expr, OhD_inv(refRotateDict[ref_p_str][p_str]))` 把表达式转到参考动量（`lattice/group_projection.py:108`）；
3. 取 `matrix_group = genLittleGroupIrrep(ref_p, irrep_name, parity)`，做行 0 投影
   P_{Γ,0}|expr⟩ = (d_Γ/|G|) · Σ_{g∈G} D_Γ(g)_{00} · U(g)|expr⟩（`lattice/group_projection.py:113-124`），
   其中 d_Γ = `matrix_group["iden"].shape[0]`，|G| = 群元个数，U(g) = `operator_transform(expr_tmp, key)`；
4. 行 0 → 目标行：对 `irrep_row_connection_dict[ref_p_str][irrep_name][row_idx]` 里的 `(coeff, rotation)` 求和 Σ c·U(rotation)（`lattice/group_projection.py:128-133`），最后 `operator_transform(result, refRotateDict[...][p_str])` 转回原动量（`lattice/group_projection.py:134-135`）。

### 4.8 `gen_connection`：行连接表的生成

以小群 irrep 每个群元矩阵的**第 0 列**为候选基（`lattice/symmetry/gen_hardcoded_rep.py:283-288`）——第 0 列即 D(g)e_0，是 irrep 行 0 在群作用下的轨道；逐个加入线性无关向量（用 `hstack(...).rank()` 判秩，`lattice/symmetry/gen_hardcoded_rep.py:316-330`）；计算 Gram 度量矩阵 G_ij = ⟨v_i,v_j⟩ 及其逆（失败时 `pinv`，`lattice/symmetry/gen_hardcoded_rep.py:332-352`）；投影系数 c = G^{-1}b，b_i = ⟨v_i, e_j⟩，对每个标准基向量 e_j 求出其在基上的展开（`lattice/symmetry/gen_hardcoded_rep.py:354-376`）；输出 `connection_results[j] = [(simplified_coeff, basis_label), ...]`，并用重构误差 `error.evalf() > 1e-10` 校验、失败 `raise ValueError`（`lattice/symmetry/gen_hardcoded_rep.py:380-385`）。该函数含大量 `print`，是为"生成 hardcoded_rep 中的 connection 表"服务的离线工具。

---

## ⑤ 不变量与校验

### 5.1 数值容差与失败路径

| 校验点 | 阈值/行为 | 出处 |
|---|---|---|
| `littleGroup` 固定点判定 | `(g @ p - p).norm() < 0.01` | `lattice/symmetry/gen_hardcoded_rep.py:107` |
| `multiplicationTable` 群元匹配 | `norm < 0.1`（很松的数值判据）；找不到匹配则 `print` 并 `exit()` 而非 raise——仅适合离线生成脚本，不适合库内运行时使用 | `lattice/symmetry/utils.py:71`、`lattice/symmetry/utils.py:76` |
| `gen_connection` 重构校验 | 误差 `error.evalf() > 1e-10` 时 `raise ValueError` | `lattice/symmetry/gen_hardcoded_rep.py:380-385` |

### 5.2 动量分派不变量

- p² 不在 {0,1,2,3,5,6} 时 `genLittleGroupIrrep` 直接 `raise NotImplementedError`（`lattice/symmetry/gen_hardcoded_rep.py:207-208`）。
- p²=5/6 不走 hardcoded 表、强制在线生成（`lattice/symmetry/gen_hardcoded_rep.py:202-206`）。
- `wignerRotate` 查不到参考类则 `raise NotImplementedError`（`lattice/symmetry/gen_hardcoded_rep.py:178`）。
- `momentunSymplify` 只接受整数分量（`int(i)`，`lattice/symmetry/gen_hardcoded_rep.py:117`）。
- `little_group_reduction_map` 只覆盖 Dic4/Dic2/Dic3 三类（`lattice/symmetry/hardcoded_rep.py:12265-12269`），p²∈{5,6} 的动量无法走硬编码的 `reductionToLittleGroup` 查表路径（`lattice/insertion/__init__.py:288` 调用处，查表失败返回 None）。

### 5.3 宇称与表示约定

- 宇称的实现约定：g/u irrep 中 `inv*` 元素矩阵 = ±（宇称因子）× 纯转动矩阵（`genIrrepOhD` 的 `lattice/symmetry/gen_hardcoded_rep.py:77-81`；hardcoded_rep 的 g/u 条目同理）。
- `T_1` 取 parity=-1 作为"极向量+宇称"的完整三维群表示（`littleGroup` 默认组、`HadronIrrepRow._rotate`、`hadron_little_group_projection` 的 rotation 均用它，`lattice/spatial_structure.py:217`、`lattice/group_projection.py:202`）。

### 5.4 表达式与符号约定

- 表达式投影仅检查表达式第一项来提取动量（`Add.make_args(expr); factors = Mul.make_args(terms[0])`，`lattice/group_projection.py:97-100`）——调用方须保证同项动量一致（多强子表达式每项动量组相同）。
- `HadronFlavorStructure` 只支持 2/3 夸克味串与 `bar{...}`；其它长度字符串在 `__init__` 会因不匹配任何分支而使 `_baryon_num` 等属性未定义（`lattice/flavor_structure.py:92-103`；`__new__` 对未知长度同样无分支，`lattice/flavor_structure.py:84-90`）。
- `HadronFlavorStructure` 的 `"ud"` 语义是 ūd（首字符为反夸克）。
- 所有 `Hadron` 变换 API（`set_time`/`conjugate`/`copy`）保持原对象不变（返回新 Hadron，`lattice/hadron.py:39-70`）；`gen_correlator` 的缓存 key 是味表达式字符串（`lattice/hadron.py:212`），依赖"味收缩与空间项无关"这一隐含约定。
- `HadronIrrepRow`/`HadronIrrep` 均为非交换 Symbol；符号名编码全部量子数；`HadronIrrepRow` 构造时急切构建 OhD T_1 表示与小群 irrep（每个 Row 符号一次 irrep 生成/查表，`lattice/spatial_structure.py:217-218`）。

### 5.5 仓库原样拼写（勿"顺手修正"）

`Qurak`（`lattice/flavor_structure.py:12`）、`falvor2irrep`/`falvor3irrep`（`lattice/symmetry/group_generator.py:294-315`）、`insersion_list`（`lattice/hadron.py` 内局部变量）、`momentunSymplify`（`lattice/symmetry/gen_hardcoded_rep.py:116`）。文档与代码引用需保持一致。

### 5.6 未使用 import（事实记录）

`lattice/symmetry/gen_hardcoded_rep.py:18`（`opt_einsum.contract`，全文无调用）、`lattice/hadron.py:12-13`（`reductionToLittleGroup, wignerRotate`）、`lattice/spatial_structure.py:12`（`reductionToLittleGroup`）、`lattice/flavor_structure.py:6`（`convert_pow_to_mul`）。

---

## ⑥ 相关测试

| 被测对象 | 测试位置 | 测的行为 |
|---|---|---|
| `OhD_mul`/`OhD_inv` 群公理 | `test/test_symmetry.py:22-73` | 单位元、逆元、结合律、具体乘积（c4x*c4x=c2x 等） |
| `Fermion_rep` | `test/test_symmetry.py:76-107` | 与 `Fermion_generator` 生成元一致；全群表示同态（`Fermion_rep[a]@Fermion_rep[b]==Fermion_rep[OhD_mul(a,b)]`） |
| `OD_irreps` | `test/test_symmetry.py:110-152` | 每个 irrep 与生成元一致；全群乘法同态 |
| `little_group_irreps` | `test/test_symmetry.py:155-206,228-340` | 固定点性质；refRotateDict 旋转正确；与各 little group 生成元一致；irrep 乘法同态 |
| `irrep_row_connection_dict` | `test/test_symmetry.py:342-360` | Σ c·D(g)·e_0 重建出第 i 行标准基 |
| `little_group_reduction_map` | `test/test_symmetry.py:362-404` | S†·D_OhD(g)·S = D_LG(g) 对每个小群群元成立（跳过 H/G 开头的 OhD irrep，`test/test_symmetry.py:366-367`） |
| `littleGroup` | `test/test_symmetry.py:304-411` | 原点/各动量固定点、包含单位元、封闭性、原点小群最大 |
| `operator_transform` | `test/test_group_projection.py:29-40` | 表达式经群元变换保持结构 |
| `hadron_little_group_projection` | `test/test_group_projection.py:81-109` | 两强子投影后宇称投影像为 0、旋转不变；共轭后的行组满足小群 irrep 行变换 |
| `expr_little_group_projection`/`multi_exprs_*` | `test/test_group_projection.py:111-124` | 返回非 None / list |
| `gen_correlator` + 投影联动 | `test/test_group_projection.py:126-166` | 用 `HadronFlavorStructure` 构造 DD̄ 同位旋标量 `I0_Cm`，`hadron_little_group_projection([D_star_p2, D_meson_p1], "E", row)` 得行组，`Hadron(row, I0_Cm)` 构造 hadron 后 `gen_correlator` 生成 2×2 关联矩阵，再验证该关联函数在小群 `genLittleGroupIrrep([0,0,1],"E")` 的每个群元作用下按 E irrep 行变换（含 `operator_transform(..., time=1)` 的时间参数） |
| `HadronIrrep`/`HadronIrrepRow` | `test/test_spatial_structure.py` 全文 | `test_creation`（属性回读、T_1→lenth=3，`test/test_spatial_structure.py:20-38`）；`test_equality/test_copy`（`test/test_spatial_structure.py:40-55`）；Row 的 `test_creation`（`test/test_spatial_structure.py:63-78`）、`test_equality`（row_idx 参与相等，`test/test_spatial_structure.py:80-95`）、`test_transform`（p=[0,0,0] 时 `transform("c4x^-1")` 等于自身，`test/test_spatial_structure.py:97-107`）、`test_conjugate`（dagger 翻转，`test/test_spatial_structure.py:109-113`） |
| `Tag`/`Qurak`/`Propagator`/`HadronFlavorStructure`/`quark_contract` | `test/test_flavor_structure.py` 全文 | Tag 不可变（NamedTuple 赋值抛 AttributeError，`test/test_flavor_structure.py:23-31`）；Qurak/Propagator 的属性与字符串形式（`test/test_flavor_structure.py:34-77`）；`HadronFlavorStructure` 的介子/重子/反重子解析与 conjugate（`test/test_flavor_structure.py:80-135`）；`quark_contract`/`_quark_contract` 的收缩行为（介子-介子、重子-反重子、简并 `degenerate=True` 时 `result.diagram.adjacency_matrix == [[1,0],[0,1]]`，`test/test_flavor_structure.py:139-190`；空 symbol 列表返回 `[S(1)]`，`test/test_flavor_structure.py:204-208`） |
| `Insertion.little_group_projection`（消费 `reductionToLittleGroup`） | `test/test_group_projection.py:170-191`、`test/test_insertion_simple.py:142` | 插入算符按动量投影到小群 irrep（后者是源码存在性断言） |
| diagram 层与投影联动 | `test/test_diagram.py:372-490` | D*D-D̄D̄ 流程，验证 diagram 层与投影联动 |

说明：

- `lattice/symmetry/two_particle.py` 在 test/ 下无对应测试（test/ 目录下无 `test_two_particle.py`），其行为仅由 `example/gen_two_particle_operators.py` 演示。
- 最小冒烟验证命令：`python -c "import lattice.symmetry as s; print(len(s.group_element), s.OhD_mul('c4x','c4x'), s.OhD_mul('c4x','c4y'))"`，预期输出 `96 c2x c3delta`。
