# lattice/insertion/ —— 插入算符（insertion）层

> 定位：`lattice/insertion/` 把 distillation 框架中"插值算符由哪些 gamma matrix、空间导数/gauge link 与动量组合而成、投影到哪个立方群 irrep 的哪一行"这件事，封装为一套稀疏符号层，并把 `Current`/`CurrentTerm` 等 current 插入算符与其 raw elemental 契约校验也收拢在同一包内；数值生成与收缩由 `lattice/generator/`、`lattice/correlator/`、`lattice/data.py` 消费本包输出。

---

## ① 模块概览（含依赖关系）

### 文件与职责

| 文件 | 行数 | 职责 |
|---|---|---|
| `lattice/insertion/gamma.py` | 349 | 16 个 4×4 Dirac gamma matrix 的构造（bit 编码整数 → 矩阵）、标签输出、8 个介子型 gamma 通道命名与量子数表、立方群作用于 gamma 指标的变换表 |
| `lattice/insertion/derivative.py` | 132 | 导数组合（∇、𝔹、𝔻、𝔼）的稀疏行编码、导数索引 ↔ 导数链（dx/dy/dz 串）双向解码、量子数查表 |
| `lattice/insertion/gauge_link.py` | 586 | gauge link path 的符号代数 `GaugeLink`（str/list/int-idx 三表示、群变换、共轭、求逆）、conserved current 的 8 方向 one-link 基 `DirectedCurrentBasis`、irrep 投影字典生成器 `gen_insertion_dict`、通道名解析 |
| `lattice/insertion/mom_dict.py` | 203 | 动量索引 → 动量字符串（`"npx npy npz"`）的静态字典与转列表函数 |
| `lattice/insertion/phase.py` | 53 | 离散动量的格点相位因子 `exp(2πi·n·x/L)` 预计算与缓存（全格点 + even/odd cb2 布局） |
| `lattice/insertion/current.py` | 1220 | current/density 插入算符体系：`CurrentTerm` 项模型、V2V/V2P/P2V/P2P 与 directed V2V raw 数据契约、端点解析（含 temporal point-split）、自旋桥接与组装器、格点散度与 WT/PCAC 校验、五个具体算符类 |
| `lattice/insertion/__init__.py` | 567 | 组装层：`Insertion`（gamma × derivative 按 O_h irrep 的 CG 组合）→ `InsertionRow` → `InsertionRowMom`（固定动量）→ `Operator`/`OperatorDisplacement`（展平为 elemental 索引）；re-export `current.py` 公开 API |

### 包内依赖

| 模块 | 依赖的包内符号 | 出处 |
|---|---|---|
| `__init__.py` | `.gamma`（scheme/group/parity/charge_conjugation/hermiticity/gamma_transform/GammaName）、`.derivative`（同构查表 + DerivativeName）、`.gauge_link`（gauge_transform_dict, GaugeLink, gauge_group, gauge_parity, gauge_charge_conjugate, gauge_hermiticity）、`.current`（re-export） | `lattice/insertion/__init__.py:8-26`、`lattice/insertion/__init__.py:563-566` |
| `current.py` | `.gauge_link.DirectedCurrentBasis`、`.gamma.gamma`（惰性 import） | `lattice/insertion/current.py:22-24`、`lattice/insertion/current.py:625`、`lattice/insertion/current.py:345` |
| `gamma.py` | `..backend.get_backend`、`lattice.symmetry.*` | `lattice/insertion/gamma.py:3-5` |
| `gauge_link.py` | `lattice.symmetry.gen_hardcoded_rep`、`lattice.symmetry.hardcoded_rep`、`lattice.symmetry.sympy_utils.find_linear_independent_exprs` | `lattice/insertion/gauge_link.py:7-10` |
| `phase.py` | `..backend.get_backend` | `lattice/insertion/phase.py:3` |
| `mom_dict.py`、`derivative.py` | 无包内依赖（纯查表） | 全文件 |

### 外部依赖与跨模块消费

| 方向 | 明细 |
|---|---|
| 外部库 | `sympy`（`GaugeLink` 继承 `sympy.Symbol`；CG 系数 `sqrt`）、`numpy`（current.py、gauge_link.py） |
| 被 correlator 层使用 | `lattice/correlator/one_particle.py:5-6`、`lattice/correlator/two_particles.py:4`、`lattice/correlator/dispersion_relation.py:3`（`Operator`/`OperatorDisplacement`/`InsertionRow`/`gamma`） |
| 被 data 层使用 | `lattice/data.py:3,9,20-33`：`get_elemental_data` 按 `Operator.parts` 逐项取 elemental |
| 被 propagator 层使用 | `lattice/propagators.py:262,431,463-464,824`（`gamma`、`GaugeLink`，含位移反转映射 `conjugate()`） |
| 被 generator 层使用 | `lattice/generator/elemental.py:61-62,70-71,728,1300-1301`（`derivative` 解码、`GaugeLink`、`build_current_raw_contract`）、`displacement_elemental.py:8,48,95-96` 与 `density_perambulator.py:9,63,160`（`MomentumPhase.get/get_cb2`）、`generalized_perambulator.py:9-10` |
| 被 current 消费链使用 | `lattice/current_elemental.py:15-20`（持久化 directed V2V 产物与逐项收缩） |

### 对外导出

`lattice/insertion/__init__.py` 直接可用：`ProjectionName`、`Row`、`GaugeRepRow`、`InsertionRowMom`、`Operator`、`OperatorDisplacement`、`InsertionRow`、`Insertion`、`InsertionGaugeLink`，以及 re-export 的 `Current, CurrentTerm, LocalVectorCurrent, LocalAxialCurrent, ConservedVectorCurrent, PseudoScalarDensity, lattice_divergence, verify_wt, verify_pcac`（`lattice/insertion/__init__.py:563-566`）。

`__init__.py` **不**导出：`DirectedCurrentBasis`、`gen_insertion_dict`、`gen_gauge_list`、`momDict_*`/`mom_dict_to_list`、`MomentumPhase`、各子模块的 `gamma/output/derivative` 函数——需从对应子模块直接 import（`current.py` 与 `lattice/generator/elemental.py` 即如此）。

---

## ② 公开 API 参考

### 2.1 `gamma.py`

| 符号 | 签名 | 说明 |
|---|---|---|
| `gamma` | `gamma(n: int)` (`lattice/insertion/gamma.py:95`) | 返回 4×4 复矩阵。构造式为 `gamma(n) = G0^{b0} @ G1^{b1} @ G2^{b2} @ G3^{b3}`，`bi = (n>>i)&1`，清除位用 4×4 单位阵替代（`lattice/insertion/gamma.py:98-102`）。`0<=n<=15` 断言（`lattice/insertion/gamma.py:96`）。**bit 编码约定**：bit0↔标签 γ1、bit1↔γ2、bit2↔γ3、bit3↔γ4；内部 `_Constant.gamma_0..gamma_3` 依次对应标签 γ1..γ4（`output` 标签与矩阵逐项核对证实） |
| `output` | `output(n: int) -> str` (`lattice/insertion/gamma.py:75`) | 人类可读标签。映射：`0→""`、`15→"γ5"`、`7→"γ5γ4"`、`8→"γ4"`、`14/13/11→"γ1γ5"/"γ2γ5"/"γ3γ5"`、`1/2/4→"γ1"/"γ2"/"γ3"`、`9/10/12→"γ1γ4"/"γ2γ4"/"γ3γ4"`、`6/5/3→"γ1γ5γ4"/"γ2γ5γ4"/"γ3γ5γ4"`（`lattice/insertion/gamma.py:77-90`）。注意部分复合标签（如 `13→"γ2γ5"`）与矩阵字面乘积 γ1γ3γ4 相差一个符号——标签是命名约定，bit 编码才是数学定义 |
| `scheme` | `scheme(name: str) -> list[int]` (`lattice/insertion/gamma.py:173`) | 返回该 gamma 通道的 gamma 分量索引列表；`assert name in _naming_scheme` |
| `group` | `group(name: str) -> str` (`lattice/insertion/gamma.py:178`) | 返回通道的 O_h irrep 名（`A_1` 或 `T_1`，`lattice/insertion/gamma.py:130-136`） |
| `parity` / `charge_conjugation` / `hermiticity` | `(name: str) -> int` (`lattice/insertion/gamma.py:183/188/193`) | 返回 ±1（`_naming_parity`/`_naming_charge_conjugation`/`_naming_hermiticity` 中 `"+"→1`、`"-"→-1`） |
| `GammaName` | class (`lattice/insertion/gamma.py:198`) | 字符串常量：`A0=R"$a_0$"`、`B0=R"$b_0$"`、`PI=R"$\pi$"`、`PI_2=R"$\pi(2)$"`、`RHO=R"$\rho$"`、`RHO_2=R"$\rho(2)$"`、`A1=R"$a_1$"`、`B1=R"$b_1$"` |
| `genGammaTransformDict` | `genGammaTransformDict() -> dict` (`lattice/insertion/gamma.py:210`) | 程序化生成 gamma 变换表：对每个 gamma 通道取其在 `genLittleGroupIrrep([0,0,0], irrep, parity)` 矩阵中的列，`+1→目标分量 idx`、`-1→目标分量 idx+16`（`lattice/insertion/gamma.py:231-241`），`'conj'` 行按通道 C 宇称生成（`lattice/insertion/gamma.py:242-245`）。已验证：返回值与硬编码 `_gamma_transform_dict` 逐键逐项完全一致 |
| `gamma_transform` | `gamma_transform(name: str, gamma_idx: int) -> int` (`lattice/insertion/gamma.py:346`) | 查 `_gamma_transform_dict[name][gamma_idx]`。返回值 ≥16 表示该分量带 −1 相位（消费端减 16 并翻号，见 `GaugeRepRow.transform`） |

核心私有数据：`_Constant`（`lru_cache(1)` 缓存的 4×4 复矩阵 `zero/one/gamma_0/gamma_1/gamma_2/gamma_3`，`lattice/insertion/gamma.py:9-68`）；`_naming_scheme`（8 通道 → 分量：`a_0=[0]`、`π=[15]`、`π(2)=[7]`、`b_0=[8]`、`a_1=[14,13,11]`、`ρ=[1,2,4]`、`ρ(2)=[9,10,12]`、`b_1=[6,5,3]`，`lattice/insertion/gamma.py:106-115`）；`_gamma_transform_dict`（`lattice/insertion/gamma.py:246-344`，dict[群元名]→list[16]，群元名含 `iden/c4x/c2z/c3delta/...`、`inv*`、`r*`、`invr*`、`conj`）。

### 2.2 `derivative.py`

| 符号 | 签名 | 说明 |
|---|---|---|
| `derivative` | `derivative(n: int) -> tuple` (`lattice/insertion/derivative.py:23`) | 把导数索引解码为方向计数元组（先确定链长，再按 base-3 逐位取方向）。运行验证：`derivative(0)=()`、`1=(0,)`、`2=(1,)`、`3=(2,)`、`4=(0,0)`、`9=(1,2)`、`13=(0,0,0)`；方向 0/1/2 对应 dx/dy/dz。注意元组顺序与 `output` 打印顺序相反：`derivative(9)=(1,2)` 对应 `output((1,9))="dzdy"`（`lattice/insertion/derivative.py:10-19` 先取 `n%3`） |
| `output` | `output(derivative_coeff_index) -> str` (`lattice/insertion/derivative.py:1`) | 输入 `(c, n)`（系数+索引），打印如 `"dydz-dzdy"`；`c==1` 无前缀、`c==-1` 前缀 `-`、其余 `str(c)`；断言 `n` 为非负 int（`lattice/insertion/derivative.py:5`） |
| `scheme` / `group` / `parity` / `charge_conjugation` / `hermiticity` | `(name: str)` (`lattice/insertion/derivative.py:102/107/112/117/122`) | 与 gamma 同构的查表接口（带 `assert name in _naming_scheme`） |
| `DerivativeName` | class (`lattice/insertion/derivative.py:127`) | 字符串常量：`IDEN=""`、`NABLA=R"$\nabla$"`、`B=R"$\mathbb{B}$"`、`D=R"$\mathbb{D}$"`、`E=R"$\mathbb{E}$"` |

通道定义（`lattice/insertion/derivative.py:36-59`，每通道为"行列表"，每行 `[[coeff, idx], ...]`）：

| 通道 | 行 | 维度 / irrep |
|---|---|---|
| `""` | `[[1,0]]`（恒等） | 1 / `A_1` |
| `∇` | `[[1,1]]`、`[[1,2]]`、`[[1,3]]`（dx、dy、dz） | 3 / `T_1` |
| `𝔹` | `[[1,11],[-1,9]]`（dydz−dzdy）、`[[1,6],[-1,10]]`、`[[1,7],[-1,5]]`（反对称组合） | 3 / `T_1` |
| `𝔻` | `[[1,11],[1,9]]`（dydz+dzdy）等（对称组合） | 3 / `T_2` |
| `𝔼` | `[[1,4],[-1,8]]`（dxdx−dydy）、`[[-1,4],[-1,8],[2,12]]`（−dxdx−dydy+2dzdz） | 2 / `E` |

量子数表（`_naming_group/_naming_hermiticity/_naming_parity/_naming_charge_conjugation/_naming_time_reversal`，`lattice/insertion/derivative.py:61-100`）。

### 2.3 `gauge_link.py`

**`class GaugeLink(Symbol)`**（`lattice/insertion/gauge_link.py:191`）——继承 `sympy.Symbol`，可与 sympy 系数做线性组合（`gen_insertion_dict` 即如此使用）。6 个基本方向 `gauge_links = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]`（`lattice/insertion/gauge_link.py:16`），即方向编码 **0..5 = +x,+y,+z,−x,−y,−z**。

| 成员 | 签名 | 说明 |
|---|---|---|
| `_VALID_NEXT` / `_VALID_COUNT` | class 属性（`lattice/insertion/gauge_link.py:196-203`） | 每个方向的合法后继：相邻两方向差值绝对值不能为 3（不能立即折返），每个方向恰有 5 个合法后继 |
| `_is_valid_gauge_list` | `@staticmethod (gauge_list) -> bool` (`lattice/insertion/gauge_link.py:235`) | 校验上述折返禁令；长度 <2 恒真 |
| `nmax_generator` | `@staticmethod (n) -> generator` (`lattice/insertion/gauge_link.py:246`) | 变长 idx 空间边界：`1, 7, 37, 187, ...`，即长度 0 占 idx 0，长度 1 占 1..6，长度 2 占 7..36，长度 k 占 `6·5^(k-1)` 个 |
| `_idx_to_gauge_list` | `@staticmethod (idx) -> list` (`lattice/insertion/gauge_link.py:206`) | idx 解码：减去前缀和后按 base-5 取"合法后继列表中的位置"，再逐步还原方向 |
| `__new__` / `__init__` | `GaugeLink(name)`（`lattice/insertion/gauge_link.py:262/282`） | 三种输入：str（`"123"`，即 name `"U123"` 的数字部分）、list（`[1,2,3]`，name 为 `"U123"`，见 `_gauge_list_to_name` :230）、int idx。非法（相邻差 3）抛 `ValueError` |
| `gauge_list` | property (`lattice/insertion/gauge_link.py:299`) | 惰性解析 name 中的数字字符为方向列表 |
| `idx` | property (`lattice/insertion/gauge_link.py:306`) | 惰性编码：`idx = 前缀和(nmax(length-1)) + 逐位 base-5 位置编码` |
| `displacement` | property (`lattice/insertion/gauge_link.py:328`) | 净位移 3 维向量（方向 <3 加 1、≥3 减 1） |
| `transform` | `transform(element: str) -> GaugeLink` (`lattice/insertion/gauge_link.py:339`) | 按硬编码 `gauge_transform_dict`（`lattice/insertion/gauge_link.py:32-129`）逐方向置换；变换后再次校验 |
| `conjugate` | `conjugate() -> GaugeLink` (`lattice/insertion/gauge_link.py:349`) | 路径反序 + 每方向 +3 mod 6（路径倒走） |
| `inv` | `inv() -> GaugeLink` (`lattice/insertion/gauge_link.py:360`) | 每方向 +3 mod 6（只反方向、不反序） |
| `genGaugeTransformDict` | `genGaugeTransformDict() -> dict` (`lattice/insertion/gauge_link.py:13`) | 程序化生成置换表：用 `genLittleGroupIrrep([0,0,0], "T_1", -1)` 的矩阵作用于 6 个单位向量（`lattice/insertion/gauge_link.py:14-30`） |

**`class DirectedCurrentBasis`**（`lattice/insertion/gauge_link.py:132`）——conserved current 原始 elemental 的稳定 8 方向 one-link 基。

| 成员 | 说明 |
|---|---|
| `SCHEMA = "lattice.current.directed-one-link-basis/v1"`、`VERSION = 1`（:134-135） | schema 标识 |
| `DIRECTIONS`（:136-145） | 8 个元组 `(index, name, 4-vector, gauge_axis, dagger)`：0..2 为 +x/+y/+z（gauge_axis 0/1/2，dagger=False），3..5 为 −x/−y/−z（dagger=True），6/7 为 ±t（gauge_axis 3）。**带时间方向是与 6 方向 `GaugeLink` 的本质区别** |
| `metadata()` classmethod（:149） | JSON 化基描述（`axis_order = ["direction","time","z","y","x","color_row","color_column"]`），被 `current.py` 用作 raw contract 的一部分 |
| `direction(index)` classmethod（:169） | `0..7 → DIRECTIONS[index]`；非 int（含 bool）抛 `TypeError`，越界抛 `ValueError` |
| `index_for_term(direction, link)` classmethod（:178） | `(0..3, "forward"/"backward") → 0..7`：空间方向 forward=direction、backward=direction+3；时间方向（direction==3）forward→6、backward→7；其余抛 `ValueError` |

**模块级函数**：

| 符号 | 签名 | 说明 |
|---|---|---|
| `gen_insertion_dict` | `gen_insertion_dict(max_lenth, insertion_form=True) -> dict` (`lattice/insertion/gauge_link.py:365`) | 枚举所有 idx < `nmax(max_lenth)` 的 `GaugeLink`，按"群变换 + 共轭"轨道去重（:382-399），用 `OD_irreps[irrep]`（`lattice/symmetry/hardcoded_rep.py`）做群平均投影（:412-418），再对 `inv()`/`conjugate()` 做 parity（:427-437）与 charge conjugation（:438-459）投影；`find_linear_independent_exprs`（`lattice/symmetry/sympy_utils`）挑线性无关基，并按 `irrep_row_connection_dict` 展开到每 irrep 行；`insertion_form=True` 时输出 `[coeff, gauge_link_idx]` 稀疏行（:497-507，与 `derivative.py` 行格式同构）。返回 `dict[通道名, list[rows]]`，通道名形如 `"T_1u-"`（`irrep + parity + charge_conjugate`，:373-387）。仅在 `test/test_gauge_link.py` 与本模块 `__main__`（:581-586）被调用，未在包 `__init__` 导出 |
| `gen_gauge_list` | `gen_gauge_list(insertion_dict)` (`lattice/insertion/gauge_link.py:520`) | 把 `gen_insertion_dict` 输出重编为连续 gauge 列表与新字典，并附带位移分布信息 dict；返回 `(gauge_list, new_insertion_dict, information_dict)` |
| `gauge_group` | `gauge_group(name)` (`lattice/insertion/gauge_link.py:556`) | 去掉通道名末两字符，如 `"T_1u-"→"T_1"` |
| `gauge_parity` | `gauge_parity(name) -> int` (`lattice/insertion/gauge_link.py:560`) | 倒数第二字符 `u→-1`、`g→1`；断言字符合法 |
| `gauge_charge_conjugate` | `gauge_charge_conjugate(name) -> int` (`lattice/insertion/gauge_link.py:568`) | 末字符 `+→1`、`-→-1` |
| `gauge_hermiticity` | `gauge_hermiticity(name)` (`lattice/insertion/gauge_link.py:576`) | 等于 `gauge_charge_conjugate` |

### 2.4 `mom_dict.py`

| 符号 | 说明 |
|---|---|
| `momDict_mom1`（`lattice/insertion/mom_dict.py:1`） | 仅 `(0,0,0)`，共 1 项 |
| `momDict_mom3`（`lattice/insertion/mom_dict.py:5`） | 27 项，分量 ∈ {−1,0,1} |
| `momDict_mom9`（`lattice/insertion/mom_dict.py:36`） | 123 项，覆盖 |n|≤3 主要壳层；`61: "0 0 0"` |
| `momDict_test`（`lattice/insertion/mom_dict.py:163`） | 23 项的测试用序列 |
| `mom_dict_to_list(mom: int = 9)`（`lattice/insertion/mom_dict.py:187`） | `mom∈{1,3,9,0}`（0 对应 `momDict_test`），否则 `raise ValueError("Unknown mom max = ...")`；返回 `List[Tuple[int,int,int]]`（把 `"npx npy npz"` 按空格切分转 int） |

字典约定：键为动量索引（int），值为严格 `f"{npx} {npy} {npz}"` 格式的空格分隔字符串；`InsertionRow.__call__` 用 `list(self.momentum_dict.values()).index(f"{npx} {npy} {npz}")` 反查索引（`lattice/insertion/__init__.py:227-228`）——**动量不在字典中会抛 `ValueError`**。

### 2.5 `phase.py`

**`class MomentumPhase`**（`lattice/insertion/phase.py:6`）——预计算并缓存离散动量的格点相位因子。

| 成员 | 签名 | 说明 |
|---|---|---|
| `__init__` | `__init__(self, latt_size: List[int])` (`lattice/insertion/phase.py:7`) | `Lx,Ly,Lz,Lt = latt_size`；构造 `self.x/y/z`，形状 `(Lz, Ly, Lx)`（如 `x = arange(Lx).reshape(1,1,Lx).repeat(Lz,0).repeat(Ly,1) * 2j*pi/Lx`，`lattice/insertion/phase.py:10-15`），再扩展为 `(Lt, Lz, Ly, Lx)`（:16-18）；cb2 版本 `self.x_cb2/y_cb2/z_cb2` 形状 `(2, Lt, Lz, Ly, Lx//2)`、dtype `"<c16"`（:19-38）。另建空缓存 `self.cache` 与 `self.cache_cb2` |
| `get` | `get(self, np: Tuple[int])` (`lattice/insertion/phase.py:41`) | 返回 `exp(npx*x + npy*y + npz*z)`，形状 `(Lt, Lz, Ly, Lx)`，按 `np` 元组缓存于 `self.cache` |
| `get_cb2` | `get_cb2(self, np: Tuple[int])` (`lattice/insertion/phase.py:48`) | cb2 版本，形状 `(2, Lt, Lz, Ly, Lx//2)`，缓存于 `self.cache_cb2` |

### 2.6 `__init__.py` —— 组装层

**`class ProjectionName`**（`lattice/insertion/__init__.py:29`）：`A1="A_1"`、`A2="A_2"`、`E="E"`、`T1="T_1"`、`T2="T_2"`。

**`class Row(list)`**（`lattice/insertion/__init__.py:37`）——稀疏符号行，扁平格式 `[gamma_idx, terms, gamma_idx, terms, ...]`（偶长度），`terms = [[coeff, derivative_idx], ...]`（coeff 为 int/float/sympy 数）。

| 成员 | 签名 | 说明 |
|---|---|---|
| `simplify` | `simplify() -> Row`（:38） | 拉平为 `(gamma_idx, derivative_idx, coeff)` 三元组，按前两者合并求和，丢弃系数 0 项后重组 `self[:]` |
| `__add__` | `__add__(self, other)`（:72） | 拼接 + simplify |
| `__mul__` / `__rmul__` | `__mul__(self, scalar)`（:78/89） | 标量乘每个 `terms[i][0]`；只接受 `int/float/sympy.Expr`，否则返回 `NotImplemented` |
| `__neg__` / `__sub__` / `__iadd__` / `__isub__` / `__imul__` | （:93-115） | 组合上述运算；使 `Insertion.construct` 能以 `row += matrix_elem * self.rows[j]` 做线性组合 |

**`class GaugeRepRow(Row)`**（`lattice/insertion/__init__.py:117`）：

| 成员 | 说明 |
|---|---|
| `transform(group_element)`（:121） | 逐 gamma 分量用 `gamma_transform` 变换，结果 ≥16 则系数翻号并减 16（:127-130）；逐 gauge-link idx 用 `GaugeLink(idx).transform(...)` 变换（:131-133）；断言行偶长度。**仓库内除定义外无引用（grep 确认）** |

**`class InsertionRowMom`**（`lattice/insertion/__init__.py:141`）：

| 成员 | 说明 |
|---|---|
| `__init__(self, row, momentum: int, profile=None)` | `row` 为稀疏行；`momentum` 为动量索引；`profile` 为随行透传的元数据（进入 `Operator.parts` 第 4 位，最终进入 `lattice/data.py` 的 elemental 取数循环，仅作 provenance，不被解码） |

**`class Operator`**（`lattice/insertion/__init__.py:147`）：

| 成员 | 签名 | 说明 |
|---|---|---|
| `__init__` | `__init__(self, name: str, insertion_rows: List[InsertionRowMom], coefficients: List[float])`（:148） | 断言行数==系数数（:152-155）；展平为 `self.parts = [gamma_idx, [[coeff*derivative_coeff, derivative_idx, momentum, profile], ...], gamma_idx, ...]`（:157-176）。**特例**：当前 gamma idx 为 5 或 13 时 `derivative_coeff *= -1`（:166-169）；已数值验证该约定自洽：`gamma(4)@gamma(1) == -gamma(5)`（即 γ3·γ1 = −gamma(5)）、`gamma(4)@gamma(1)@gamma(8) == -gamma(13)`（即 γ3·γ1·γ4 = −gamma(13)），与行内注释 `gamma_3gamma_1 = -gamma(5), gamma_3gamma_1gamma_4 = -gamma(13)`（:167）一致——即生成端采用 γ3 在前的乘积次序，而 `gamma(5)`/`gamma(13)` 按 bit 编码存的是 γ1 在前的矩阵 |
| `__str__` | （:178） | 打印每个 gamma idx 及其 `(coeff, derivative_idx, momentum)` 项 |
| `set_gamma` | `set_gamma(self, i_row, gamma_idx)`（:189） | 改写 `self.parts[2*i_row]` |
| `set_derivative` | `set_derivative(self, i_row, i_term, deriv_idx)`（:195） | 改写 `self.parts[2*i_row+1][i_term][1]` |

**`class OperatorDisplacement(Operator)`**（`lattice/insertion/__init__.py:202`）：

| 成员 | 说明 |
|---|---|
| `__init__(self, name, insertion_rows, coefficients, distances: List[int])` | 断言行数==距离数（:206）；断言每项 `derivative_idx == 0`（位移算符不能同时带导数，:211-215）；随后 `set_derivative(i_row, i_term, distances[irow])`（:216-218）——位移 elemental 的第一轴就是 `GaugeLink.idx` 空间（与 `lattice/generator/displacement_elemental.py` 生成端、`lattice/propagators.py:431-446` 的 `GaugeLink(disp_idx).displacement` 用法一致） |

**`class InsertionRow`**（`lattice/insertion/__init__.py:221`）：

| 成员 | 签名 | 说明 |
|---|---|---|
| `__init__` | `__init__(self, row, momentum_dict: Dict[int,str], profile=None)`（:222） | 绑定稀疏行与动量字典 |
| `__call__` | `__call__(self, npx, npy, npz) -> InsertionRowMom`（:227） | 用 `momentum_dict.values().index(f"{npx} {npy} {npz}")` 反查动量索引 |
| `__str__` | （:230） | 用 `gamma.output`/`derivative.output` 打印如 `γ1 * (dx) + γ2 * (dydz-dzdy)` |

**`class Insertion`**（`lattice/insertion/__init__.py:249`）：

| 成员 | 签名 | 说明 |
|---|---|---|
| `__init__` | `__init__(self, gamma: GammaName, derivative: DerivativeName, projection: ProjectionName, momentum_dict: Dict[int,str], profile=None)`（:250） | `self.gamma = gamma_scheme(gamma)`（分量索引列表）、`self.derivative = derivative_scheme(derivative)`（稀疏行列表）；`self.parity/charge_conjugation/hermiticity = gamma_* × derivative_*`（:256-259）；`self.projection = [gamma_gourp(gamma), derivative_gourp(derivative), projection]`（左=gamma irrep，右=derivative irrep，目标=projection，:260）；随后调用 `construct()` |
| `__getitem__` | `__getitem__(self, idx) -> InsertionRow`（:270） | 把第 idx 行包装为 `InsertionRow` |
| `__str__` | （:273） | 逐行打印 |
| `multiply` | `multiply(self, coeff, derivative)`（:279） | 把一行 terms 的系数统一乘 `coeff`，返回新 terms |
| `little_group_projection` | `little_group_projection(self, momentum, irrep_name, idx=0)`（:285） | `momentum==[0,0,0]` 原样返回；否则取 `reductionToLittleGroup(momentum, self.projection[-1], self.parity, irrep_name)[idx]`（`lattice/symmetry/gen_hardcoded_rep.py`）作为重组合矩阵，把静止系 rows 线性组合成 little group irrep 的行，存入 `self.little_group_irreps_dict[str(momentum)]` 并替换 `self.rows`（:285-297）。即先在静止系按 O_h 投影，再按参考动量的 little group 二次投影 |
| `construct` | `construct(self)`（:300） | 按 `left × right → projection` 的 CG 表生成 `length[projection]` 行（`length = {"A_1":1,"A_2":1,"E":2,"T_1":3,"T_2":3}`，:302）。详见 ④ |

**`class InsertionGaugeLink(Insertion)`**（`lattice/insertion/__init__.py:419`）：与 `Insertion` 同构，但 "derivative" 槽位换成 gauge-link 基行：`self.derivative = insertion_dict[gauge_link_irrep_name][gauge_link_idx]`（:427，`insertion_dict` 即 `gauge_link.gen_insertion_dict` 输出）；量子数改为 gamma×gauge-link 通道乘积（:429-432）；`construct()`（:444-561）与父类逐行相同（复制粘贴）。**仓库内无外部引用（grep 确认）**。

**re-export**（`lattice/insertion/__init__.py:563-566`）：`Current, CurrentTerm, LocalVectorCurrent, LocalAxialCurrent, ConservedVectorCurrent, PseudoScalarDensity, lattice_divergence, verify_wt, verify_pcac`。

### 2.7 `current.py` —— current/density 体系

模块常量（`lattice/insertion/current.py:12-20`）：`CURRENT_API_VERSION="1.2.0"`；`CURRENT_ELEMENTAL_SCHEMA="lattice.current.raw-spatial-displacement-basis/v1"`（legacy spatial 四通道）；`CURRENT_DIRECTED_RAW_SCHEMA="lattice.current.raw-directed-one-link-basis/v1"`（directed V2V-only）；`CURRENT_TERM_SCHEMA="lattice.current.term/v1"`；`CURRENT_ASSEMBLER_SCHEMA="lattice.current.assembler/v1"`；`_VECTOR_GAMMA=(1,2,4,8)`（γ1..γ4）、`_AXIAL_GAMMA=(14,13,11,7)`（γ1γ5,γ2γ5,γ3γ5,γ5γ4 槽位，按 `gamma.output` 标签）。

#### `CurrentTerm`（frozen dataclass，`lattice/insertion/current.py:394`）

| 字段 | 类型/默认 | 含义 |
|---|---|---|
| `coefficient` | `complex`（必填） | 项系数（组装器唯一加权来源之一） |
| `direction` | `int`（必填） | −1..3；−1=无方向（`PseudoScalarDensity` 用），0..2=空间，3=时间 |
| `displacement` | `tuple[int,int,int,int]`（必填） | 4 维位移 |
| `gamma_index` | `int`（必填） | 0..15，`gamma(n)` 的 bit 编码 |
| `link` | `str = "none"` | `none|forward|backward`（Wilson link 取向） |
| `wilson_r` | `Optional[float] = None` | Wilson 参数；带 link 时必须提供 |
| `spin_structure` | `str = "gamma"` | 非空字符串标签（如 `"r-gamma_3"`、`"gamma_3gamma_5"`） |
| `normalization` | `complex = 1` | 项归一化因子 |
| `bar_offset` / `field_offset` / `link_origin_offset` | `tuple = (0,0,0,0)` | 三个端点的 4 维 offset |
| `link_dagger` | `bool = False` | link 是否取 dagger |
| `boundary_policy` | `str = "caller-supplied"` | 当前只允许该值 |
| `temporal_point_split` | `bool = False` | 是否为时间向 point-split 项 |

`__post_init__`（`lattice/insertion/current.py:410-453`）校验：coefficient/normalization/wilson_r 为有限数值标量（bool 被拒，:359-372）；`direction∈[-1,3]`、`gamma_index∈[0,15]`、三个 offset 均为 4 元组整数（:374-391）；`link∈{none,forward,backward}`；`spin_structure` 非空字符串；`boundary_policy=="caller-supplied"`；**含时间分量的 offset（offset[3]≠0）必须 `direction==3`**；`temporal_point_split` 必须与 "direction==3 且有 temporal endpoint" 一致。`as_dict()`（:455）输出 v1 schema dict。`_coerce_term(term)`（:478）接受 `CurrentTerm` 或带正确 schema 的 Mapping（未知/缺字段报错）。

#### raw 数据契约与校验

| 符号 | 签名 | 说明 |
|---|---|---|
| `build_current_raw_contract` | `(raw, *, boundary, available_ne, used_ne, momentum_count) -> dict`（`lattice/insertion/current.py:112`） | 为"v2v-only、8 个 one-link 方向"的 directed 原始数据生成描述 dict。强制 `raw` 只含键 `"v2v"`、`boundary∈{periodic,open}`、v2v 形状 `(8, Lt, momentum, used_ne, used_ne)`、dtype complex（:118-130）；`cache_identity` 为去掉自身后的语义字段 sha256 指纹（:30-35） |
| `_validate_current_raw_contract_metadata` | `(contract) -> dict`（:52） | 键集合精确等于 `_current_raw_contract_keys()`（:37-45）；basis 必须等于 `DirectedCurrentBasis.metadata()`；`channels==["v2v-one-link"]`；`axes["v2v"]==["direction","time","momentum","sink_ne","source_ne"]`；`shape[0]==8`；Ne 元数据对称（source==sink==used，`raw_generator_used_ne_is_symmetric is True`）；`shape[3:]==[used,used]`；指纹校验（:105-108） |
| `validate_legacy_spatial_current_raw` | `(raw, contract) -> dict`（:200） | **V2V/V2P/P2V/P2P 变体的形状权威定义**（见 ③）；返回带 `legacy_spatial_only=True` 的确认 dict |
| `validate_current_raw_contract` | `(raw, contract, *, require_temporal=False) -> dict`（:274） | 按 `contract["schema"]` 分派；legacy spatial 契约在 `require_temporal=True` 时直接报错（legacy 数据永远不含时间方向 link，:278-280）；directed 契约用 `build_current_raw_contract` 重建期望元数据并要求逐字段相等（:283-294），再检查数值有限（:295-297） |
| `current_raw_cache_key` | `(key, contract) -> str`（:300） | `f"{key}:{cache_identity}"`，缓存层防串味 |

#### 端点解析与自旋桥接

| 符号 | 签名 | 说明 |
|---|---|---|
| `resolve_current_term_endpoints` | `(term, *, anchor_time, temporal_extent=None, boundary) -> dict`（:594） | `bar_time/field_time/link_origin_time = anchor_time + 对应 offset[3]`；`periodic` 对 `temporal_extent` 取模；`open` 越界抛 `IndexError`；`unbounded` 原样返回；返回值附 `temporal_point_split` 与 `boundary`。`_boundary_inputs`（:582）校验 anchor/extent |
| `resolve_directed_current_raw` | `(raw, contract, term, *, endpoints, source_ne, sink_ne, momentum=0) -> dict`（:321） | term 必须带 link（`"none"` 报错 :332）；`direction = DirectedCurrentBasis.index_for_term(term.direction, term.link)`（:345-346）；返回切片 `raw["v2v"][direction, bar_time, momentum, :sink_ne, :source_ne]` + provenance（`raw_anchor="bar_endpoint"`）。**只解决取向/锚点，不加权** |
| `_spin_matrix_for_term` | `(term) -> ndarray`（:624） | 取 `gamma(gamma_index)`；`link=="none"` 直接返回 Γ；否则必须有 `wilson_r`：forward → `r·I − Γ`，backward → `r·I + Γ`（:640-645）——Wilson 型 conserved current 的中点结构 |
| `resolve_current_term_spin` | `(term, raw_resolver, *, endpoints, source_ne, sink_ne) -> dict`（:670） | 调 `raw_resolver`（回调契约：返回 `{"value","source_ne","sink_ne"}`，value 形状 `(sink_ne, source_ne)`，见 `_validate_spin_raw_result` :646），然后 `value = spin_matrix[:,:,None,None] * raw[None,None,:,:]`，得 `(4,4,sink_ne,source_ne)`；**只施加一个显式 spin matrix，绝不施加 coefficient/normalization**（:697-701） |
| `make_spin_aware_current_resolver` | `(raw_resolver) -> callable`（:705） | 把 spinless raw 回调适配为 `assemble_current_terms` 的 resolver |
| `assemble_spin_aware_current` | `(current_or_terms, raw_resolver, **kwargs)`（:719） | `Current` 实例取 `.terms` 后转调 `assemble_current_terms` |
| `spin_aware_current_adapter` | `(assembled) -> dict`（:727） | 校验 vertex 形状 `(4,4,sink_ne,source_ne)` 与 Ne provenance，输出 `{"schema","api_version","vertex","axes","ne","term_count","terms"}` |
| `consume_spin_aware_current` | `(adapter, sink_spin, source_spin)`（:785） | `np.einsum("a,abij,b->ij", sink_spin, vertex, source_spin)`；spin 权重须形状 `(4,)` |
| `legacy_current_vertex_adapter` | `(vertices_by_time)`（:766） | 包装 `lattice.quark_diagram.CurrentVertexAdapter` 适配 legacy `get(t)` 协议；docstring 明确：temporal point-split 项必须走 `lattice.current_elemental.contract_directed_current_v2v` |

#### 组装器

`assemble_current_terms(terms, resolver, *, available_source_ne, available_sink_ne, used_source_ne=None, used_sink_ne=None, anchor_time=0, temporal_extent=None, boundary="unbounded")`（`lattice/insertion/current.py:803`）：

1. terms 非空、resolver 可调用（:810-819）；Ne 用 `_requested_ne` 解析（used 缺省=available，used≤available，:570-580）。
2. 逐 term：`_coerce_term` → 解析端点 → 调 resolver → 校验 `{"value","source_ne","sink_ne"}` 且 Ne 精确匹配（:830-859）。
3. 所有 term 的 value 形状一致（:860-868）。
4. **唯一加权点**：`weighted = coefficient * normalization * value`，逐项累加（:869-871）。
5. 返回 `{"schema","api_version","value","term_count","ne":{source,sink:{available,used}},"boundary","anchor_time","terms":tuple(records)}`，records 含每项 schema/端点/resolver provenance（:872-892）。

#### 散度与物理校验

| 符号 | 签名 | 说明 |
|---|---|---|
| `lattice_divergence` | `(current, *, site_axes=None, periodic=True, lattice_spacing=1.0)`（:894） | **后向格点散度**：要求 `arr.ndim>=5` 且 `arr.shape[0]==4`（axis 0 为方向；site 轴默认 `(1,2,3,4)`，须 4 个唯一整数轴）；对每个 μ：`out += arr[μ] − shift(arr[μ], +1)`（periodic 用 `np.roll`，open 用 0 填充），最后 `/ lattice_spacing`，即 $D_\mu J_\mu(x)=\sum_\mu [J_\mu(x)-J_\mu(x-\hat\mu)]/a$ |
| `verify_wt` | `(*, divergence=None, current=None, site_axes=None, periodic=True, lattice_spacing=1.0, atol=1e-10, rtol=1e-7)`（:933） | 无 divergence 时可由 current 现算；判据 `np.all(|div| <= atol)`；返回 `{"condition":"WT divergence = 0","passed","max_abs","array"}`（`_zero_result` :926） |
| `verify_pcac` | `(*, axial_divergence=None, axial_current=None, pseudoscalar=None, mass=None, improvement_residual=0, ...)`（:945） | 恰好提供 divergence/current 之一（:951-955）；必须提供 pseudoscalar 与 mass；`lhs = div A`，`rhs = 2*m*P + improvement`（标量可广播）；判据 `np.all(|lhs−rhs| <= atol + rtol*|rhs|)`，即 PCAC 关系 $\partial_\mu A_\mu = 2mP + E$ |

容差校验辅助：`_finite_real_scalar/_tolerances/_lattice_spacing`（:519-541）——atol/rtol 非负、spacing>0。

#### 算符类（`lattice/insertion/current.py:997-1207`）

| 类 | 关键类属性 | `terms` |
|---|---|---|
| `Current`（:997，基类） | `name="current"`、`elemental_key="v2v"`、`verification_checks=frozenset()`、`requires_z/requires_wt/requires_pcac=False`、`gamma_indices=_VECTOR_GAMMA` | 4 个局域项 `CurrentTerm(1, mu, (0,0,0,0), gamma_indices[mu])`（:1009-1011） |
| `LocalVectorCurrent`（:1140） | `name="local_vector"`，其余同基类 | 继承局域向量项 γ_μ |
| `LocalAxialCurrent`（:1143） | `gamma_indices=_AXIAL_GAMMA`、`verification_checks={"pcac"}`、`requires_pcac=True` | 系数 `(1,-1,1,-1)`、`spin_structure=f"gamma_{mu}gamma_5"`（:1149-1156） |
| `ConservedVectorCurrent`（:1160） | `verification_checks={"z","wt"}`、`requires_z/requires_wt=True`、`name="conserved_vector"` | 每个 μ 两项：forward `CurrentTerm(-0.5, mu, fwd, gamma_μ, "forward", wilson_r, "r-gamma_μ", field_offset=fwd, link_dagger=False, temporal_point_split=(mu==3))`；backward `CurrentTerm(+0.5, mu, back, gamma_μ, "backward", wilson_r, "r+gamma_μ", bar_offset=fwd, link_dagger=True, temporal_point_split=(mu==3))`（:1166-1190）。`_operator`（:1192）附加 `wilson_r, point_split:True, link_orientations`。**μ=3 的两项 `temporal_point_split=True`，bar/field 端点相差一个时间步——temporal point-split 的来源** |
| `PseudoScalarDensity`（:1198） | `gamma_indices=(15,)`、`verification_checks={"pcac"}`、`requires_pcac=True`、`name="pseudoscalar_density"` | 单项 `CurrentTerm(1, -1, (0,0,0,0), 15, spin_structure="gamma_5")`（direction=−1 无方向） |

`Current` 实例方法：

| 方法 | 签名 | 说明 |
|---|---|---|
| `__init__` | `__init__(self, generator=None, *, z=None, lattice_spacing=1.0, wilson_r=1.0)`（:1005） | 保存 generator、测得 Z、格距、Wilson 参数 |
| `terms` | property（:1009） | 见上表 |
| `compute_elemental` | `compute_elemental(self, generator=None, t=None, *, used_source_ne=None, used_sink_ne=None, **kwargs)`（:1017） | 适配 `generator.calc_all(t)`：轻量路径（generator 无 `Ne/usedNe` 属性，:1035-1047）直接取 `results[self.elemental_key]`；完整路径（:1049-1104）要求 `calc_all` 返回含 `v2v/v2p/p2v/p2p` 四键，按 usedNe/usedNp 校验形状并生成与 legacy spatial 契约同构的 contract dict（:1079-1100） |
| `verify_z` | `@staticmethod verify_z(z=None, *, atol=1e-10, rtol=1e-7)`（:1106） | 校验测得 Z==1（`_zero_result(z−1, "Z = 1")`） |
| `verify` | `verify(self, *, z=None, renormalization=None, divergence=None, current=None, axial_divergence=None, axial_current=None, pseudoscalar=None, mass=None, improvement_residual=0, ...)`（:1110） | 按类的 `verification_checks`/`requires_*` 标志与入参是否提供，分别调用 Z/WT/PCAC 校验 |

`__all__`（`lattice/insertion/current.py:1209-1220`）导出全部 schema 常量、contract/validate/cache-key 函数、解析与组装函数、五个算符类、`lattice_divergence/verify_wt/verify_pcac`。

---

## ③ 数据结构与数组形状约定

### 行与算符的稀疏表示

| 对象 | 表示 | 出处 |
|---|---|---|
| Derivative 通道行 | `[[coeff, derivative_idx], ...]`；`derivative_idx` 经 `derivative(n)` 三进制解码为方向链（长度 ≤k 的链共 `(3^(k+1)-1)/2` 个，与 `lattice/generator/elemental.py:70` 的 `num_derivative=(3**(num_nabla+1)-1)//2` 公式互相印证） | `lattice/insertion/derivative.py:36-59` |
| Row（稀疏符号行） | 扁平 `[gamma_idx, terms, gamma_idx, terms, ...]`（偶长度），`terms=[[coeff, derivative_idx], ...]` | `lattice/insertion/__init__.py:37` |
| Gauge-link 行 | `[coeff, gauge_link_idx]` 稀疏行（与 derivative 行同构）；`gauge_link_idx` 属于 `GaugeLink.nmax_generator` 定义的变长 idx 空间（长度 0/1/2/3 的边界为 1/7/37/187） | `lattice/insertion/gauge_link.py:497-507,246` |
| `Operator.parts` | `[gamma_idx, [[coeff, derivative_idx, momentum_idx, profile], ...], gamma_idx, ...]`（偶长度） | `lattice/insertion/__init__.py:157-176` |
| 动量 | dict[int → `"npx npy npz"`]；`InsertionRow.__call__` 反查索引 | `lattice/insertion/mom_dict.py`、`lattice/insertion/__init__.py:227-228` |

### 数组形状

| 数组 | 形状 | 出处 |
|---|---|---|
| `gamma(n)` | `(4, 4)` 复矩阵 | `lattice/insertion/gamma.py:95-103` |
| `MomentumPhase.x/y/z` | `(Lz, Ly, Lx)`，扩展后 `(Lt, Lz, Ly, Lx)` | `lattice/insertion/phase.py:10-18` |
| `MomentumPhase.x_cb2/y_cb2/z_cb2` | `(2, Lt, Lz, Ly, Lx//2)`，dtype `"<c16"`；**`cb2[0]` 恒为 `(t+z+y+x)` 为偶的格点**（`ieo=(it+iz+iy)%2` 决定偶/奇切片归属，`lattice/insertion/phase.py:26-38`） | `lattice/insertion/phase.py:19-38` |
| `MomentumPhase.get(np)` | `(Lt, Lz, Ly, Lx)` | `lattice/insertion/phase.py:41-46` |
| `MomentumPhase.get_cb2(np)` | `(2, Lt, Lz, Ly, Lx//2)` | `lattice/insertion/phase.py:48-53` |
| directed raw `v2v` | `(8, Lt, momentum, used_ne, used_ne)`，dtype complex；axes `["direction","time","momentum","sink_ne","source_ne"]` | `lattice/insertion/current.py:118-130` |
| legacy spatial `v2v` | 4 维，axes `("displacement","momentum","sink_ne","source_ne")`，末两维 `(used, used)` | `lattice/insertion/current.py:229-234,258-259` |
| legacy spatial `v2p` | 4 维，axes `("displacement","sink_ne","point","color")`，末三维 `(used, used_np, 3)` | `lattice/insertion/current.py:260-261` |
| legacy spatial `p2v` | 4 维，axes `("displacement","point","color","source_ne")`，末三维 `(used_np, 3, used)` | `lattice/insertion/current.py:262-263` |
| legacy spatial `p2p` | `"sparse-per-displacement"`：列表，每项 `{"type":"identity"}` 或 `{"type":"sparse","indices":(N,2) 整数,"values":(N,3,3) complex}`——逐位移的 3×3 色矩阵（恒等或 (索引对, 值) 稀疏存法） | `lattice/insertion/current.py:180-198` |
| spin-aware vertex | `(4, 4, sink_ne, source_ne)`（axes `("sink_spin","source_spin","sink_ne","source_ne")`） | `lattice/insertion/current.py:670-703,727-764` |
| `lattice_divergence` 输入 | `ndim>=5`，`shape[0]==4`（方向轴），site 轴默认 `(1,2,3,4)` | `lattice/insertion/current.py:894-924` |
| spin 权重 | `(4,)` | `lattice/insertion/current.py:778-783` |

Ne/Np 语义：`Ne` 为 distillation 低模空间维数（vector 通道），`Np` 为点源数（point 通道）；约束 `0<=used<=available`、requested ≤ used、raw Ne 对称（`lattice/insertion/current.py:237-256`）。前缀 V/P 即 vector/point 通道（依据 axes 与 Ne/Np 元数据）。

### elemental 索引空间的对齐

elemental 第一轴 `derivative_idx`/displacement 索引必须匹配生成端索引空间：calc_deriv 模式用 `derivative.py` 索引（`lattice/generator/elemental.py:70`），calc_disp 模式用 `GaugeLink.idx`（`lattice/generator/elemental.py:71`）；`Operator.parts` 的 `derivative_idx`（或 `OperatorDisplacement` 置入的距离 idx）直接作为该轴下标取数（`lattice/data.py:20-33`）。

---

## ④ 算法与流程

### gamma 矩阵构造与 bit 编码

`gamma(n)` 按 n 的四个二进制位选乘 `_Constant.gamma_0..gamma_3`（bit i=0 时用单位阵替代），bit i ↔ 标签 γ_{i+1}。由 `output` 标签核对：`n=15 → "γ5"`（γ1γ2γ3γ4）、`n=7 → "γ5γ4"`、`n=8 → "γ4"`、`n=14 → "γ1γ5"`。

### 导数索引编码

`derivative(n)`：先用 `while n>=0: num+=1; n-=3^num` 确定链长 `num`，再按 base-3 逐位取方向，返回方向计数元组（反序还原）。链长 ≤k 的索引总数 `(3^(k+1)-1)/2`。

### GaugeLink idx 编码（受限随机游走）

路径方向序列受"相邻差 ≠3"约束（不能立即折返）；idx = 长度前缀和（`nmax_generator`：长度 0/1/2/3 边界 1/7/37/187）+ 后续方向在"合法后继列表"（每方向 5 个）中的 base-5 位置编码。`conjugate()` = 反序 + 每方向 +3 mod 6；`inv()` = 只 +3 不反序；`transform()` 查 `gauge_transform_dict` 置换表并重新校验。

### gen_insertion_dict 的 irrep 投影流程

1. 枚举 idx < `nmax(max_lenth)` 的所有 `GaugeLink`，按"群变换轨道 + 共轭"去重分组（`lattice/insertion/gauge_link.py:379-399`）；
2. 对每组用 `OD_irreps[irrep]` 做群平均 `Σ_key conj(irrep[key][0,0]) * link.transform(key)`（:412-418）；
3. 对 `inv()` 与 `conjugate()` 分别做 parity（u/g）与 charge conjugation（+/−）投影（:427-459）；
4. `find_linear_independent_exprs` 挑线性无关基，`irrep_row_connection_dict` 展开到每 irrep 行（:461-495）；
5. `insertion_form=True` 时输出 `[coeff, gauge_link_idx]` 稀疏行（:497-507）。

### Insertion.construct 的 O_h CG 组合

按 `left(gamma irrep) × right(derivative irrep) → projection` 生成长度 `length[projection]` 的行（`lattice/insertion/__init__.py:300-417`）：

| 分解 | 行构造 |
|---|---|
| `A_1 × A_1 → A_1`（left=="A_1"，断言 right==projection） | `[gamma[0], derivative[i]]` |
| `T_1 × A_1 → T_1` | `[gamma[i], derivative[0]]` |
| `T_1 × E → T_1 / T_2` | 含 `1/√3`、`2/√3`、`√3` 系数的组合，末尾 `row.simplify()`（:315-333） |
| `T_1 × (A_1|A_2) → A_1/A_2` | 三对角求和 `[γ0,D0, γ1,D1, γ2,D2]`（:334-343） |
| `T_1 × T_1 → E` 与 `T_1 × T_2 → E` | `1/√3` 组合（:344-373） |
| `T_1 × T_1 → T_1/T_2`（及 T_2 情形） | `j=(i+1)%3, k=(i+2)%3`；`right==projection` 时反对称 `[γj,Dk, −γk,Dj]`，否则对称 `[γj,Dk, γk,Dj]`（:374-390） |

系数为硬编码 O_h CG 系数（sympy `sqrt` 表示），并用 `irrep_T1` 表（:303-309）断言分解合法性。

### little group 二次投影

`little_group_projection(momentum, irrep_name, idx=0)`：先在静止系按 O_h 投影（`construct`），再取 `reductionToLittleGroup(momentum, projection[-1], parity, irrep_name)[idx]` 作为重组合矩阵，把静止系 rows 线性组合成 little group irrep 的行并替换 `self.rows`（`lattice/insertion/__init__.py:285-297`）。

### Operator 展平与符号约定

每个 `InsertionRowMom` 的行展平为 `parts` 项：`[coefficient*derivative_coeff, derivative_idx, momentum, profile]`；gamma idx 为 5/13 时 `derivative_coeff` 翻号（`lattice/insertion/__init__.py:166-169`）。数值验证：γ3·γ1 = `gamma(4)@gamma(1)` = −`gamma(5)`，γ3·γ1·γ4 = `gamma(4)@gamma(1)@gamma(8)` = −`gamma(13)`，即生成端采用 γ3 在前的乘积次序，翻号补偿 bit 编码矩阵 γ1 在前的次序差。

### current 项的解析—桥接—组装流水线

1. `assemble_current_terms` 逐 term 调 `resolve_current_term_endpoints`（时间边界策略）；
2. resolver（如 `make_spin_aware_current_resolver` 包装的链）先由 `resolve_directed_current_raw` 取 raw 切片（不加权），再由 `resolve_current_term_spin` 施加唯一显式 spin matrix（`Γ` 或 `r·I±Γ`）得 `(4,4,sink_ne,source_ne)` vertex；
3. 组装器做唯一加权 `coefficient * normalization * value` 并累加；
4. 消费端 `spin_aware_current_adapter` + `consume_spin_aware_current` 用 `einsum("a,abij,b->ij")` 与显式 spin 权重收缩。

---

## ⑤ 不变量与校验

| 位置 | 不变量 |
|---|---|
| `lattice/insertion/gamma.py:76,96` | `gamma(n)`/`output(n)` 的 `0<=n<=15` 断言 |
| `lattice/insertion/gamma.py:173-193` | 查表函数 `assert name in _naming_scheme` |
| `lattice/insertion/gauge_link.py:235` | gauge path 相邻方向差 ≠3（不立即折返）；`transform`/`conjugate` 后重新校验 |
| `lattice/insertion/gauge_link.py:16` | 方向编码 0..5 = ±x,±y,±z 固定次序 |
| `lattice/insertion/__init__.py:152-155` | `Operator`：行数 == 系数数 |
| `lattice/insertion/__init__.py:166-169` | gamma idx ∈ {5,13} 时项系数翻号（γ3 在前的乘积次序约定，已数值验证自洽） |
| `lattice/insertion/__init__.py:206,211-215` | `OperatorDisplacement`：行数 == 距离数；每项 `derivative_idx == 0`（位移与导数互斥） |
| `lattice/insertion/__init__.py:306-310` | `construct`：`left=="A_1"` 时断言 `right==projection`；`left=="T_1"` 时断言 `projection ∈ irrep_T1[right]` |
| `lattice/insertion/__init__.py:227-228` | 动量必须存在于 `momentum_dict`（否则 `.index` 抛 `ValueError`） |
| `lattice/insertion/mom_dict.py:197` | `mom_dict_to_list` 仅接受 `mom∈{0,1,3,9}` |
| `lattice/insertion/current.py:410-453` | `CurrentTerm` 全字段校验（见 ②）；temporal endpoint 必须配 direction==3；`temporal_point_split` 与端点结构一致 |
| `lattice/insertion/current.py:52-110` | directed raw contract 键集合精确匹配、basis 指纹一致、Ne 对称、shape/Ne 一致、cache_identity 未篡改 |
| `lattice/insertion/current.py:200-272` | legacy spatial 四通道形状/Ne/Np 约束（见 ③）；三 dense 通道有限 complex；displacement 轴长度一致 |
| `lattice/insertion/current.py:274-298` | legacy 契约 + `require_temporal=True` → 直接报错；directed 契约与数据逐字段互证 |
| `lattice/insertion/current.py:646-668,830-868` | resolver 返回值必须含 `value/source_ne/sink_ne`、Ne 精确匹配、value 形状一致且非空有限 |
| `lattice/insertion/current.py:697-701,869-871` | 自旋矩阵与 coefficient×normalization 加权严格分离：桥接层只加 spin matrix，组装器只做唯一加权 |
| `lattice/insertion/current.py:901-911` | `lattice_divergence`：方向轴 shape[0]==4、site_axes 4 个唯一整数轴 |
| `lattice/insertion/current.py:519-541` | atol/rtol 非负、lattice_spacing>0、标量有限非 bool |

---

## ⑥ 相关测试

| 测试文件 | 覆盖内容 |
|---|---|
| `test/test_gamma.py` | `gamma(0)=np.eye(4)`（:32-37）、所有 `gamma(n)` 形状 (4,4)（:39-47）、γ5 Hermitian 且平方为 I（:50-58）、`output` 标签（:28-30）、量子数函数返回类型（:91-125）、`gamma_transform` 恒等与已知键值（:128-139） |
| `test/test_gauge_link.py` | 三种初始化等价、idx 往返、`transform("c4z")/iden` 一致、复合路径变换、与 idx 创建对象的交叉一致（:14-120+） |
| `test/test_gauge_links_methods.py` | `GaugeLink` + `mom_dict_to_list` + `CurrentElementalGenerator` 端到端方法测试（:15-19） |
| `test/test_insertion_simple.py` | **降级测试**：不 import 模块，而是读 `__init__.py` 源码文本断言包含 `class ProjectionName/Row/Insertion/Operator` 及关键 import（:31-60）；不验证运行行为 |
| `test/test_current.py` | term v1 兼容与非法值拒绝（:26,:70）、组装器只施加 coefficient×normalization（:82）、Ne provenance 与边界（:146,:185,:200,:327）、端点 periodic/open 策略（:216）、conserved temporal 与 legacy adapter 的 point-split 约束（:262,:360,:378）、`compute_elemental` 与 numpy generator 适配（:533,:599,:668）、局域流局域性 / LocalAxial γ_μγ5 / Conserved Wilson 中点项（:733,:747,:759）、Z/WT/PCAC 门控（:778,:791,:797）、WT/PCAC 不误报与空/非有限数组拒绝（:819,:837）、容差与 PCAC improvement（:862,:871,:891）、散度 periodic/non-periodic/spacing（:943,:962） |
| `test/test_conserved_charge_v2v.py` | `ConservedVectorCurrent.terms` + `build_current_raw_contract` 与 `lattice.current_elemental.contract_directed_current_v2v` 联用，验证 conserved-charge 显式（无隐式共轭）投影与比值构造 |
| `test/test_current_consumption.py` / `test_current_v2v_contraction.py` | 消费端 VSV 存储桥不得自行加载传播子（:17-30）、逐 term/pair 的 V2V 收缩 schema |
| `test/test_current_elemental.py` | 生成器侧 V2P/P2V/P2P/OverlapMatrix 数据类测试（:21-181） |
| `test/test_current_vertex.py` | `QuarkDiagram` 带 current vertex 的图展开为 VSV/PSV/VSP/PSP 多图（:1-25 docstring 与 :14-118 测试） |
| `test/test_group_projection.py`、`test/test_insertion_simple.py`（grep 确认） | 间接涉及 `derivative.py`；`derivative.py` 无专门测试文件 |
| （无专门文件） | `phase.py` 无直接测试，经 `lattice/generator/` 系列测试间接覆盖 |

运行验证：`pytest test/test_gamma.py test/test_gauge_link.py test/test_gauge_links_methods.py test/test_insertion_simple.py test/test_current.py -q` → 156 passed, 1 skipped。
