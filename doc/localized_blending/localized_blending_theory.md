# Localized Blending：格点 QCD all-to-all 相关函数的无偏抽样理论

本文档把方法的目标限定为：在固定规范场上，用有限个空间点近似 all-to-all 传播子所构造的**完整相关函数**，并使相关函数估计量对点集随机性严格无偏。低模/高模分解是计算手段；目标量仍是指定插值算符、流和 Wick 缩并所定义的全空间求和。

这里必须区分三个问题：

1. **代数正确性**：低模与补空间是否完备，所有 mode sector 和 Wick scene 是否恰好计入一次；
2. **统计正确性**：在明确的点集抽样设计下，是否有

   $$
   \mathbb E_A[\widehat C(U,A)\mid U]=C(U);
   \tag{1}
   $$

3. **计算效率**：达到相同物理精度时，方差、求逆数、存储和墙钟成本是否优于基准方法。

有限样本中的“不拒绝差异”不能证明第 2 点，低方差也不能补救有偏估计，第 2 点成立也不自动意味着第 3 点成立。

## 1. 目标相关函数与条件无偏性

令 $U$ 表示一个固定规范场，$D_U$ 为 Dirac 算符，$S_U=D_U^{-1}$ 为精确 all-to-all 传播子。给定插值算符、流、动量投影和 Wick 拓扑后，精确相关函数记为 $C(U)$。点集抽样 $A$ 是额外的算法随机性。

本方法要求的是设计无偏性，即式 (1) 对每个固定 $U$ 成立。随后由迭代期望，

$$
\mathbb E_U\mathbb E_A[\widehat C(U,A)\mid U]
=\mathbb E_U[C(U)].
\tag{2}
$$

因此，点集无偏性可以在小格子、固定传播子上穷举验证，不必依赖大量规范组态。规范平均是否代表 QCD 系综则是另一层问题，涉及热化、自相关、有限体积和离散化等误差。

本文的“无偏”首先指原始相关函数或其线性组合。比值 $C_3/C_2$、有效质量 $\log[C(t)/C(t+1)]$、广义本征值和非线性拟合参数即使由无偏相关函数构造，通常也不是有限样本严格无偏的；它们需要单独的 jackknife/bootstrap、拟合稳定性和系统误差分析。

## 2. 低模/补空间恒等分解

在每个时间片上，空间–颜色空间的维数为 $N_cM$，其中 $M=L^3$。取规范协变空间 Laplace 算符最低的 $N_e$ 个正交本征向量 $v_i(t)$，定义

$$
P_e(t)=\sum_{i=1}^{N_e}|v_i(t)\rangle\langle v_i(t)|,
\qquad
Q_e(t)=I-P_e(t).
\tag{3}
$$

$P_e^2=P_e$、$Q_e^2=Q_e$、$P_eQ_e=0$，且 $P_e+Q_e=I$。传播子端点的完整分解为

$$
S=(P_e+Q_e)S(P_e+Q_e)
=P_eSP_e+P_eSQ_e+Q_eSP_e+Q_eSQ_e.
\tag{4}
$$

双线性算符 $\mathcal O$ 也有相同的 LL、LH、HL、HH 四块。相关函数必须在展开后保留所有非零 sector；只计算 LL 或删去“看起来小”的混合项，一般会改变目标量而产生截断偏差。

用空间点基 $|x,a\rangle$ 表示补空间时，颜色 $a$ 在每个已选空间点上精确求和：

$$
Q_e
=Q_e I Q_e
=\sum_{x\in\mathcal U}\sum_{a=1}^{N_c}
Q_e|x,a\rangle\langle x,a|Q_e,
\qquad |\mathcal U|=M.
\tag{5}
$$

因此这里抽样的有限总体是 $M$ 个**空间坐标**，不是 $N_cM$ 个 site-color 对。若代码实际逐个抽 site-color 对，或使用补空间中的 Haar 随机正交向量，有限总体和权重必须相应改变，不能直接套用本文的 $M/N_p$。Hu 等人的原始 blending 论文使用的是补空间维数 $\dim\mathcal L_2=N_cM-N_e$ 下的随机正交向量和 $\omega_n$ 权重；本项目的 localized point sampling 是另一种抽样设计，但可用同一联合包含概率原则证明。

## 3. 一般抽样定理

### 3.1 抽样组

一个“抽样组” $g$ 表示所有真正共享同一个随机子集的点端。设

$$
A_g\subset\mathcal U,
\qquad |A_g|=N_g,
\tag{6}
$$

且 $A_g$ 是从 $M$ 个空间点中简单随机、无放回抽取。不同时间片、不同随机种子或明确独立生成的子集应属于不同组；同一时间片上复用同一子集的多个传播子端或多个流属于同一组。

把某一完整相关函数展开为有限总体求和。对组 $g$，一个具体项含 $r_g$ 个带标签的点位置

$$
\mathbf x_g=(x_{g1},\ldots,x_{gr_g})\in\mathcal U^{r_g}.
\tag{7}
$$

这些位置可以相等。令 $d_g(\mathbf x_g)$ 为其中不同空间点的集合，$k_g=|d_g|$。

### 3.2 联合包含概率与权重

无放回等概率抽样中，任意指定的 $k$ 个不同点同时进入 $A_g$ 的概率为

$$
\pi_{g,k}
=\frac{\binom{M-k}{N_g-k}}{\binom{M}{N_g}}
=\frac{(N_g)_k}{(M)_k},
\tag{8}
$$

其中 $(n)_k=n(n-1)\cdots(n-k+1)$ 是下降阶乘。因此 Horvitz–Thompson 权重为

$$
w_{g,k}=\pi_{g,k}^{-1}=\frac{(M)_k}{(N_g)_k}.
\tag{9}
$$

若各组独立，一个总体项的总包含概率和权重分别为

$$
\Pi(\mathbf x)=\prod_g\pi_{g,k_g},
\qquad
W(\mathbf x)=\prod_g w_{g,k_g}.
\tag{10}
$$

于是对

$$
C(U)=\sum_{\mathbf x_1\in\mathcal U^{r_1}}\cdots
\sum_{\mathbf x_G\in\mathcal U^{r_G}}
F_U(\mathbf x_1,\ldots,\mathbf x_G),
\tag{11}
$$

定义

$$
\widehat C(U,A)=
\sum_{\mathbf x_1\in A_1^{r_1}}\cdots
\sum_{\mathbf x_G\in A_G^{r_G}}
W(\mathbf x)
F_U(\mathbf x_1,\ldots,\mathbf x_G).
\tag{12}
$$

对任意固定总体项，它被观察到的概率是 $\Pi(\mathbf x)$，观察后乘以 $1/\Pi(\mathbf x)$，所以其期望恰为原项。对所有项使用期望的线性性即得式 (1)。复相关函数的实部和虚部分别满足同一证明。

若各组不独立，式 (10) 的乘积一般失效，必须使用实际的跨组联合包含概率。若是非均匀、分层、排斥距离或自适应抽样，也必须用该设计真实且非零的 $\Pi(\mathbf x)$；仅使用 $M/N_g$ 通常会有偏。

### 3.3 可估条件

所有目标总体项都必须有正包含概率。对固定大小的无放回样本，这要求

$$
N_g\ge \max_{\mathbf x:F_U(\mathbf x)\ne0} k_g.
\tag{13}
$$

若 $N_g<k_g$，该类项永远不会被观察，不能用有限权重恢复。代码返回权重 0 不是“安全退化”，而是把该总体项删掉；除非能证明该项恒为零，否则估计量有偏或不可识别。

## 4. scene 必须是带标签集合划分

同一抽样组的 $r$ 个点位置需要按相等关系分 scene。正确对象是标签集合 $\{1,\ldots,r\}$ 的**集合划分**，不是只记录块大小的整数划分。

例如 $r=3$ 有五个 scene：

$$
123,
\quad 12|3,
\quad 13|2,
\quad 23|1,
\quad 1|2|3.
\tag{14}
$$

中间三个 scene 的块大小都为 $[2,1]$，权重相同，但 $F(x,x,y)$、$F(x,y,x)$ 和 $F(y,x,x)$ 一般不同，不能只计算其中一个。scene 数是 Bell 数

$$
B_0,B_1,B_2,B_3,B_4=1,1,2,5,15,
\tag{15}
$$

而不是整数划分数 $1,1,2,3,5$。每个集合划分有 $k$ 个块时使用 $w_{g,k}$；算符中的 Kronecker delta、位移约束和 Wick 对称性应在 scene 枚举之后应用，只有经过证明的等价项才能合并，并保留正确的多重度。

## 5. 基本特殊情形

### 5.1 无抽样点或一个抽样点

没有点端时权重为 1。只有一个点端时，

$$
\widehat T_1=\frac{M}{N}\sum_{x\in A}F(x),
\qquad
\mathbb E_A[\widehat T_1]=\sum_{x\in\mathcal U}F(x).
\tag{16}
$$

### 5.2 两个点来自同一子集

精确总和 $T_2=\sum_{x,y\in\mathcal U}F(x,y)$ 的无偏估计为

$$
\widehat T_2=
\frac{M}{N}\sum_{x\in A}F(x,x)
+\frac{M(M-1)}{N(N-1)}
\sum_{\substack{x,y\in A\\x\ne y}}F(x,y).
\tag{17}
$$

等价地，

$$
\widehat T_2=w_2\sum_{x,y\in A}F(x,y)
+(w_1-w_2)\sum_{x\in A}F(x,x).
\tag{18}
$$

这正是同一随机表示被二次使用时需要的二阶修正。仅有 $\mathbb E[\widehat S]=S$ 并不能推出 $\mathbb E[\widehat S\widehat S]=SS$；共享随机点集带来的协方差必须通过联合包含概率或独立副本处理。

### 5.3 两个点来自独立子集

若 $x\in A_1$、$y\in A_2$，且两个子集独立，

$$
\widehat T_{1\times1}
=\frac{M}{N_1}\frac{M}{N_2}
\sum_{x\in A_1}\sum_{y\in A_2}F(x,y).
\tag{19}
$$

这里 $x=y$ 完全允许；“来自不同集合”不等于“空间坐标不相同”。跨组强行加入 $x\ne y$ 会改变目标总和并造成偏差。

### 5.4 局域与非局域双线性算符

局域流 $\bar q(x)\Gamma q(x)$ 令左右端点相同，所以 pp sector 只有一个不同空间点，使用 $w_1$。空间位移流 $\bar q(x)\Gamma U(x,x+d)q(x+d)$ 在 $d\ne0$ 时通常需要同组中的两个不同点，使用 $w_2$。若两端由不同独立设计抽样，则使用两个一阶权重的乘积。

结构化点集如果保证 $x\in A\Rightarrow x+d\in A$，其包含概率不再是式 (8)，必须按结构化设计重新推导。时间点分裂的守恒流还需要按两端实际所在时间片决定抽样组，不能只看算符名称。

## 6. 各类相关函数

### 6.1 current $\rightarrow$ meson

若 meson 端由精确低模 distillation 算符构造，而 current 端允许低模或采样点表示，则 current 有四个 sector：

$$
vv,\quad vp,\quad pv,\quad pp.
\tag{20}
$$

$vv$ 无点权重；$vp$ 和 $pv$ 各有一个采样点，使用 $w_1$；$pp$ 的权重由 current 的局域性决定：局域流因两端相同使用 $w_1$，一般双局域流需区分相同点 $w_1$ 与不同点 $w_2$。四个 sector 的完整和才是目标相关函数的无偏估计。省略混合 sector 即使数值上较小，也不是无偏近似。

### 6.2 非等时 current $\rightarrow$ current

两个流位于不同时间片且各时间片独立抽点时，它们属于两个抽样组，权重按式 (10) 相乘。两个局域流的典型权重为 $w_{1,t_1}w_{1,t_2}$；两个一般双局域流最多分别需要 $w_{2,t_1}w_{2,t_2}$。相同的空间坐标可同时出现在两个时间片样本中，因为它们是不同的时空点和不同抽样事件。

每个 current 的左右端均可为 $v$ 或 $p$，因此一般有 $4\times4=16$ 个 mode state。对每个 state 正确加权并验证 state sum 闭合，是必要的内部检查，但不是外部无偏性的充分证明。

### 6.3 等时 current $\rightarrow$ current 与接触项

当两个流同处一个时间片并复用同一随机点集时，所有点端属于同一抽样组，不能把两个 current 当独立组处理。

对两个一般双局域 pp 流，最多有四个带标签点端，理论上应覆盖 $B_4=15$ 个集合划分。若两个流都严格局域，算符约束先令各流内部左右端相等，最终只剩两个物理插入坐标：两坐标相同用 $w_1$，不同用 $w_2$。接触项、守恒流时间位移、反周期边界和 $\gamma_5$-hermiticity 路径仍需独立验证。

等时 $\Delta t=0$ 数据是接触项/代数实现的验证对象，不应进入有效质量、cosh 或谱拟合。

### 6.4 更高点、重子与 disconnected loops

对任意 $n$ 点函数，先按“真实共享的子集身份 + 时间片”分组，再对每组的所有带标签点端枚举集合划分，最后对独立组取笛卡尔积并乘权重。局域 $n$ 个流若在同一时间片共享样本，至多需 $B_n$ 个插入坐标 scene；双局域流可能达到 $B_{2n}$，随后由算符约束删去恒零项。

断开圈通过规范场相关并不改变点集权重。决定权重的是算法上是否复用同一随机子集：复用则属于同组并使用高阶联合包含概率；独立抽样才可分解为权重乘积。

## 7. $N_e$ 与 $N_p$ 分别控制什么

### 7.1 $N_e$：确定性分解或物理算符参数

若 $N_e$ 只改变恒等式的分解，目标算符保持不变，且所有 P/Q sector 都完整计算，则

$$
\mathbb E_A[\widehat C_{N_e,N_p}(U,A)\mid U]=C(U)
\tag{21}
$$

应与 $N_e$ 无关。单次随机 realization 和各 sector 会随 $N_e$ 变化；总方差也不保证单调，因为 sector 间协方差会同时变化。

但若外部 hadron 插值算符本身定义为 $P_e$ smearing，改变 $N_e$ 就改变有限时距的相关函数和激发态重叠。这时不能把不同 $N_e$ 曲线当作同一总体量的无偏重复；只有在共同物理目标（例如控制激发态后的基态矩阵元）上比较才有意义。

### 7.2 $N_p$：点集覆盖与抽样方差

在抽样设计、权重和可估条件均正确时，改变 $N_p$ 不改变期望，只改变方差和成本。对单点总体总和，简单随机无放回给出

$$
\operatorname{Var}_A(\widehat T_1\mid U)
=M^2\left(1-\frac{N_p}{M}\right)\frac{S_F^2}{N_p},
\tag{22}
$$

其中 $S_F^2$ 是有限总体方差。多点函数没有只依赖 $N_p$ 的通用方差公式；它还取决于 $F$ 的空间结构、不同 scene 的协方差以及不同抽样组是否共享随机数。由于

$$
w_k=\frac{(M)_k}{(N_p)_k}
\tag{23}
$$

随 $k$ 增长得更快，含更多不同采样坐标的相关函数通常对小 $N_p$ 更敏感，但这只是结构性预期，不能替代逐通道测量。

### 7.3 效率必须在相同目标和成本下定义

可用的比较量是固定目标、固定误差预算下的成本，或例如

$$
\mathrm{FOM}=\operatorname{Var}(\widehat C)\times
\mathrm{cost},
\tag{24}
$$

其中 cost 至少记录 Dirac 求逆数、墙钟、GPU-hours、内存与 I/O。若目标或插值算符不同，或者一个结果只含 vv 而另一个含完整 4/16 states，就不是效率对比。

## 8. 当前实测数据支持到什么程度

当前权威包是 [localized-blending-delivery-v1.4.0](../../../final-results/localized-blending-delivery-v1.4.0/REPORT.md)，统计单位为先平均 18 个源时刻后的 8 个规范组态。结论必须按证据层级解释。

### 8.1 已有正面证据

- 4-state current$\rightarrow$meson 与 16-state current$\rightarrow$current 的 state sum 对 total 闭合：展示点最大绝对差 $1.82\times10^{-12}$，全部非零时距 cell 最大差 $1.73\times10^{-18}$。这强力支持 sector bookkeeping，但不是对精确 all-point 目标的外部验证。
- 非零时距 $\Delta t=20\ldots51$、$N_p=64$ 时，对 $N_e=32,64,96$ 与 128 的联合配对 sign-flip 检验，current$\rightarrow$current 的 $p$ 值为 $0.3906,0.3906,0.7109$；current$\rightarrow$meson 为 $0.5625,0.2891,0.9453$。没有发现可检出的系统差异，但非拒绝不等于等价，也不能证明期望严格相同。
- 等时 $N_p=8\rightarrow64$ 扫描中，configuration-level 标准误之比约为 4.19（current$\rightarrow$current）和 1.67（current$\rightarrow$meson）。锚定 working model 给出 $\gamma_{N_p}=0.78\pm0.38$ 与 $0.30\pm0.13$。这支持“current$\rightarrow$current 对稀疏点采样更敏感”的通道差异。
- 等时 $N_e=0\rightarrow128$ 的标准误之比约为 1.002 和 1.093；working model 的 $\beta_{N_e}=0.0017\pm0.0025$ 与 $0.088\pm0.092$。在此数据范围内，改变 $N_e$ 对总配置误差的影响弱于改变 $N_p$，但误差条很大。

### 8.2 统计限制

- 只有 8 个独立规范组态。sign-flip 的中心对称条件未被验证；Hotelling 检验又依赖高斯均值模型。等时 current$\rightarrow$current 的 $N_e$ 联合检验对方法敏感：sign-flip $p=0.0781$，Hotelling $p=0.0489$。
- 等时 $N_p$ 联合检验未拒绝参考点差异：current$\rightarrow$current 的 sign-flip $p=0.7578$，current$\rightarrow$meson 为 $0.1797$；这仍不是预先给定容差下的等价检验。
- 没有同一 cfg/source/parameter 下的独立点集重复，所以总配置方差无法拆成规范噪声与点抽样噪声；$N_p$ 趋势也可能含共享输入导致的协方差。
- 没有完整 $N_e\times N_p$ 因子网格，交互项不可识别；当前加性模型把交互固定为零。
- 没有非零时距 $N_p$ 扫描、没有 $N_p=M$ 的 all-point 参考、没有匹配的传统方法成本，也没有规范链自相关/分块分析。因此当前数据不支持“已证明无偏”或“已证明更高效”的强结论。

最稳健的经验性表述是：**在已验证实现和现有 8 cfg 数据上，完整 state sum 与内部代数闭合；未观察到明确的 $N_e$ 依赖；降低 $N_p$ 时 current$\rightarrow$current 的误差增长明显强于 current$\rightarrow$meson。严格无偏性主要来自抽样设计证明，效率仍待匹配成本实验确认。**

## 9. 当前实现的一般性缺口

理论要求 $r=3,4$ 时分别有 5、15 个带标签集合划分。当前活动源码 [quark_diagram.py](../../lattice/quark_diagram.py) 使用整数划分，现有 [test_sampling_weight.py](../../test/test_sampling_weight.py) 还把 $r=3,4$ 的期望 scene 数写成 3、5。因此：

- 活动源码对 $r\le2$ 的权重结构正确；
- 对 $r\ge3$ 的一般非对称相关函数，除非算符约束能被证明使遗漏的标签划分恒零或已用正确多重度合并，否则不能宣称无偏；
- 活动源码在 $k>N_p$ 时返回 0，也应改为显式“不可估”错误或在构图前证明这些项恒零。

v1.4 数值证据所对应的归档候选核心使用了 `point_set_partitions`，可枚举 $1,2,5,15$ 个 scene；这解释了为什么“已验证结果包”与“当前活动源码的一般性”必须分开表述。理论文档不能用特殊的局域/等时成功案例替一般 $n$ 点函数背书。

## 10. 必须补齐的验证设计

1. 在小格子上固定 $U,S,\mathcal O$，对每个抽样组穷举全部 $\binom{M}{N_p}$ 个子集；对 $r=1\ldots4$ 的每个带标签集合划分逐项验证式 (1)。
2. 分别覆盖局域流、非零空间位移流、跨时间守恒流、同一时间片双流、不同时间片独立流和复用点集的 disconnected loops。
3. 加入 $N_p=M$ 的精确 all-point oracle；比较的是逐配置、逐时距、逐 state 的同一目标，而不仅是规范平均后的曲线。
4. 在每个 cfg/source/$N_e$/$N_p$ cell 生成多个独立点集，估计 $\operatorname{Var}_A(\widehat C\mid U)$；再用 law of total variance 分离点噪声与规范噪声。
5. 做完整 $N_e\times N_p$ 因子网格；若 $N_e$ 改变外部插值算符，另设“固定物理目标”分析，不把曲线相等当必要条件。
6. 预先给出物理可接受容差并做配对等价检验/置信区间；同时报告效应量，不把 $p>0.05$ 写成无偏证明。
7. 记录求逆、墙钟、GPU-hours、内存和 I/O，在相同完整 observable 上报告方差–成本前沿。
8. $\Delta t=0$ 只作为接触项与实现 oracle；谱学结论使用非零时距并单独处理自相关和拟合系统误差。

## 11. 实现契约

一个可宣称“一般相关函数无偏”的实现至少应满足：

- 抽样设计、随机子集身份、时间片与随机种子在数据中可追溯；
- 每个点端被分配到正确抽样组；
- 同组使用带标签集合划分，独立组才取权重乘积；
- 权重由真实联合包含概率计算，且所有非零总体项概率为正；
- LL/LH/HL/HH 及所有 Wick 拓扑完整且不重复；
- 局域/位移/接触约束不被误当作抽样独立性；
- 固定输入的穷举设计均值等于 all-point oracle；
- 原始相关函数的无偏性与派生比值/拟合量的统计性质分别报告。

## 12. 文献与项目证据

### 原始文献

1. D. G. Horvitz and D. J. Thompson, “A Generalization of Sampling Without Replacement from a Finite Universe,” *JASA* **47** (1952) 663–685, [DOI](https://doi.org/10.1080/01621459.1952.10483446)。本文的有限总体联合包含概率推导是其设计无偏思想在带标签点元组上的直接应用。
2. M. Peardon et al., “A novel quark-field creation operator construction for hadronic physics in lattice QCD,” *Phys. Rev. D* **80** (2009) 054506, [arXiv:0905.2160](https://arxiv.org/abs/0905.2160)。给出 distillation 的 Laplace 低模投影与 reduced all-to-all propagator。
3. C. Morningstar et al., “Improved stochastic estimation of quark propagation with Laplacian Heaviside smearing in lattice QCD,” *Phys. Rev. D* **83** (2011) 114505, [arXiv:1104.3870](https://arxiv.org/abs/1104.3870)。给出 stochastic LapH 与 dilution 的方差降低框架。
4. Z.-C. Hu et al., “Realization of all-to-all fermion propagator for the first principle high accuracy strong interaction prediction,” [arXiv:2505.01719v2](https://arxiv.org/abs/2505.01719)。式 (1)–(3) 与补充材料式 (10)–(27)讨论 blending 权重和二阶矩；本文进一步把坐标点无放回抽样推广到任意相关函数的集合划分形式。
5. J. Foley et al., “Practical all-to-all propagators for lattice QCD,” *Comput. Phys. Commun.* **172** (2005) 145–162, [arXiv:hep-lat/0505023](https://arxiv.org/abs/hep-lat/0505023)。给出低模精确部分与噪声/dilution all-to-all 估计的早期实现框架。

### 当前项目证据

- [v1.4 科学结论](../../../final-results/localized-blending-delivery-v1.4.0/VERDICT.json)
- [v1.4 非零时距有限样本分析](../../../final-results/localized-blending-delivery-v1.4.0/integration/nonzero-full-rerun/nonzero-spinfix-finite-sample-v1.json)
- [等时小网格分析](../../../final-results/localized-blending-delivery-v1.4.0/integration/dt0-small-grid-supplement/evidence/dt0-smallgrid-analysis-v1.json)
- [$\gamma_5$-hermiticity 与有限样本复核](../../../docs/archive/evidence/validation-work-reports/dt0-spinfix-v1.0.3.md)

历史 `v1.0.x` 汇总中若与 v1.4 当前 rerun 冲突，应视为审计历史而非当前统计结论。
