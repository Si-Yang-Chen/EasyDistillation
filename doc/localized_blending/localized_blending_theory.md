# Localized Blending 理论部分

本文档说明 Localized Blending 方法的理论基础。

## 理论部分

有一个维数 $3 \cdot L^3$ 大小的空间，他的基底可以用三个空间坐标和颜色标志表示，即

$$
I=\sum_{x=0}^{L-1}\sum_{y=0}^{L-1}\sum_{z=0}^{L-1}\sum_{a=0}^{2}|x,y,z,a\rangle \langle x,y,z,a|
\tag{1.1}
$$

其中：

- $(x,y,z)$ 表示空间坐标，每个维度取值范围为 $[0, L-1]$
- $a \in \{0,1,2\}$ 表示颜色通道（如 RGB 三个通道）

假设我们有所有的点，对 $I$ 的表示为：

$$
\hat{I} = \sum_{x=1}^{M} \sum_{a=0}^{2} |\eta_x, a\rangle \langle \eta_x, a|
\tag{1.2}
$$

其中：

- $M = L^3$ 是所有空间点的总数
- $\eta_x$ 是所有的空间点
- 每个点 $\eta_x$ 包含该空间点的所有颜色通道 $a \in \{0,1,2\}$
- 因为使用了所有点，不需要权重因子

另一种基底是拉普拉斯算符的本征向量 $\xi_i$，即：

$$
\nabla^2 |\xi_i\rangle = \lambda_i |\xi_i\rangle
\tag{1.3}
$$

其中：

- $\nabla^2$ 是拉普拉斯算符
- $\lambda_i$ 是相应的本征值
- $|\xi_i\rangle$ 是本征向量，构成该空间的正交归一基底

前 $N_v$ 个本征向量是特殊的（low mode），在这种基底表示下，单位矩阵可以分解为：

$$
I = I_{\text{low}} + I_{\text{high}}
\tag{2.1}
$$

其中 low mode 部分：

$$
I_{\text{low}} = \sum_{i=1}^{N_v} |\xi_i\rangle \langle \xi_i|
\tag{2.2}
$$

high mode 部分：

$$
I_{\text{high}} = I - I_{\text{low}} = \sum_{k=N_v+1}^{3 \cdot L^3} |\xi_k\rangle \langle \xi_k|
\tag{2.3}
$$

将一个算符 $\mathcal{O}$ 投影到 eigenvector 的 low mode 和 high mode 上：

**Low mode 投影：**

$$
\mathcal{O}_{\text{low}} = I_{\text{low}} \mathcal{O} I_{\text{low}} = \sum_{i=1}^{N_v} \sum_{j=1}^{N_v} |\xi_i\rangle \langle \xi_i| \mathcal{O} |\xi_j\rangle \langle \xi_j| = \sum_{i=1}^{N_v} \sum_{j=1}^{N_v} \mathcal{O}_{i,j} |\xi_i\rangle \langle \xi_j|
\tag{3.1}
$$

其中 $\mathcal{O}_{i,j} = \langle \xi_i| \mathcal{O} |\xi_j\rangle$ 是算符 $\mathcal{O}$ 在本征向量基底下的矩阵元。

**High mode 投影：**

$$
\mathcal{O}_{\text{high}} = (I - I_{\text{low}}) \mathcal{O} (I - I_{\text{low}}) = \sum_{k=N_v+1}^{3 \cdot L^3} \sum_{l=N_v+1}^{3 \cdot L^3} |\xi_k\rangle \langle \xi_k| \mathcal{O} |\xi_l\rangle \langle \xi_l| = \sum_{k=N_v+1}^{3 \cdot L^3} \sum_{l=N_v+1}^{3 \cdot L^3} \mathcal{O}_{k,l} |\xi_k\rangle \langle \xi_l|
\tag{3.2}
$$

**Low 到 High 的混合项：**

$$
\mathcal{O}_{\text{low-high}} = I_{\text{low}} \mathcal{O} (I - I_{\text{low}}) = \sum_{i=1}^{N_v} \sum_{k=N_v+1}^{3 \cdot L^3} |\xi_i\rangle \langle \xi_i| \mathcal{O} |\xi_k\rangle \langle \xi_k| = \sum_{i=1}^{N_v} \sum_{k=N_v+1}^{3 \cdot L^3} \mathcal{O}_{i,k} |\xi_i\rangle \langle \xi_k|
\tag{3.3}
$$

**High 到 Low 的混合项：**

$$
\mathcal{O}_{\text{high-low}} = (I - I_{\text{low}}) \mathcal{O} I_{\text{low}} = \sum_{k=N_v+1}^{3 \cdot L^3} \sum_{j=1}^{N_v} |\xi_k\rangle \langle \xi_k| \mathcal{O} |\xi_j\rangle \langle \xi_j| = \sum_{k=N_v+1}^{3 \cdot L^3} \sum_{j=1}^{N_v} \mathcal{O}_{k,j} |\xi_k\rangle \langle \xi_j|
\tag{3.4}
$$

完整的算符分解为：

$$
\mathcal{O} = \mathcal{O}_{\text{low}} + \mathcal{O}_{\text{high}} + \mathcal{O}_{\text{low-high}} + \mathcal{O}_{\text{high-low}}
\tag{3.5}
$$

因为 high mode 部分并没有数据，所以需要将 high mode 投影到空间点上。

## 使用全部点的情况

当使用所有点时（$N = M = L^3$），权重因子为 1，公式简化。

将 $I_{\text{high}} = I - I_{\text{low}}$ 投影到所有点 $\eta_x$ 上为：

$$
\hat{I}_{\text{high}} = (I - I_{\text{low}}) \hat{I} (I - I_{\text{low}}) = \sum_{x=1}^{M} \sum_{a=0}^{2} |\tilde{\eta}_x, a\rangle \langle \tilde{\eta}_x, a|
\tag{4.1}
$$

其中 $M = L^3$ 是空间点的总数，$|\tilde{\eta}_x, a\rangle = (I - I_{\text{low}})|\eta_x, a\rangle$ 是投影后的向量。

类似地，high mode 算符投影为：

$$
(I - I_{\text{low}}) \hat{\mathcal{O}} (I - I_{\text{low}}) = \sum_{x=1}^{M} \sum_{y=1}^{M} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{xa,yb} |\tilde{\eta}_x, a\rangle \langle \tilde{\eta}_y, b|
\tag{4.2}
$$

其中 $\hat{\mathcal{O}}$ 是算符 $\mathcal{O}$ 通过所有点得到的：

$$
\hat{\mathcal{O}} = \sum_{x=1}^{M} \sum_{y=1}^{M} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{xa,yb} |\eta_x, a\rangle \langle \eta_y, b|
\tag{4.3}
$$

其中 $\mathcal{O}_{xa,yb} = \langle \eta_x, a| \mathcal{O} |\eta_y, b\rangle$ 是算符 $\mathcal{O}$ 在空间点基底下的矩阵元（包含颜色指标）。注意 $\mathcal{O}$ 只能由 $\eta$ 和 $\xi$ 展开，旁边的向量（bra 和 ket）可以写成组合的形式。

对于混合项，投影算符插在最外侧且两端下标不同：

**Low 到 High 混合项：**

$$
I_{\text{low}} \hat{\mathcal{O}} (I - I_{\text{low}}) = \sum_{i=1}^{N_v} \sum_{y=1}^{M} \sum_{a=0}^{2} \mathcal{O}_{i,ya} |\xi_i\rangle \langle \tilde{\eta}_y, a|
\tag{4.4}
$$

其中 $\mathcal{O}_{i,ya} = \langle \xi_i| \mathcal{O} |\eta_y, a\rangle$。

**High 到 Low 混合项：**

$$
(I - I_{\text{low}}) \hat{\mathcal{O}} I_{\text{low}} = \sum_{x=1}^{M} \sum_{j=1}^{N_v} \sum_{a=0}^{2} \mathcal{O}_{xa,j} |\tilde{\eta}_x, a\rangle \langle \xi_j|
\tag{4.5}
$$

其中 $\mathcal{O}_{xa,j} = \langle \eta_x, a| \mathcal{O} |\xi_j\rangle$。

## 部分点采样的无偏估计

在实际计算中，我们可能无法使用所有 $M = L^3$ 个空间点，而是随机采样 $N < M$ 个点。此时需要对单位矩阵和算符做无偏估计。

**采样点的单位矩阵估计**：

如果从 $M$ 个点中随机抽取 $N$ 个点 $\{\eta_{x_1}, \eta_{x_2}, \ldots, \eta_{x_N}\}$，则单位矩阵的无偏估计为：

$$
\hat{I} = \frac{M}{N} \sum_{i=1}^{N} \sum_{a=0}^{2} |\eta_{x_i}, a\rangle \langle \eta_{x_i}, a|
\tag{4.0a}
$$

其中权重因子 $\frac{M}{N}$ 保证了估计的无偏性：$\mathbb{E}[\hat{I}] = I$。

**采样点的算符估计**：

算符 $\mathcal{O}$ 需要区分左右两端点是否相同：

$$
\hat{\mathcal{O}} = \frac{M(M-1)}{N(N-1)} \sum_{\substack{i,j=1 \\ i \neq j}}^{N} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{x_i a, x_j b} |\eta_{x_i}, a\rangle \langle \eta_{x_j}, b|
+ \frac{M}{N} \sum_{i=1}^{N} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{x_i a, x_i b} |\eta_{x_i}, a\rangle \langle \eta_{x_i}, b|
\tag{4.0b}
$$

**推导**：

- **不同点项**：抽取前有 $M(M-1)$ 种有序对，抽取后有 $N(N-1)$ 种，补偿系数 $\frac{M(M-1)}{N(N-1)}$
- **相同点项**：抽取前有 $M$ 种，抽取后有 $N$ 种，补偿系数 $\frac{M}{N}$
- 左右两端独立遍历采样点索引 $i, j$
- 当 $N = M$ 时，两个补偿系数均为 1

**高模投影的估计**：

将 $I_{\text{high}} = I - I_{\text{low}}$ 投影到采样点上为：

$$
\hat{I}_{\text{high}} = (I - I_{\text{low}}) \hat{I} (I - I_{\text{low}}) = \frac{M}{N} \sum_{i=1}^{N} \sum_{a=0}^{2} |\tilde{\eta}_{x_i}, a\rangle \langle \tilde{\eta}_{x_i}, a|
\tag{4.0c}
$$

算符的高模投影估计（同样需要区分左右点是否相同）：

$$
(I - I_{\text{low}}) \hat{\mathcal{O}} (I - I_{\text{low}}) = \frac{M(M-1)}{N(N-1)} \sum_{\substack{i,j=1 \\ i \neq j}}^{N} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{x_i a, x_j b} |\tilde{\eta}_{x_i}, a\rangle \langle \tilde{\eta}_{x_j}, b|
+ \frac{M}{N} \sum_{i=1}^{N} \sum_{a=0}^{2} \sum_{b=0}^{2} \mathcal{O}_{x_i a, x_i b} |\tilde{\eta}_{x_i}, a\rangle \langle \tilde{\eta}_{x_i}, b|
\tag{4.0d}
$$

**在缩并计算中的应用**：

对于 $Tr[O_1 S O_2 S]$ 等包含点对的计算，需要根据每对点是否相同应用补偿：

$$
Tr[\cdots] = \sum_{\text{point pairs}} w_{ij} \times (\text{contraction result for points } x_i, x_j)
$$

其中权重 $w_{ij}$ 为：

- $w_{ij} = \frac{M(M-1)}{N(N-1)}$ 当 $i \neq j$（不同点对）
- $w_{ij} = \frac{M}{N}$ 当 $i = j$（相同点）

**具体实现**：

- 在程序中，$N$ = `usedNp`，$M = L^3$
- 完成缩并后，需要识别哪些项对应相同点、哪些对应不同点
- 对 High-High 项（公式 6.2）：涉及四个点 $(x, y, z, r)$，需要根据点对 $(x,y)$ 和 $(z,r)$ 是否相同分别应用补偿
- 对 Low-High/High-Low 混合项（公式 6.3, 6.4）：涉及两个点，根据这两个点是否相同应用补偿
- 采样点的选择通常是随机的或根据物理考虑（如均匀分布）

## 实际计算：$Tr[O_{1,\text{low}} \; S \; O_2 \; S]$ 的展开

实际计算的是 $Tr[O_{1,\text{low}} \; S \; O_2 \; S]$，其中 $S$ 是传播子（propagator），$O_1$ 仅取其 low 部分，$O_2$ 保持一般形式（包含 low、high 与混合）。

**传播子矩阵元的简化标记（左右用逗号分隔）**：

- $S_{i,j} = \langle \xi_i| S |\xi_j\rangle$：low mode 之间的传播子矩阵元
- $S_{xa,yb} = \langle \eta_x, a| S |\eta_y, b\rangle$：空间点之间的传播子矩阵元（不带投影）
- $S_{xa,i} = \langle \eta_x, a| S |\xi_i\rangle$：空间点到 low mode 的传播子矩阵元（不带投影）
- $S_{i,xa} = \langle \xi_i| S |\eta_x, a\rangle$：low mode 到空间点的传播子矩阵元（不带投影）
- $\mathcal{O}_{i,j} = \langle \xi_i| \mathcal{O} |\xi_j\rangle$：算符在本征向量基底的矩阵元
- $\mathcal{O}_{xa,yb} = \langle \eta_x, a| \mathcal{O} |\eta_y, b\rangle$：算符在点基底的矩阵元
- $\mathcal{O}_{i,ya} = \langle \xi_i| \mathcal{O} |\eta_y, a\rangle$、$\mathcal{O}_{xa,j} = \langle \eta_x, a| \mathcal{O} |\xi_j\rangle$：混合基底的矩阵元
- $\tilde{S}_{xa,i} = \langle \tilde{\eta}_x, a| S |\xi_i\rangle$：high mode 到 low mode 的传播子矩阵元（带投影）
- $\tilde{S}_{i,xa} = \langle \xi_i| S |\tilde{\eta}_x, a\rangle$：low mode 到 high mode 的传播子矩阵元（带投影）
- $\tilde{S}_{xa,yb} = \langle \tilde{\eta}_x, a| S |\tilde{\eta}_y, b\rangle$：high mode 之间的传播子矩阵元（带投影）

**带 tilde 的传播子矩阵元如何从不带 tilde 的得到：**
**规则：只要 $\eta$ 带 tilde（$\tilde{\eta}$），对应的传播子矩阵元 $S$ 也带 tilde（$\tilde{S}$）。** 记号约定：为便于实现，代码中将 $\tilde{S}$ 统一命名为 `S_highmode`，但本文公式仍使用 $\tilde{S}$ 表示。

由于 $|\tilde{\eta}_x, a\rangle = (I - I_{\text{low}})|\eta_x, a\rangle$，带 tilde 的传播子矩阵元可以通过投影算符作用于不带 tilde 的传播子得到：

$$
\tilde{S}_{xa,yb} = \langle \tilde{\eta}_x, a| S |\tilde{\eta}_y, b\rangle = \langle \eta_x, a| (I - I_{\text{low}}) S (I - I_{\text{low}}) |\eta_y, b\rangle
$$

定义转移矩阵（overlap matrix）：

$$
M_{xi,a} = \langle \eta_x, a| \xi_i\rangle, \quad M_{ix,a}^* = \langle \xi_i| \eta_x, a\rangle = M_{xi,a}^*
$$

即 $M_{xi,a}$ 是从空间点基底到 eigenvector 基底的转移矩阵元，$M_{ix,a}^* = M_{xi,a}^*$ 是其共轭。

展开 $(I - I_{\text{low}})$ 得到：

$$
\tilde{S}_{xa,yb} = S_{xa,yb} - \sum_{i=1}^{N_v} M_{xi,a} \, \tilde{S}_{i,yb} - \sum_{j=1}^{N_v} S_{xa,j} \, M_{jy,b}
\tag{5.1}
$$

其中 $\tilde{S}_{i,yb} = \langle \xi_i| S (I - I_{\text{low}}) |\eta_y, b\rangle$，
$S_{xa,j} = \langle \eta_x, a| S |\xi_j\rangle$，$M_{xi,a} = \langle \eta_x,a|\xi_i\rangle$，$M_{jy,b} = \langle \xi_j|\eta_y,b\rangle$。

类似地，混合项也有：

$$
\tilde{S}_{xa,i} = \langle \tilde{\eta}_x, a| S |\xi_i\rangle = \langle \eta_x, a| (I - I_{\text{low}}) S |\xi_i\rangle = S_{xa,i} - \sum_{j=1}^{N_v} M_{xj,a} S_{j,i}
\tag{5.2}
$$

$$
\tilde{S}_{i,xa} = \langle \xi_i| S |\tilde{\eta}_x, a\rangle = \langle \xi_i| S (I - I_{\text{low}}) |\eta_x, a\rangle = S_{i,xa} - \sum_{j=1}^{N_v} S_{i,j} M_{jx,a}^*
\tag{5.3}
$$

将 $O_{1,\text{low}}$ 与 $O_2$ 按上述四项方式展开，$Tr[O_{1,\text{low}} S O_2 S]$ 分解为四类 $O_2$ 贡献：

$$
Tr[O_{1,\text{low}} S O_2 S]
= Tr[O_{1,\text{low}} S O_{2,\text{low}} S]
\; + Tr[O_{1,\text{low}} S O_{2,\text{high}} S]
\; + Tr[O_{1,\text{low}} S O_{2,\text{(low-high)}} S]
\; + Tr[O_{1,\text{low}} S O_{2,\text{(high-low)}} S]
$$

**各项的展开结果：**

1) $O_2$ 为 low：

$$
Tr[O_{1,\text{low}} S O_{2,\text{low}} S] = \sum_{i,j,k,l=1}^{N_v} (O_1)_{i,j} (O_2)_{k,l} \, S_{j,k} \, S_{l,i}
\tag{6.1}
$$

2) $O_2$ 为 high（点-点）：

$$
Tr[O_{1,\text{low}} S O_{2,\text{high}} S] = \sum_{i,j=1}^{N_v} \sum_{x,y=1}^{M} \sum_{a,b=0}^{2} (O_1)_{i,j} (O_2)_{xa,yb} \, \tilde{S}_{j,xa} \, \tilde{S}_{yb,i}
\tag{6.2}
$$

3) $O_2$ 为混合（low-high）：

$$
Tr[O_{1,\text{low}} S O_{2,\text{(low-high)}} S] = \sum_{i,j,k=1}^{N_v} \sum_{y=1}^{M} \sum_{b=0}^{2} (O_1)_{i,j} (O_2)_{k,yb} \, S_{j,k} \, \tilde{S}_{yb,i}
\tag{6.3}
$$

4) $O_2$ 为混合（high-low）：

$$
Tr[O_{1,\text{low}} S O_{2,\text{(high-low)}} S] = \sum_{i,j=1}^{N_v} \sum_{x=1}^{M} \sum_{a=0}^{2} \sum_{l=1}^{N_v} (O_1)_{i,j} (O_2)_{xa,l} \, \tilde{S}_{j,xa} \, S_{l,i}
\tag{6.4}
$$

**采样情况下的补偿（当使用 $N < M$ 个点时）**：

对于涉及点的 Trace 项，采样版本可以写成"完整求和 + 相同点的额外补偿"形式。以 High-High 项（公式 6.2）为例：

$$
Tr[O_{1,\text{low}} S O_{2,\text{high}} S]_{\text{sampled}} = \frac{M(M-1)}{N(N-1)} \sum_{i,j=1}^{N_v} \sum_{x,y=1}^{N} \sum_{a,b=0}^{2} (O_1)_{i,j} (O_2)_{x_i a, x_j b} \tilde{S}_{j,x_i a} \tilde{S}_{x_j b,i}
+ \left( \frac{M}{N} - \frac{M(M-1)}{N(N-1)} \right) \sum_{i,j=1}^{N_v} \sum_{x=1}^{N} \sum_{a,b=0}^{2} (O_1)_{i,j} (O_2)_{x_i a, x_i b} \tilde{S}_{j,x_i a} \tilde{S}_{x_i b,i}
\tag{6.2-sampled}
$$

对于混合项（公式 6.3），只有一个点 $y$，采样版本为：

$$
Tr[O_{1,\text{low}} S O_{2,\text{(low-high)}} S]_{\text{sampled}} = \frac{M}{N} \sum_{i,j,k=1}^{N_v} \sum_{y=1}^{N} \sum_{b=0}^{2} (O_1)_{i,j} (O_2)_{k,yb} S_{j,k} \tilde{S}_{yb,i}
\tag{6.3-sampled}
$$

类似地，公式 6.4 也只有一个点 $x$，采样版本为：

$$
Tr[O_{1,\text{low}} S O_{2,\text{(high-low)}} S]_{\text{sampled}} = \frac{M}{N} \sum_{i,j=1}^{N_v} \sum_{x=1}^{N} \sum_{a=0}^{2} \sum_{l=1}^{N_v} (O_1)_{i,j} (O_2)_{xa,l} \tilde{S}_{j,xa} S_{l,i}
\tag{6.4-sampled}
$$

**单点采样（公式 6.3, 6.4）**：

- 只有一个点，补偿系数为 $\frac{M}{N}$（直接从无偏估计公式 4.0a 得到）
- 无需区分相同/不同点，因为只有一个采样点

**两点采样（公式 6.2）**：

**情形 1：两个点从同一采样集合中选取（相关采样）**：

- 第一项：对所有点对（包括相同和不同）应用非对角系数 $\frac{M(M-1)}{N(N-1)}$
- 第二项：对相同点对补偿差值 $\frac{M(N-M)}{N(N-1)}$
- 实际计算中，先完成完整缩并（所有点对），然后提取对角项并加权求和
- 当 $N = M$ 时，补偿项消失，回到完整公式

**情形 2：两个点独立选取（独立采样）**：

如果两个点 $x$ 和 $y$ 是从两个独立的采样集合中选取的（例如 $x$ 从 $N_1$ 个点中选，$y$ 从 $N_2$ 个点中选），则补偿系数为各自独立：

$$
Tr[O_{1,\text{low}} S O_{2,\text{high}} S]_{\text{independent}} = \frac{M}{N_1} \frac{M}{N_2} \sum_{i,j=1}^{N_v} \sum_{x=1}^{N_1} \sum_{y=1}^{N_2} \sum_{a,b=0}^{2} (O_1)_{i,j} (O_2)_{x_i a, y_j b} \tilde{S}_{j,x_i a} \tilde{S}_{y_j b,i}
\tag{6.2-independent}
$$

**推导**：

- 点 $x$ 从 $M$ 个点中独立抽取 $N_1$ 个，补偿系数 $\frac{M}{N_1}$
- 点 $y$ 从 $M$ 个点中独立抽取 $N_2$ 个，补偿系数 $\frac{M}{N_2}$
- 由于独立，总补偿系数为 $\frac{M}{N_1} \times \frac{M}{N_2}$
- 此时 $x$ 和 $y$ 不可能相同（来自不同集合），无需对角项补偿

**应用场景**：

- 单点采样：混合项（6.3, 6.4），直接应用 $\frac{M}{N}$
- 相关采样：High-High 项（6.2-sampled），两个点来自同一个采样集合
- 独立采样：High-High 项（6.2-independent），两个点来自不同的采样集合，常见于多时间片或不同算符的采样策略

备注：任一算符 $\mathcal{O}$ 的模式分解包含四项（左右用逗号分隔左右指标）：

$$
\mathcal{O} 
= \sum_{i,j=1}^{N_v} \mathcal{O}_{i,j} |\xi_i\rangle\langle\xi_j|
\; + \sum_{x,a}\sum_{y,b} \mathcal{O}_{xa,yb} |\tilde{\eta}_x,a\rangle\langle\tilde{\eta}_y,b|
\; + \sum_{i}\sum_{y,b} \mathcal{O}_{i,yb} |\xi_i\rangle\langle\tilde{\eta}_y,b|
\; + \sum_{x,a}\sum_{j} \mathcal{O}_{xa,j} |\tilde{\eta}_x,a\rangle\langle\xi_j|.
\tag{6.5}
$$

在上述四个 $Tr[\cdot]$ 的公式中，$O_{1}$ 与 $O_{2}$ 若替换为其完整四项展开，则会得到对应的四类（LL、LH、HL、HH）以及混合（L-\~H、\~H-L）贡献的逐项求和表达。为保持简洁，本文已按目标项（如"Low-High"）收敛到所需的那一类指标求和与相应的 $\tilde{S}$ 组合。

**实际计算时**：需将 $O_2$ 完整展开为四项，对每项分别计算公式 (6.1)-(6.4)，最后求和得到完整结果。

**计算特点：**

- **Low mode 部分**：使用精确的 $\xi_i$ 基底，计算复杂度为 $O(N_v^4)$，这是可管理的
- **High mode 部分**：使用 $\tilde{\eta}_x$ 基底（投影后的空间点），计算复杂度为 $O(M^4) = O(L^{12})$，这在实际中难以直接计算
- **混合项**：需要计算 $\langle \xi_i| S |\tilde{\eta}_x, a\rangle$ 和 $\langle \tilde{\eta}_x, a| S |\xi_i\rangle$ 类型的矩阵元，这些将 $\xi$ 基底和 $\eta$ 基底连接起来

这种分解允许我们：

- 精确处理 low mode（低频/长波长）部分
- 通过投影到空间点处理 high mode（高频/短波长）部分
- 在两者之间建立连接，实现局部混合（localized blending）

