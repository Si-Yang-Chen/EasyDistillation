# unify_vertex_point_color_indices 方法说明

## 概述

`unify_vertex_point_color_indices` 是 `QuarkDiagram` 类的一个方法，用于统一设置某些 vertex 的 point 和 color subscripts。

## 方法签名

```python
def unify_vertex_point_color_indices(
    self, constraints: List[tuple[int | None, int | None]]
) -> None
```

## 输入格式

参数 `constraints` 是一个 tuple 列表，格式如下：

- 每个 tuple 的**索引位置**对应 vertex 的索引
- 每个 tuple 的**第一个元素**代表 left
- 每个 tuple 的**第二个元素**代表 right
- **相同数字**出现时代表设置成一样的（使用相同的 point 和 color indices）
- **None** 表示该位置是 vector（eigenvector），不需要/不能设置 point/color

## 示例解析

对于约束 `[(0,1),(1,None),(None,2)]`：

### Tuple 0: `(0,1)` - 对应 vertex 0
- `0` 在 left 位置 → 代表 vertex 0 的 **left**
- `1` 在 right 位置 → 代表 vertex 0 的 **right**

### Tuple 1: `(1,None)` - 对应 vertex 1
- `1` 在 left 位置 → 代表 vertex 1 的 **left**
- `None` 在 right 位置 → vertex 1 的 right 是 eigenvector

### Tuple 2: `(None,2)` - 对应 vertex 2
- `None` 在 left 位置 → vertex 2 的 left 是 eigenvector
- `2` 在 right 位置 → 代表 vertex 2 的 **right**

### 等价类分析

- 等价类 `{0}`: 仅包含 vertex 0 的 left
- 等价类 `{1}`: 包含 vertex 0 的 right **和** vertex 1 的 left（这两个位置会使用相同的 point/color indices）
- 等价类 `{2}`: 仅包含 vertex 2 的 right

## Vertex 类型限制

不同 vertex 类型的 left/right 端有不同的限制：

| Vertex Type | Left 端 | Right 端 |
|-------------|---------|----------|
| **V2V** | Eigenvector (必须为 None) | Eigenvector (必须为 None) |
| **V2P** | Eigenvector (必须为 None) | Point (可以设置数字) |
| **P2V** | Point (可以设置数字) | Eigenvector (必须为 None) |
| **P2P** | Point (可以设置数字) | Point (可以设置数字) |

**重要**：如果为 eigenvector 位置设置了数字（非 None），方法会抛出 `ValueError`。

## 使用示例

```python
from lattice.quark_diagram import QuarkDiagram

# 创建一个包含 current vertices 的 diagram
adjacency_matrix = [[0, 1], [1, 0]]
vertex_list = [0, 1]  # 两个都是 current vertices

diagram = QuarkDiagram(adjacency_matrix, vertex_list=vertex_list, debug=True)

# 检查是否有 expanded_diagrams
if hasattr(diagram, "expanded_diagrams") and diagram.expanded_diagrams:
    # 使用其中一个 expanded diagram
    test_diagram = diagram.expanded_diagrams[3]  # 例如 P2P 类型
    
    # 定义约束
    # vertex 0: left=0, right=1
    # vertex 1: left=1, right=None
    # 这意味着 vertex 0 的 right 和 vertex 1 的 left 使用相同的 indices
    constraints = [(0, 1), (1, None)]
    
    # 应用统一
    test_diagram.unify_vertex_point_color_indices(constraints)
```

## 工作原理

1. **构建等价类**：遍历所有约束，将相同数字的位置归为一组
2. **验证约束**：检查是否为 eigenvector 位置设置了数字（会抛出错误）
3. **遍历 contraction groups**：对于每个等价类，找到对应的 vertex subscripts
4. **提取和替换**：从第一个非空位置提取 point/color indices，然后应用到等价类中的所有位置
5. **处理 expanded_diagrams**：递归处理所有展开的 diagrams

## 注意事项

- 方法会修改 `self.subscripts`
- 如果存在 `expanded_diagrams`，会递归调用所有展开的 diagrams
- 使用 `debug=True` 创建 QuarkDiagram 可以看到详细的处理信息
- Point indices 使用字符集: `pqrstuv`
- Color indices 使用字符集: `wxyz`

