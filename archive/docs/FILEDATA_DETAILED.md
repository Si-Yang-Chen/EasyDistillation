# FileData 模块详细文档

**基于源代码的深度分析** | 最后更新: 2026-06-01

---

## 📋 目录

1. [概览](#概览)
2. [架构设计](#架构设计)
3. [核心设计：File 与 FileData 的关系](#核心设计file-与-filedata-的关系)
4. [抽象基类](#抽象基类)
5. [BinaryFile - 原始二进制格式](#1-binaryfile-binarypy---原始二进制格式)
6. [NdarrayFile - NumPy 格式](#2-ndarrayfile-ndarraypy---numpy-npy-格式)
7. [IldgFile - ILDG 标准格式](#4-ildgfile-ildgpy---ildg-标准格式)
8. [QDPLazyDiskMapObjFile - QDP 时间片格式](#5-qdplazydiskmapobjfile-timeslicepy---qdp-时间片格式)
9. [SliceLoader - 自定义加载器](#6-sliceloader-sliceloaderpy---自定义加载器)
10. [性能优化技术](#-性能优化技术)
11. [类型对比表](#-类型对比表)
12. [使用建议](#-使用建议)

---

## 📋 概览

`lattice/filedata` 模块提供了统一的文件数据访问接口，支持多种格式的格点 QCD 数据文件。所有类型都实现了**惰性加载**和**内存映射**技术，以优化大文件访问性能。

---

## 🏗️ 架构设计

### 核心抽象层次

```
File (抽象文件加载器 - 工厂模式)
    ├── BinaryFile
    ├── NdarrayFile
    ├── NdarrayTimeslicesFile
    ├── IldgFile
    └── QDPLazyDiskMapObjFile
    ↓ get_file_data(name, elem)
    ↓ 创建并缓存
    ↓
FileData (抽象数据访问 - 切片操作)
    ├── BinaryFileData        # 原始二进制
    ├── NdarrayFileData       # NumPy .npy 格式
    ├── NdarrayTimeslicesFileData  # 时间片 .npy 格式
    ├── IldgFileData          # ILDG 标准格式
    └── QDPLazyDiskMapObjFileData  # QDP 时间片格式
```

### 设计模式

1. **抽象工厂模式**: `File` 类负责创建对应的 `FileData` 实例
2. **惰性初始化**: 只在第一次访问时加载元数据
3. **单例缓存**: 同一文件只创建一个 `FileData` 实例
4. **职责分离**: File 管理缓存，FileData 管理数据访问

---

## 📦 详细类型说明

### 0. 核心设计：File 与 FileData 的关系

#### 设计理念

`File` 和 `FileData` 是两个紧密协作的抽象层：

```
用户代码 → File.get_file_data(name, elem) → FileData.__getitem__(key) → 数据
           ↑                                ↑
           工厂模式 + 缓存                  数据访问逻辑
```

**职责分离**：
- **File**: 工厂类，负责创建、缓存和管理 FileData 实例
- **FileData**: 数据访问类，负责实际的文件读取和切片操作

#### File (抽象工厂基类)

```python
class File(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_file_data(self, name: str, elem: FileMetaData) -> FileData:
        """根据文件名和元数据返回 FileData 实例"""
        pass
```

**核心功能**：
1. **工厂方法**: 创建对应的 FileData 实例
2. **缓存管理**: 避免重复解析同一文件的元数据
3. **文件切换**: 处理不同文件之间的切换

#### File 的具体实现

每个 File 子类都实现了相同的缓存模式：

```python
class BinaryFile(File):
    def __init__(self) -> None:
        self.file: str = None          # 缓存的文件名
        self.data: BinaryFileData = None  # 缓存的 FileData 对象

    def get_file_data(self, name: str, elem: FileMetaData) -> BinaryFileData:
        # 缓存机制：只在文件名变化时才重新创建 FileData
        if self.file != name:
            self.file = name
            self.data = BinaryFileData(name, elem)
        return self.data
```

**缓存逻辑详解**：

```python
# 第一次调用：创建新的 FileData
data1 = file.get_file_data("cfg_1000.npy", elem)  # 创建并缓存
# self.file = "cfg_1000.npy"
# self.data = BinaryFileData("cfg_1000.npy", elem)

# 第二次调用相同文件：返回缓存的 FileData
data2 = file.get_file_data("cfg_1000.npy", elem)  # 返回缓存
# self.file == "cfg_1000.npy" → 不重新创建
# data2 is data1  # True，同一对象

# 第三次调用不同文件：重新创建 FileData
data3 = file.get_file_data("cfg_1001.npy", elem)  # 重新创建
# self.file != "cfg_1000.npy" → 重新创建
# self.file = "cfg_1001.npy"
# self.data = BinaryFileData("cfg_1001.npy", elem)
```

**性能优势**：
- ✅ 避免重复解析文件元数据（如 NumPy 文件头、ILDG XML 元数据）
- ✅ 减少文件描述符的打开/关闭操作
- ✅ 保持性能统计数据的连续性（`time_in_sec`, `size_in_byte`）

#### File 与 FileData 的协作流程

```python
# 1. 用户创建 File 实例（通常是通过 Mixin 组合）
class PerambulatorNpy(NdarrayFile, Perambulator):
    def __init__(self, prefix: str, suffix: str, shape, Ne):
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c16", 0), Ne)
        self.prefix = prefix
        self.suffix = suffix

# 2. 用户调用 load 方法（来自 Perambulator Mixin）
peram = PerambulatorNpy("/data/cfg_", ".npy", shape, Ne)
data = peram.load("1000")  # 内部调用 File.get_file_data()

# 3. Perambulator.load() 内部实现
def load(self, key: str):
    name = f"{self.prefix}{key}{self.suffix}"  # 构建文件名
    file_data = self.get_file_data(name, self.elem)  # 调用 File.get_file_data()
    # 这里会触发缓存机制
    return file_data  # 返回 FileData 对象

# 4. 用户访问数据（调用 FileData.__getitem__）
data = peram.load("1000")
t_slice = data[(10, 20)]  # 访问 t_src=10, t_snk=20
```

#### 所有 File 子类的统一模式

| File 子类 | 创建的 FileData 类型 | 文件格式 | 缓存字段 |
|-----------|---------------------|---------|---------|
| `BinaryFile` | `BinaryFileData` | 原始二进制 | `self.file`, `self.data` |
| `NdarrayFile` | `NdarrayFileData` | NumPy .npy | `self.file`, `self.data` |
| `NdarrayTimeslicesFile` | `NdarrayTimeslicesFileData` | 时间片 .npy | `self.file`, `self.data` |
| `IldgFile` | `IldgFileData` | ILDG .lime | `self.file`, `self.data` |
| `QDPLazyDiskMapObjFile` | `QDPLazyDiskMapObjFileData` | QDP .mod | `self.file`, `self.data` |

**重要特性**：所有 File 子类都实现了完全相同的缓存模式，只是创建的 FileData 类型不同。

#### Mixin 组合模式

File 类通常不单独使用，而是通过 Mixin 与物理数据类组合：

```python
# 组合模式：File (文件访问) + Perambulator (物理数据)
class PerambulatorNpy(NdarrayFile, Perambulator):
    # NdarrayFile 提供 get_file_data() 方法
    # Perambulator 提供 load() 方法和物理语义
    pass

# 使用
peram = PerambulatorNpy(prefix, suffix, shape, Ne)
data = peram.load("1000")  # Perambulator.load() → NdarrayFile.get_file_data()
```

**优势**：
- ✅ 分离关注点：文件访问 vs 物理数据
- ✅ 可扩展性：易于添加新的文件格式或数据类型
- ✅ 代码复用：File 类可被多种数据类型共享

#### 最佳实践

**1. 重复访问同一文件时保持性能统计**：
```python
peram = PerambulatorNpy(...)

# 多次访问同一文件
data1 = peram.load("1000")
slice1 = data1[(10,)]

# 文件名相同，返回缓存的 FileData，性能统计累积
data2 = peram.load("1000")
slice2 = data2[(20,)]

print(f"Total time: {data2.time_in_sec:.2f}s")  # 包含所有访问的时间
```

**2. 切换文件时自动更新缓存**：
```python
data_1000 = peram.load("1000")  # 创建新的 FileData
data_1001 = peram.load("1001")  # 替换缓存，创建新的 FileData
# 注意：data_1000 可能已失效（取决于实现）
```

**3. 多个实例独立缓存**：
```python
peram1 = PerambulatorNpy(...)
peram2 = PerambulatorNpy(...)

data1 = peram1.load("1000")  # peram1 的缓存
data2 = peram2.load("1001")  # peram2 的缓存
# 两个实例的缓存独立，互不影响
```

---

### 1. 抽象基类 (abstract.py)

#### FileMetaData
```python
class FileMetaData:
    def __init__(self, shape: List[int], dtype: str = "<c16", extra: Any = None):
        self.shape = shape      # 数据形状
        self.dtype = dtype      # 数据类型，如 '<c16', '>c8'
        self.extra = extra      # 额外参数（用于时间片格式）
```

**关键字段**:
- `shape`: 数据的维度信息
- `dtype`: NumPy 数据类型字符串
  - `<` 小端序，`>` 大端序
  - `c16` = complex128, `c8` = complex64
  - `i` 整数，`f` 浮点数
- `extra`: 用于 `QDPLazyDiskMapObjFile` 的时间片轴数

#### FileData (抽象基类)
```python
class FileData(metaclass=abc.ABCMeta):
    shape = None           # 数据形状
    dtype = None           # 数据类型
    time_in_sec = 0.0      # 累计读取时间（性能统计）
    size_in_byte = 0       # 累计读取字节数（性能统计）

    @abc.abstractmethod
    def __getitem__(self, key: Tuple[int]):
        """子类必须实现切片访问"""
        pass
```

**核心特性**:
- 定义统一的切片访问接口 `__getitem__`
- 内置性能统计功能
- 所有子类必须实现数据访问逻辑

#### File (抽象文件加载器)
```python
class File(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_file_data(self, name: str, elem: FileMetaData) -> FileData:
        """根据文件名和元数据返回 FileData 实例"""
        pass
```

**缓存机制**:
```python
def get_file_data(self, name: str, elem: FileMetaData) -> FileData:
    if self.file != name:  # 文件名变化时才重新加载
        self.file = name
        self.data = ...  # 创建新的 FileData
    return self.data
```

---

### 2. BinaryFile (binary.py) - 原始二进制格式

#### 特性
- **最简单的格式**: 无文件头，纯二进制数据
- **自定义形状**: 用户必须提供 `shape` 和 `dtype`
- **大端序支持**: 典型使用 `>c16` (大端 complex128)

#### BinaryFileData 实现细节

```python
class BinaryFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData) -> None:
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        # 计算步幅（用于切片计算）
        self.stride = [prod(self.shape[i:]) for i in range(1, len(self.shape))] + [1]
        # 解析数据类型字节数
        self.bytes = int(re.match(r"^[<>=]?[iufc](?P<bytes>\d+)$", elem.dtype).group("bytes"))
```

**步幅计算示例**:
```python
# 对于 shape = [Lt, Ns, Ns, Ne, Ne]
# stride = [Ns*Ns*Ne*Ne, Ns*Ne*Ne, Ne*Ne, Ne, 1]
# 这允许快速计算任意切片的字节偏移量
```

#### 内存映射读取
```python
def __getitem__(self, key: Tuple[int]):
    with open(self.file, "rb") as f:
        # 使用 mmap 进行内存映射
        with mmap.mmap(
            f.fileno(),
            int(numpy.prod(self.shape)) * self.bytes,
            access=mmap.ACCESS_READ,
            offset=0
        ) as mm:
            # 创建 NumPy 数组视图（不复制数据）
            file = numpy.ndarray.__new__(
                numpy.memmap,
                shape=tuple(self.shape),
                dtype=self.dtype,
                buffer=mm,
                offset=0
            )
            # 复制切片数据到后端数组
            ret = backend.asarray(file[key].copy())
```

**关键优化**:
1. 使用 `mmap` 系统调用，避免将整个文件加载到内存
2. 只复制需要的切片数据
3. 按页对齐访问（`ALLOCATIONGRANULARITY`）

#### 典型用途
```python
# Chroma/QDPXX 生成的传播子文件
class PerambulatorBinary(BinaryFile, Perambulator):
    def __init__(self, prefix: str, suffix: str,
                 shape: List[int] = [128, 128, 4, 4, 70, 70],
                 totNe: int = 70) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.nev70.peram"  # 典型文件后缀
```

---

### 3. NdarrayFile (ndarray.py) - NumPy .npy 格式

#### 特性
- **标准 NumPy 格式**: 包含文件头和元数据
- **自动解析形状**: 从文件头读取 shape 和 dtype
- **小端序优化**: 通常使用 `<c16` (小端 complex128)

#### NdarrayFileData 实现细节

```python
def __getitem__(self, key: Tuple[int]):
    with open(self.file, "rb") as f:
        # 1. 读取 NumPy 文件头
        N = len(numpy.lib.format.MAGIC_PREFIX) + 2
        magic = f.read(N)
        assert magic[:-2] == numpy.lib.format.MAGIC_PREFIX

        major, minor = magic[-2:]
        version = (major, minor)
        assert version in [(1, 0), (2, 0)]

        # 2. 解析数组头
        shape, fortran_order, dtype = numpy.lib.format._read_array_header(f, version)
        assert not fortran_order  # 只支持 C 顺序

        # 3. 计算数据起始偏移
        self_offset = f.tell()

        # 4. 内存映射读取
        start = self_offset - self_offset % mmap.ALLOCATIONGRANULARITY
        offset = self_offset - start

        with mmap.mmap(
            f.fileno(),
            offset + int(numpy.prod(shape)) * dtype.itemsize,
            access=mmap.ACCESS_READ,
            offset=start
        ) as mm:
            file = numpy.ndarray.__new__(
                numpy.memmap,
                shape=tuple(shape),
                dtype=dtype,
                buffer=mm,
                offset=offset
            )
            ret = backend.asarray(file[key].copy())
```

**与 BinaryFile 的区别**:
1. 自动读取文件头获取 shape 和 dtype
2. 支持验证 fortran_order（必须是 C 顺序）
3. 数据偏移量不是 0，需要从头计算

#### NdarrayTimeslicesFileData - 时间片优化版本

**特殊设计**: 为每个时间片保存为独立文件

```python
class NdarrayTimeslicesFileData(FileData):
    def __getitem__(self, key: Tuple[int]):
        tsrc_idx, *rest = key  # 第一个索引是时间片索引
        key = tuple(rest)

        # 根据时间片索引替换文件名模式
        # 例如: "data.t???.npy" → "data.t000.npy", "data.t001.npy", ...
        import re
        t_file = re.sub(r"\.t\?\?\?\.", f".t{tsrc_idx:03d}.", self.file)

        # 读取对应时间片的文件
        with open(t_file, "rb") as f:
            # ... 同 NdarrayFileData 的读取逻辑
```

**优势**:
- **内存效率**: 只加载需要的时间片，而不是整个文件
- **并行友好**: 不同时间片可以独立处理
- **灵活性**: 可以选择性加载特定时间片

**典型用途**:
```python
# 时间片格式的传播子
class PerambulatorTimeslicesNpy(NdarrayTimeslicesFile, Perambulator):
    def __init__(self, prefix: str, suffix: str,
                 shape: List[int] = [128, 4, 4, 70, 70],
                 totNe: int = 70) -> None:
        super().__init__()
        Perambulator.__init__(self, FileMetaData(shape, "<c16", 0), totNe)
        self.prefix = prefix
        self.suffix = ".t???.npy"  # 时间片文件名模式
```

**文件命名约定**:
```
data.t000.npy  # t_src=0
data.t001.npy  # t_src=1
data.t002.npy  # t_src=2
...
```

---

### 4. IldgFile (ildg.py) - ILDG 标准格式

#### 特性
- **国际格点数据网格标准格式**
- **包含 XML 元数据**: 描述格点尺寸、精度等
- **大端序**: 通常使用 `>c16` 或 `>c8`
- **复杂结构**: 包含多个命名记录

#### ILDG 文件结构
```
[Magic Number: 0x456789AB0001]
[Length] [Header: "ildg-format"]
[XML Metadata]
[Length] [Header: "ildg-binary-data"]
[Binary Data]
...
```

#### IldgFileData 实现细节

```python
class IldgFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData,
                 offset: Tuple[int], xmlTree: ET.ElementTree) -> None:
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        self.offset = offset[0]  # 二进制数据起始位置

        # 从 XML 提取格点尺寸
        tag = re.match(r"\{.*\}", xmlTree.getroot().tag).group(0)
        self.latt_size = [
            int(xmlTree.find(f"{tag}lx").text),
            int(xmlTree.find(f"{tag}ly").text),
            int(xmlTree.find(f"{tag}lz").text),
            int(xmlTree.find(f"{tag}lt").text),
        ]

        # 验证精度匹配
        assert self.bytes == int(xmlTree.find(f"{tag}precision").text) // 8 * 2
        # 验证数据大小匹配
        assert prod(elem.shape) * self.bytes == offset[1]
```

#### XML 元数据解析
```python
def read_meta_data(self, f: BufferedReader):
    obj_pos_size: Dict[str, Tuple[int]] = {}

    # 读取所有记录
    buffer = f.read(8)
    while buffer != b"" and buffer != b"\x0A":
        assert buffer.startswith(b"\x45\x67\x89\xAB\x00\x01")

        # 读取记录长度
        length = (struct.unpack(">Q", f.read(8))[0] + 7) // 8 * 8

        # 读取记录头
        header = f.read(128).strip(b"\x00").decode("utf-8")
        obj_pos_size[header] = (f.tell(), length)

        # 跳到下一个记录
        f.seek(length, SEEK_CUR)
        buffer = f.read(8)

    # 提取 ildg-binary-data 的偏移量
    offset = obj_pos_size["ildg-binary-data"]

    # 解析 XML 元数据
    f.seek(obj_pos_size["ildg-format"][0])
    xml_tree = ET.ElementTree(
        ET.fromstring(
            f.read(obj_pos_size["ildg-format"][1]).strip(b"\x00").decode("utf-8")
        )
    )

    return offset, xml_tree
```

**XML 元数据示例**:
```xml
<ildgFormat>
  <lx>16</lx>
  <ly>16</ly>
  <lz>16</lz>
  <lt>64</lt>
  <precision>64</precision>
</ildgFormat>
```

#### 数据读取
```python
def __getitem__(self, key: Tuple[int]):
    # 注意：ILDG 格式可能需要字节序转换
    ret = backend.asarray(file[key].copy().astype("<c16"))
    #                                     ^^^^^^^^^^^^^^^^
    #                                     转换为小端序 complex128
```

**自动字节序转换**: ILDG 通常是大端序，但代码会自动转换为小端序以便后续处理。

#### 典型用途
```python
# ILDG 格式的规范场
class GaugeFieldIldg(IldgFile, GaugeField):
    def __init__(self, prefix: str, suffix: str,
                 shape: List[int] = [64, 16, 16, 16, 4, 3, 3]) -> None:
        super().__init__()
        GaugeField.__init__(self, FileMetaData(shape, ">c16", 0))
        self.prefix = prefix
        self.suffix = ".lime"  # ILDG 文件通常使用 .lime 扩展名
```

---

### 5. QDPLazyDiskMapObjFile (timeslice.py) - QDP 时间片格式

#### 特性
- **Chroma/QDPXX 软件输出格式**
- **多记录文件**: 一个文件包含多个时间片数据
- **键值映射**: 通过键（如时间片索引）访问不同记录
- **惰性字典式访问**: 类似 Python dict 的访问方式

#### 文件结构
```
[Magic: "XXXXQDPLazyDiskMapObjFileXXXX"]
[Version: 0]
[XML Metadata String]
[File Position Pointer]
[Number of Records: N]
[Key 0] [Position 0]
[Key 1] [Position 1]
...
[Key N-1] [Position N-1]
[Record 0 Data]
[Record 1 Data]
...
[Record N-1 Data]
```

#### QDPLazyDiskMapObjFileData 实现细节

```python
class QDPLazyDiskMapObjFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData,
                 offsets: Dict[Tuple[int], int],
                 xml_tree: ET.ElementTree) -> None:
        self.file = file
        # 注意：shape 是去掉 extra 维度后的形状
        self.shape = elem.shape[elem.extra:]
        self.dtype = elem.dtype
        self.extra = elem.extra  # extra 维度数量（通常是 1）
        self.extraShape = elem.shape[0:elem.extra]  # extra 维度的形状
        self.offsets = offsets  # 键到文件偏移量的映射

        # 从 XML 提取格点信息
        latt_size = [int(x) for x in xml_tree.find("lattSize").text.split(" ")]
        decay_dir = int(xml_tree.find("decay_dir").text)
        assert decay_dir == 3  # 时间方向必须是第 3 维
```

#### 元数据解析
```python
def read_meta_data(self, f: BufferedReader) -> Dict[Tuple[int], int]:
    # 1. 验证魔数
    assert self.magic == read_str(f)

    # 2. 读取版本号
    self.version = struct.unpack(">i", f.read(4))[0]

    # 3. 读取 XML 元数据
    xml_tree = ET.ElementTree(ET.fromstring(read_str(f)))

    # 4. 跳转到记录索引位置
    f.seek(read_pos(f))

    # 5. 读取记录数量
    num_record = struct.unpack(">I", f.read(4))[0]

    # 6. 构建键值偏移映射
    offsets: Dict[Tuple[int], int] = {}
    for _ in range(num_record):
        key = read_tuple(f)      # 读取键（例如 (0,), (1,), ...）
        val = read_pos(f)        # 读取文件偏移量
        offsets[key] = val

    return offsets, xml_tree
```

#### 数据访问
```python
def __getitem__(self, key: Tuple[int]):
    if isinstance(key, int):
        key = (key,)

    # 1. 检查键是否存在
    if key[0:self.extra] not in self.offsets:
        raise IndexError(f"index {key} is out of bounds for axes")

    # 2. 获取该记录的文件偏移量
    self_offset = self.offsets[key[:self.extra]]

    # 3. 内存映射读取
    start = self_offset - self_offset % mmap.ALLOCATIONGRANULARITY
    offset = self_offset - start

    with open(self.file, "rb") as f:
        with mmap.mmap(
            f.fileno(),
            offset + int(numpy.prod(self.shape)) * self.bytes,
            access=mmap.ACCESS_READ,
            offset=start
        ) as mm:
            file = numpy.ndarray.__new__(
                numpy.memmap,
                shape=tuple(self.shape),
                dtype=self.dtype,
                buffer=mm,
                offset=offset
            )
            # 注意：QDP 格式通常需要类型转换
            ret = backend.asarray(file[key[self.extra:]].copy().astype("<c8"))
            #                                                 ^^^^^^^^^^^
            #                                                 转换为小端序 complex64
```

**自动类型转换**: QDP 格式通常是单精度（`complex64`），代码会自动转换。

#### 典型用途
```python
# QDP 时间片格式的本征矢量
class EigenvectorTimeSlice(QDPLazyDiskMapObjFile, Eigenvector):
    def __init__(self, prefix: str, suffix: str,
                 shape: List[int] = [128, 70, 16**3, 3],
                 totNe: int = 70) -> None:
        super().__init__()
        # 注意 extra=2，因为前两个维度 [Lt, Ne] 作为键
        Eigenvector.__init__(self, FileMetaData(shape, ">c8", 2), totNe)
        self.prefix = prefix
        self.suffix = ".stout.n20.f0.12.laplace_eigs.3d.mod"
```

**访问示例**:
```python
eigs = EigenvectorTimeSlice(...)

# 访问时间片 t=10，本征矢量 e=5
data = eigs[(10, 5)]  # 使用元组作为键
data = eigs[10, 5]    # 等价写法

# 访问时间片 t=10 的所有本征矢量
all_eigs_t10 = eigs[(10,)]  # 返回 shape [Ne, L^3, Nc]
```

---

### 6. SliceLoader (sliceloader.py) - 自定义加载器

#### 特性
- **不使用 FileData 体系**: 独立的加载器类
- **优化的切片逻辑**: 自定义实现高效的切片访问
- **支持两种格式**: `binloader` (二进制) 和 `npyloader` (NumPy)

#### binloader - 二进制加载器

```python
class binloader:
    def __init__(self, filename: str, dtype, offset: int = 0, shape: Tuple[int] = None):
        # 验证文件大小和形状
        with open(filename, "rb") as fid:
            fid.seek(0, 2)
            flen = fid.tell()

            if shape is None:
                # 自动推断形状（1D 数组）
                size = (flen - offset) // dtype.itemsize
                shape = (size,)
            else:
                # 验证文件大小足够
                size = numpy.prod(shape)
                bytes = offset + size * dtype.itemsize
                if flen < bytes:
                    raise ValueError("Size of available data is less than acquired")
```

**优化的切片算法**:
```python
def __getitem__(self, item):
    # 1. 解析切片
    shape = self.shape
    stride = ...  # 计算步幅

    # 2. 分析每个维度
    for i, dim in enumerate(item):
        if isinstance(dim, int):
            # 整数索引：跳过该维度
            skip += dim * stride[i]
        elif isinstance(dim, slice):
            # 切片索引：计算范围
            start, stop, step = dim.start, dim.stop, dim.step
            # ...
        else:
            # 列表索引：支持花式索引
            dim_list = list(dim)
            # ...

    # 3. 优化连续读取
    # 识别连续区域，一次性读取多个元素
    rescount = 1
    i = ndim - 1
    while i != 0 and full[i]:
        rescount *= shape[i]
        i -= 1

    # 4. 执行读取
    return self.load(resshape, realitem, realshape, realstride, rescount, resoffset)
```

**关键优化**:
- **连续块检测**: 识别可以一次性读取的连续区域
- **批量读取**: 一次读取多个连续元素，减少 I/O 操作
- **智能跳转**: 根据 step 优化文件指针移动

#### npyloader - NumPy 加载器

```python
class npyloader:
    def __init__(self, filename: str):
        # 验证 .npy 格式
        magic_prefix = numpy.lib.format.MAGIC_PREFIX

        with open(filename, "rb") as fid:
            magic = fid.read(len(magic_prefix))
            if magic != magic_prefix:
                raise ValueError(f"{filename} is not a .npy file.")

            # 读取头信息
            version = numpy.lib.format.read_magic(fid)
            shape, fortran_order, dtype = numpy.lib.format._read_array_header(fid, version)
            offset = fid.tell()

        # 委托给 binloader
        self.loader = binloader(filename, dtype, offset, shape)
```

**设计模式**: `npyloader` 是 `binloader` 的包装器，先解析 NumPy 头，然后使用 `binloader` 的优化切片逻辑。

---

## 🔍 性能优化技术

### 1. 内存映射 (mmap)

所有类型都使用内存映射技术：

```python
# 传统方式：加载整个文件到内存
data = numpy.fromfile(file, dtype)  # 占用全部内存

# 内存映射方式：按需加载
with mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ, offset=start) as mm:
    data = numpy.ndarray(shape, dtype, buffer=mm, offset=offset)
    sliced_data = data[key].copy()  # 只复制需要的部分
```

**优势**:
- 只将访问的数据页加载到内存
- 操作系统自动管理缓存
- 支持超大文件（远超物理内存）

### 2. 页对齐访问

```python
# 内存映射要求偏移量对齐到页边界
start = self_offset - self_offset % mmap.ALLOCATIONGRANULARITY
offset = self_offset - start  # 相对于页起始的偏移

with mmap.mmap(f.fileno(), size, access=mmap.ACCESS_READ, offset=start) as mm:
    # 创建数组时使用相对偏移
    file = numpy.ndarray(shape, dtype, buffer=mm, offset=offset)
```

### 3. 缓存机制

所有 `File` 类都实现了缓存：

```python
class BinaryFile(File):
    def __init__(self):
        self.file: str = None
        self.data: BinaryFileData = None

    def get_file_data(self, name: str, elem: FileMetaData) -> BinaryFileData:
        if self.file != name:  # 只在文件名变化时重新加载
            self.file = name
            self.data = BinaryFileData(name, elem)
        return self.data
```

**效果**: 重复访问同一文件不会重复解析元数据。

### 4. 后端适配

所有读取操作都支持多后端：

```python
from ..backend import get_backend

backend = get_backend()  # NumPy 或 CuPy
ret = backend.asarray(file[key].copy())
```

**优势**: 无需修改代码即可切换 CPU/GPU 后端。

---

## 📊 类型对比表

| 类型 | 文件格式 | 典型扩展名 | 字节序 | 头信息 | 时间片支持 | 使用场景 |
|------|---------|-----------|--------|--------|----------|---------|
| **BinaryFile** | 原始二进制 | `.peram`, `.bin` | 可配置 | ❌ 无 | ❌ | Chroma 输出 |
| **NdarrayFile** | NumPy | `.npy` | 小端 | ✅ 自动 | ❌ | EasyDistillation 输出 |
| **NdarrayTimeslicesFile** | NumPy 时间片 | `.t???.npy` | 小端 | ✅ 自动 | ✅ 多文件 | 大数据优化 |
| **IldgFile** | ILDG | `.lime` | 大端 | ✅ XML | ❌ | 规范场标准格式 |
| **QDPLazyDiskMapObjFile** | QDP 时间片 | `.mod` | 大端 | ✅ XML + 索引 | ✅ 单文件多记录 | 本征矢量、规范场 |

---

## 🎯 使用建议

### 选择合适的类型

```python
# ✅ 推荐使用时间片格式（内存效率高）
peram = PerambulatorTimeslicesNpy(prefix, suffix, shape, Ne)

# ⚠️ 避免使用单文件格式（内存占用大）
peram = PerambulatorNpy(prefix, suffix, shape, Ne)  # 可能 91GB+

# ✅ 使用 ILDG 格式加载规范场（标准格式）
gauge = GaugeFieldIldg(prefix, suffix, shape)

# ✅ 使用 QDP 时间片格式加载本征矢量（Chroma 输出）
eigs = EigenvectorTimeSlice(prefix, suffix, shape, Ne)
```

### 性能优化技巧

```python
# 1. 批量访问连续区域
data = peram[t_snk, :, :, :, :, :]  # ✅ 一次读取完整切片

# 避免多次小切片
for i in range(Ne):
    data_i = peram[t_snk, :, :, :, i, i]  # ❌ 多次小读取

# 2. 及时释放大对象
data = large_file[:]
# ... 处理 ...
del data  # 释放内存

# 3. 使用性能统计
print(f"Read time: {file_data.time_in_sec:.2f}s")
print(f"Data size: {file_data.size_in_byte / 1024**3:.2f} GB")
```

---

## 🔧 扩展开发

### 添加新的文件格式

```python
# 1. 创建 FileData 子类
class MyCustomFileData(FileData):
    def __init__(self, file: str, elem: FileMetaData, custom_param):
        super().__init__()
        self.file = file
        self.shape = elem.shape
        self.dtype = elem.dtype
        # ... 初始化逻辑

    def __getitem__(self, key: Tuple[int]):
        # 实现切片访问逻辑
        # 建议使用 mmap 进行内存映射
        return data

# 2. 创建 File 子类
class MyCustomFile(File):
    def __init__(self):
        self.file: str = None
        self.data: MyCustomFileData = None

    def get_file_data(self, name: str, elem: FileMetaData) -> MyCustomFileData:
        if self.file != name:
            self.file = name
            # 解析自定义格式
            self.data = MyCustomFileData(name, elem, custom_param)
        return self.data

# 3. 组合使用
class MyDataNpy(MyCustomFile, MyData):
    def __init__(self, prefix: str, suffix: str, param: int):
        super().__init__()
        MyData.__init__(self, FileMetaData([...], "<c16", 0), param)
        self.prefix = prefix
        self.suffix = suffix
```

---

## 📝 注意事项

### 常见陷阱

1. **字节序问题**
   ```python
   # ILDG 和 QDP 通常是大数据端
   dtype = ">c16"  # 大端 complex128

   # NumPy 通常是小数据端
   dtype = "<c16"  # 小端 complex128

   # 注意自动转换
   data = file[key].astype("<c16")  # 确保统一为小端序
   ```

2. **时间片索引**
   ```python
   # NdarrayTimeslicesFile: 使用文件名模式
   file.t000.npy, file.t001.npy, ...

   # QDPLazyDiskMapObjFile: 使用键值映射
   data = file[(10, 5)]  # 元组作为键
   ```

3. **内存映射限制**
   ```python
   # 32 位系统有 2GB 限制
   # 建议使用 64 位 Python

   # 大文件访问可能需要增加虚拟内存
   ```

4. **文件锁定**
   ```python
   # 内存映射期间文件被锁定
   # 确保在写入前关闭所有读取操作
   ```

---

## 📚 参考资料

- [NumPy 内存映射文档](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [ILDG 格式规范](http://ildg.sasr.edu.au/)
- [QDP++ 文档](https://usqcd-software.github.io/qdp++/)
- [Python mmap 文档](https://docs.python.org/3/library/mmap.html)

---

**文档版本**: 1.0 | **基于源代码**: Git commit b4c7933
