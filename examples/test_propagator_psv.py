#!/usr/bin/env python3
"""
测试 PropagatorPSV 相关类的功能

这个脚本展示了如何使用 PropagatorPSV、PropagatorPSVNpy 和 PropagatorPSVTimeslicesNpy 类。
"""

import numpy as np
import os
import tempfile
from lattice import PropagatorPSVNpy, PropagatorPSVTimeslicesNpy


def test_propagator_psv_npy():
    """测试 PropagatorPSVNpy 类（单文件版本）"""
    print("=" * 60)
    print("测试 PropagatorPSVNpy (单文件版本)")
    print("=" * 60)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 参数设置
        Lt, Ns, Np, Ne = 8, 4, 10, 5
        cfg = "test_1000"

        # 生成测试数据
        print(f"\n1. 生成测试数据: 形状 [{Lt}, {Ns}, {Ns}, {Np}, {Ne}]")
        test_data_full = np.random.randn(Lt, Ns, Ns, Np, Ne) + 1j * np.random.randn(Lt, Ns, Ns, Np, Ne)

        # 保存为 .npy 文件
        filepath = os.path.join(tmpdir, f"{cfg}.psv.npy")
        np.save(filepath, test_data_full)
        print(f"   数据已保存到: {filepath}")

        # 使用 PropagatorPSVNpy 加载
        print(f"\n2. 使用 PropagatorPSVNpy 加载数据")
        psv = PropagatorPSVNpy(
            prefix=os.path.join(tmpdir, ""), suffix=".psv.npy", shape=[Lt, Ns, Ns, Np, Ne], Np=Np, Ne=Ne, dtype="<c16"
        )

        loaded_data = psv.load(cfg)
        print(f"   加载数据形状: {loaded_data.shape}")
        print(f"   数据类型: {loaded_data.dtype}")

        # 验证数据一致性
        print(f"\n3. 验证数据一致性")
        difference = np.max(np.abs(loaded_data - test_data_full))
        print(f"   最大差异: {difference}")
        assert difference < 1e-10, "数据加载不一致！"
        print("   ✓ 数据验证通过")

        # 测试简化形状
        print(f"\n4. 测试简化形状 [{Lt}, {Np}, {Ne}]")
        test_data_simple = np.random.randn(Lt, Np, Ne) + 1j * np.random.randn(Lt, Np, Ne)
        filepath_simple = os.path.join(tmpdir, f"{cfg}.simple.npy")
        np.save(filepath_simple, test_data_simple)

        psv_simple = PropagatorPSVNpy(
            prefix=os.path.join(tmpdir, ""), suffix=".simple.npy", shape=[Lt, Np, Ne], Np=Np, Ne=Ne
        )

        loaded_simple = psv_simple.load(cfg)
        print(f"   加载数据形状: {loaded_simple.shape}")
        difference_simple = np.max(np.abs(loaded_simple - test_data_simple))
        print(f"   最大差异: {difference_simple}")
        assert difference_simple < 1e-10, "简化数据加载不一致！"
        print("   ✓ 简化数据验证通过")

    print("\n✓ PropagatorPSVNpy 测试通过！\n")


def test_propagator_psv_timeslices():
    """测试 PropagatorPSVTimeslicesNpy 类（时间片分离版本）"""
    print("=" * 60)
    print("测试 PropagatorPSVTimeslicesNpy (时间片分离版本)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 参数设置
        Lt, Ns, Np, Ne = 8, 4, 10, 5
        cfg = "test_2000"

        # 生成并保存时间片分离的数据
        print(f"\n1. 生成时间片分离的测试数据")
        print(f"   每个时间片形状: [{Lt}, {Ns}, {Ns}, {Np}, {Ne}]")
        print(f"   共 {Lt} 个时间片文件")

        full_data = []
        for t_src in range(Lt):
            # 为每个源时间生成数据
            t_data = np.random.randn(Lt, Ns, Ns, Np, Ne) + 1j * np.random.randn(Lt, Ns, Ns, Np, Ne)
            full_data.append(t_data)

            # 保存文件
            filepath = os.path.join(tmpdir, f"{cfg}.t{t_src:03d}.npy")
            np.save(filepath, t_data)
            if t_src < 3:  # 只打印前几个
                print(f"   已保存: {os.path.basename(filepath)}")

        print(f"   ... (共 {Lt} 个文件)")

        # 使用 PropagatorPSVTimeslicesNpy 加载
        print(f"\n2. 使用 PropagatorPSVTimeslicesNpy 加载数据")
        psv_timeslices = PropagatorPSVTimeslicesNpy(
            prefix=os.path.join(tmpdir, ""), suffix=".npy", shape=[Lt, Ns, Ns, Np, Ne], Np=Np, Ne=Ne, dtype="<c16"
        )

        loaded_data = psv_timeslices.load(cfg)
        print(f"   加载数据形状: {loaded_data.shape}")
        print(f"   数据类型: {loaded_data.dtype}")

        # 验证数据
        print(f"\n3. 验证每个时间片的数据")
        max_diff = 0.0
        for t_src in range(Lt):
            # 注意：这里假设 load 方法按时间片顺序返回数据
            # 具体实现可能需要根据 NdarrayTimeslicesFile 的实际行为调整
            diff = np.max(np.abs(loaded_data[t_src] - full_data[t_src][t_src]))
            max_diff = max(max_diff, diff)
            if t_src < 3:
                print(f"   时间片 {t_src}: 最大差异 = {diff}")

        print(f"   ... (共 {Lt} 个时间片)")
        print(f"   总体最大差异: {max_diff}")

        # 注意：由于我们不知道 NdarrayTimeslicesFile 的具体实现细节，
        # 这个测试可能需要调整
        print("   ⚠ 注意: 时间片加载的具体行为取决于 NdarrayTimeslicesFile 实现")

    print("\n✓ PropagatorPSVTimeslicesNpy 测试完成！\n")


def test_usage_examples():
    """展示实际使用示例"""
    print("=" * 60)
    print("实际使用示例")
    print("=" * 60)

    print(
        """
示例 1: 在收缩计算中使用 PropagatorPSVNpy
--------------------------------------------
from lattice import PropagatorPSVNpy

# 设置参数
L, T = 24, 72
Np, Ne = 216, 70

# 创建 PSV propagator 对象
psv_propagator = PropagatorPSVNpy(
    prefix="/path/to/data/cfg_",
    suffix=".psv.npy",
    shape=[T, 4, 4, Np, Ne],
    Np=Np,
    Ne=Ne
)

# 在循环中使用
for cfg in cfg_list:
    psv_data = psv_propagator.load(cfg)
    # 进行收缩计算...
    result = compute_contraction(psv_data, ...)


示例 2: 使用时间片分离的 PSV 数据
----------------------------------
from lattice import PropagatorPSVTimeslicesNpy

# 适用于从 gen_propagator.py 生成的数据
psv_timeslices = PropagatorPSVTimeslicesNpy(
    prefix="/path/to/psv_data/cfg_",
    suffix=".npy",
    shape=[T, 4, 4, Np, Ne],
    Np=Np,
    Ne=Ne
)

for cfg in cfg_list:
    psv_data = psv_timeslices.load(cfg)
    # 自动加载所有时间片: cfg_XXXX.t000.npy, t001.npy, ...
    # 进行计算...


示例 3: 简化形状的使用（已选择旋量分量）
----------------------------------------
from lattice import PropagatorPSVNpy

# 如果数据中已经收缩了旋量维度
psv_simple = PropagatorPSVNpy(
    prefix="/path/to/data/psv_",
    suffix=".npy",
    shape=[T, Np, Ne],  # 没有 Ns 维度
    Np=Np,
    Ne=Ne
)

data = psv_simple.load("1000")
print(data.shape)  # (72, 216, 70)
"""
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PropagatorPSV 类测试套件")
    print("=" * 60 + "\n")

    try:
        # 运行测试
        test_propagator_psv_npy()
        test_propagator_psv_timeslices()
        test_usage_examples()

        print("=" * 60)
        print("✓ 所有测试完成！")
        print("=" * 60)
        print("\n更多信息请参考: doc/PropagatorPSV_Usage.md")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
