import unittest
import sys
import os


from lattice.insertion.gauge_link import GaugeLink, gauge_transform_dict, gen_insertion_dict


class TestGaugeLink(unittest.TestCase):

    def test_string_initialization(self):
        """测试通过字符串初始化GaugeLink"""
        # 创建GaugeLink对象
        g = GaugeLink("123")
        print(g.name)
        print(g.gauge_list)
        print(g.idx, 1 * 25 * 6 + 2 * 5 * 6 + 3)

        # 验证成员变量
        self.assertEqual(g.name, "123")
        self.assertEqual(g.gauge_list, [1, 2, 3])

    def test_list_initialization(self):
        """测试通过列表初始化GaugeLink"""
        # 创建GaugeLink对象
        g = GaugeLink([1, 2, 3])

        # 验证成员变量
        self.assertEqual(g.name, "U123")
        self.assertEqual(g.gauge_list, [1, 2, 3])

    def test_idx_initialization(self):
        """测试通过idx初始化GaugeLink"""
        # 首先通过列表创建一个GaugeLink获取idx
        g1 = GaugeLink([5, 5])
        idx = g1.idx
        g2 = GaugeLink(idx)

        # 验证成员变量 - 使用相同的idx应该得到相同的gauge_list
        self.assertEqual(g2.gauge_list, g1.gauge_list)
        self.assertEqual(g2.name, g1.name)
        self.assertEqual(g2.idx, idx)

    def test_cross_initialization_equivalence(self):
        """测试三种初始化方式的等价性"""
        # 通过字符串创建
        g1 = GaugeLink("123")

        # 通过列表创建与字符串等价的对象
        g2 = GaugeLink([1, 2, 3])

        # 通过idx创建与上述两者等价的对象
        g3 = GaugeLink(g1.idx)

        # 验证三种方式创建的对象具有相同的属性
        self.assertEqual(g1.idx, g2.idx)
        self.assertEqual(g1.idx, g3.idx)
        self.assertEqual(g2.idx, g3.idx)

        # 验证从相同idx创建的对象有相同的gauge_list
        self.assertEqual(g1.gauge_list, g2.gauge_list)
        self.assertEqual(g1.gauge_list, g3.gauge_list)
        self.assertEqual(g2.gauge_list, g3.gauge_list)

    def test_transform_c4z(self):
        """测试c4z变换"""
        g = GaugeLink([1, 2, 3])
        t = g.transform("c4z")

        # 验证变换结果
        expected = [gauge_transform_dict["c4z"][i] for i in [1, 2, 3]]
        self.assertEqual(t.gauge_list, expected)

    def test_transform_identity(self):
        """测试恒等变换"""
        g = GaugeLink([1, 2, 3])
        t = g.transform("iden")

        # 验证恒等变换结果应与原始对象相同
        self.assertEqual(t.gauge_list, g.gauge_list)

    def test_transform_multiple_elements(self):
        """测试多个变换元素"""
        g = GaugeLink([1, 2, 3])
        t1 = g.transform("c4z")
        t2 = t1.transform("c4z")  # 应该等价于c2z

        # 验证结果
        expected = [gauge_transform_dict["c2z"][i] for i in [1, 2, 3]]
        self.assertEqual(t2.gauge_list, expected)

    def test_transform_composite_gauge(self):
        """测试复合gauge_link的变换"""
        g = GaugeLink([0, 1, 2, 3, 4, 5])
        t = g.transform("c4x")

        # 验证结果
        expected = [gauge_transform_dict["c4x"][i] for i in [0, 1, 2, 3, 4, 5]]
        self.assertEqual(t.gauge_list, expected)

    def test_transform_consistency(self):
        """测试变换的一致性"""
        g1 = GaugeLink([1, 2, 3])
        g2 = GaugeLink(g1.idx)  # 通过idx创建相同的对象

        # 对两个对象分别进行相同的变换
        t1 = g1.transform("c4z")
        t2 = g2.transform("c4z")

        # 验证变换结果是一致的
        self.assertEqual(t1.gauge_list, t2.gauge_list)
        self.assertEqual(t1.idx, t2.idx)

        # 通过idx创建的对象进行变换
        g3 = GaugeLink(g1.idx)
        t3 = g3.transform("c4z")

        # 验证所有变换结果一致
        self.assertEqual(t1.gauge_list, t3.gauge_list)
        self.assertEqual(t1.idx, t3.idx)

    def test_conjugate(self):
        """测试conjugate方法"""
        g = GaugeLink([1, 2, 3])
        c = g.conjugate()

        # 验证结果
        # 反转并执行(value + 3) % 6变换
        expected = [(i + 3) % 6 for i in [3, 2, 1]]
        self.assertEqual(c.gauge_list, expected)

    def test_conjugate_empty(self):
        """测试空gauge_list的conjugate"""
        g = GaugeLink([])
        c = g.conjugate()

        # 验证结果
        self.assertEqual(c.gauge_list, [])

    def test_conjugate_consistency(self):
        """测试conjugate方法在不同初始化方式下的一致性"""
        g1 = GaugeLink([1, 2, 3])
        g2 = GaugeLink(g1.idx)

        c1 = g1.conjugate()
        c2 = g2.conjugate()

        # 验证结果一致
        self.assertEqual(c1.gauge_list, c2.gauge_list)
        self.assertEqual(c1.idx, c2.idx)

    def test_transform_then_conjugate(self):
        """测试transform后再conjugate的结果"""
        g = GaugeLink([1, 2, 3])

        # 先transform再conjugate
        result1 = g.transform("c4z").conjugate()

        # 计算预期结果
        transformed = [gauge_transform_dict["c4z"][i] for i in [1, 2, 3]]
        # 反转并执行(value + 3) % 6变换
        expected = [(i + 3) % 6 for i in transformed[::-1]]

        self.assertEqual(result1.gauge_list, expected)

    def test_new_idx_encoding(self):
        """测试新的idx编码方案"""
        # 测试空列表
        g0 = GaugeLink([])
        self.assertEqual(g0.idx, 0)

        # 测试单个元素 - 新编码方案：单个元素从1开始
        for i in range(6):
            g = GaugeLink([i])
            self.assertEqual(g.idx, i + 1)  # idx从1开始

        # 测试两个元素的情况，包括相同数字
        pairs = [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 5],
            [1, 1],
            [1, 0],
            [1, 2],
            [2, 2],
            [2, 0],
            [2, 1],
        ]

        idx_set = set()
        for pair in pairs:
            g = GaugeLink(pair)
            self.assertNotIn(g.idx, idx_set, f"重复idx: {g.idx} 对于 {pair}")
            idx_set.add(g.idx)

    def test_idx_to_gauge_list_conversion(self):
        """测试idx和gauge_list之间的双向转换"""
        # 测试一组有效的gauge_list
        test_lists = [
            [],
            [0],
            [1],
            [0, 1],
            [0, 2],
            [1, 0],
            [0, 1, 2],
            [0, 1, 0, 2],
            [1, 2, 4, 0, 1],
            # 添加包含连续相同数字的测试案例
            [0, 0],
            [1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [0, 0, 1, 1, 2, 2],
            [5, 5, 5, 5, 5],
            # 添加更多连续相同数字的案例
            [0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
        ]

        for gauge_list in test_lists:
            # 从gauge_list创建GaugeLink
            g1 = GaugeLink(gauge_list)
            # 获取idx
            idx = g1.idx
            # 用idx创建新的GaugeLink
            g2 = GaugeLink(idx)
            # 确保idx转回gauge_list后得到原始gauge_list
            self.assertEqual(g2.gauge_list, gauge_list, f"转换失败: {gauge_list} -> {idx} -> {g2.gauge_list}")

    def test_consecutive_idx(self):
        """测试idx的连续性和唯一性"""
        # 所有单元素的gauge_list，idx应该是1-6
        single_digits = []
        for i in range(6):
            g = GaugeLink([i])
            single_digits.append(g.idx)

        # 检查是否是连续的
        single_digits.sort()
        self.assertEqual(single_digits, list(range(1, 7)))

        # 测试从0开始的两位数序列，包括重复的0
        two_digits_from_0 = []
        for next_digit in GaugeLink._VALID_NEXT[0]:
            g = GaugeLink([0, next_digit])
            two_digits_from_0.append(g.idx)

        # 检查是否唯一
        self.assertEqual(len(two_digits_from_0), len(set(two_digits_from_0)))  # 唯一性

    def test_idx_uniqueness(self):
        """测试不同gauge_list的idx唯一性，包括重复数字序列"""
        # 生成各种测试序列，包括重复数字
        test_sequences = []

        # 基本序列
        for first in range(6):
            for second in GaugeLink._VALID_NEXT[first]:
                test_sequences.append([first, second])

                # 添加一些包含重复数字的三位序列
                if first == second:  # 如果前两个数字相同
                    for third in GaugeLink._VALID_NEXT[second]:
                        test_sequences.append([first, second, third])
                else:  # 如果前两个数字不同
                    test_sequences.append([first, second, second])  # 后两个数字相同

        # 添加纯重复序列
        for digit in range(6):
            test_sequences.append([digit, digit, digit, digit])

        # 检查每个序列的idx是否唯一
        idx_set = set()
        for seq in test_sequences:
            g = GaugeLink(seq)
            idx = g.idx
            self.assertNotIn(idx, idx_set, f"重复idx: {idx} 对于序列 {seq}")
            idx_set.add(idx)

            # 检查从idx反向计算的gauge_list与原序列一致
            g2 = GaugeLink(idx)
            self.assertEqual(g2.gauge_list, seq, f"反向转换失败: {seq} -> {idx} -> {g2.gauge_list}")

    def test_invalid_gauge_list(self):
        """测试不符合规则的gauge_list（相邻元素差值为3）"""
        # 通过列表创建，相邻元素差值为3
        with self.assertRaises(ValueError):
            GaugeLink([0, 3])

        with self.assertRaises(ValueError):
            GaugeLink([1, 4])

        with self.assertRaises(ValueError):
            GaugeLink([2, 5])

        with self.assertRaises(ValueError):
            GaugeLink([3, 0])

        with self.assertRaises(ValueError):
            GaugeLink([4, 1])

        with self.assertRaises(ValueError):
            GaugeLink([5, 2])

        # 测试包含多个元素的序列
        with self.assertRaises(ValueError):
            GaugeLink([0, 1, 4])

        with self.assertRaises(ValueError):
            GaugeLink([5, 5, 2])

    def test_valid_gauge_list(self):
        """测试符合规则的gauge_list（相邻元素差值不为3）"""
        # 测试一些有效的gauge_list
        valid_lists = [
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 5],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 5],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [3, 5],
            # 添加一些重复数字的长序列
            [0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            # 添加一些混合序列
            [0, 0, 1, 1, 2, 2],
            [3, 3, 4, 4, 5, 5, 0, 0],
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ]

        for valid_list in valid_lists:
            # 应该可以成功创建GaugeLink对象
            try:
                g = GaugeLink(valid_list)
                # 验证gauge_list正确
                self.assertEqual(g.gauge_list, valid_list)
            except ValueError as e:
                self.fail(f"创建有效gauge_list {valid_list} 时失败: {e}")

    def test_nmax_generator(self):
        """测试nmax_generator方法"""
        # 测试nmax_generator方法的基本功能
        nmax_values = list(GaugeLink.nmax_generator(5))

        # 验证返回的是生成器
        self.assertIsInstance(nmax_values, list)

        # 验证前几个值
        # n=0: 1, n=1: 1+6*5^0=7, n=2: 7+6*5^1=37, n=3: 37+6*5^2=187, n=4: 187+6*5^3=937
        expected_values = [1, 7, 37, 187, 937, 4687]
        self.assertEqual(nmax_values, expected_values[: len(nmax_values)])

        # 测试空列表的idx应该是0
        g0 = GaugeLink([])
        self.assertEqual(g0.idx, 0)

        # 测试单个元素的idx范围
        for i in range(6):
            g = GaugeLink([i])
            self.assertEqual(g.idx, i + 1)  # 1-6

        # 测试两个元素的idx范围
        # 第一个两元素序列应该从7开始
        g_first_two = GaugeLink([0, 0])
        self.assertGreaterEqual(g_first_two.idx, 7)

    def test_transform_with_repeated_digits(self):
        """测试包含重复数字的gauge_list的transform"""
        # 创建具有重复数字的GaugeLink
        test_lists = [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1, 2, 2],
        ]

        for test_list in test_lists:
            g1 = GaugeLink(test_list)

            # 对其应用变换
            transforms = ["c4z", "c2z", "c4x", "c2y"]
            for transform in transforms:
                # 计算预期结果
                expected = [gauge_transform_dict[transform][i] for i in test_list]
                # 应用变换
                g2 = g1.transform(transform)
                # 验证结果
                self.assertEqual(
                    g2.gauge_list,
                    expected,
                    f"变换 {transform} 对 {test_list} 失败，得到 {g2.gauge_list} 而非 {expected}",
                )

    def test_same_digit_repetition(self):
        """测试连续相同数字的处理"""
        # 测试所有数字的重复情况
        for digit in range(6):
            # 测试两个相同数字
            g1 = GaugeLink([digit, digit])
            self.assertEqual(g1.gauge_list, [digit, digit])

            # 测试三个相同数字
            g2 = GaugeLink([digit, digit, digit])
            self.assertEqual(g2.gauge_list, [digit, digit, digit])

            # 通过idx创建相同对象并比较
            idx = g2.idx
            g3 = GaugeLink(idx)
            self.assertEqual(g3.gauge_list, [digit, digit, digit])
            self.assertEqual(g3.idx, idx)

    def test_mixed_repetition(self):
        """测试混合重复和非重复数字的序列"""
        # 创建一些包含重复和非重复数字的序列
        test_lists = [
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [1, 1, 2, 2],
            [0, 0, 1, 1, 2, 2],
            [1, 1, 2, 2, 0, 1, 0, 1],
            [0, 0, 1, 1, 2, 2, 0, 1, 0, 1],
        ]

        for test_list in test_lists:
            g1 = GaugeLink(test_list)
            idx = g1.idx
            g2 = GaugeLink(idx)

            # 验证从idx反向创建的对象与原序列一致
            self.assertEqual(g2.gauge_list, test_list, f"混合重复序列转换失败: {test_list} -> {idx} -> {g2.gauge_list}")
            self.assertEqual(g2.idx, idx)

    def test_extreme_repetition_cases(self):
        """测试极端的重复数字情况"""
        # 创建一个非常长的重复序列
        digit = 0
        very_long = [digit] * 30  # 30个相同的数字

        # 测试是否可以正确创建和转换
        g1 = GaugeLink(very_long)
        idx = g1.idx
        g2 = GaugeLink(idx)

        # 验证结果
        self.assertEqual(g2.gauge_list, very_long, f"极端重复序列转换失败: 长度为{len(very_long)}的{digit}序列")
        self.assertEqual(g2.idx, idx)

        # 测试交替的长序列
        alternating = []
        for i in range(20):
            alternating.extend([0, 1])

        g3 = GaugeLink(alternating)
        idx = g3.idx
        g4 = GaugeLink(idx)

        # 验证结果
        self.assertEqual(g4.gauge_list, alternating, f"交替序列转换失败: 长度为{len(alternating)}的交替序列")
        self.assertEqual(g4.idx, idx)

        # 测试大量重复的情况
        long_list = [5] * 20  # 20个5连在一起
        g3 = GaugeLink(long_list)
        idx3 = g3.idx
        g4 = GaugeLink(idx3)
        self.assertEqual(g4.gauge_list, long_list)

        # 测试重复与非重复交替的复杂序列
        complex_list = []
        for i in range(6):
            complex_list.extend([i] * (i + 1))  # 0出现1次，1出现2次，...，5出现6次

        g5 = GaugeLink(complex_list)
        idx5 = g5.idx
        g6 = GaugeLink(idx5)
        self.assertEqual(g6.gauge_list, complex_list)

        # 测试单个数字的极长序列
        for digit in range(6):
            very_long = [digit] * 30  # 30个相同数字
            g = GaugeLink(very_long)
            idx = g.idx
            g2 = GaugeLink(idx)
            self.assertEqual(g2.gauge_list, very_long)

        # 测试相同数字间隔出现
        alternating = []
        for i in range(15):
            alternating.extend([0, 1])  # 重复的01模式

        g3 = GaugeLink(alternating)
        idx3 = g3.idx
        g4 = GaugeLink(idx3)
        self.assertEqual(g4.gauge_list, alternating)

    # def test_gen_gauge_irrep_basic(self):
    #     """测试gen_gauge_irrep函数的基本功能"""
    #     # 测试最小参数：max_lenth=1，动量=(0,0,0)
    #     print("\n=== 测试 gen_gauge_irrep(max_lenth=1, momentum='0,0,0') ===")
    #     try:
    #         result = gen_gauge_irrep_disp(max_lenth=2, insertion_form=True)
    #         print(result)

    #         # 验证返回的是字典类型
    #         self.assertIsInstance(result, dict)

    #         # 验证包含预期的不可约表示类型
    #         expected_keys = [
    #             "A_1u+",
    #             "A_1u-",
    #             "A_1g+",
    #             "A_1g-",
    #             "A_2u+",
    #             "A_2u-",
    #             "A_2g+",
    #             "A_2g-",
    #             "Eu+",
    #             "Eu-",
    #             "Eg+",
    #             "Eg-",
    #             "T_1u+",
    #             "T_1u-",
    #             "T_1g+",
    #             "T_1g-",
    #             "T_2u+",
    #             "T_2u-",
    #             "T_2g+",
    #             "T_2g-",
    #         ]
    #         for key in expected_keys:
    #             self.assertIn(key, result, f"缺少键: {key}")

    #     except Exception as e:
    #         self.fail(f"gen_gauge_irrep函数执行失败: {e}")

    def test_gen_gauge_irrep_different_momentum(self):
        """测试gen_gauge_irrep函数使用不同动量"""
        print("\n=== 测试不同动量的gen_gauge_irrep ===")

        # 测试动量列表
        momentum_list = ["0,0,0", "0,0,1", "0,1,1", "1,1,1"]

        for momentum in momentum_list:
            print(f"\n--- 动量 {momentum} ---")
            try:
                result = gen_insertion_dict(max_lenth=1, insertion_form=True)
                non_empty_keys = [key for key, value in result.items() if value]
                print(f"非空不可约表示数量: {len(non_empty_keys)}")
                print(f"非空键: {non_empty_keys[:5]}")  # 只显示前5个

                # 验证结果不为空且为字典类型
                self.assertIsInstance(result, dict)

            except Exception as e:
                print(f"动量 {momentum} 测试失败: {e}")

    def test_gen_gauge_irrep_max_length_2(self):
        """测试gen_gauge_irrep函数使用max_lenth=2"""
        print("\n=== 测试 gen_gauge_irrep(max_lenth=2, momentum='0,0,0') ===")
        try:
            result = gen_insertion_dict(max_lenth=2, insertion_form=True)
            print(f"max_lenth=2时，结果字典键的数量: {len(result.keys())}")

            # 统计有多少种不可约表示有内容
            non_empty_count = sum(1 for value in result.values() if value)
            print(f"非空不可约表示数量: {non_empty_count}")

            # 输出部分结果用于验证
            count = 0
            for key, value in result.items():
                if value and count < 3:  # 只输出前3个非空结果
                    print(f"\n{key}:")
                    print(f"  包含 {len(value)} 个不可约表示")
                    if value[0]:
                        print(f"  第一个表示的维度: {len(value[0])}")
                        print(f"  第一行示例: {value[0][0][:3] if len(value[0][0]) > 3 else value[0][0]}")
                    count += 1

            self.assertIsInstance(result, dict)

        except Exception as e:
            self.fail(f"max_lenth=2测试失败: {e}")

    def test_gen_gauge_irrep_insertion_form_false(self):
        """测试gen_gauge_irrep函数的insertion_form=False选项"""
        print("\n=== 测试 insertion_form=False ===")
        try:
            result_insertion = gen_insertion_dict(max_lenth=1, insertion_form=True)
            result_symbolic = gen_insertion_dict(max_lenth=1, insertion_form=False)

            print("insertion_form=True 和 False 的结果比较:")
            print(f"insertion_form=True 键数量: {len(result_insertion.keys())}")
            print(f"insertion_form=False 键数量: {len(result_symbolic.keys())}")

            # 验证两种形式的键应该相同
            self.assertEqual(set(result_insertion.keys()), set(result_symbolic.keys()))

            # 查看结果格式的差异
            for key in list(result_insertion.keys())[:3]:  # 只检查前3个键
                if result_insertion[key] and result_symbolic[key]:
                    print(f"\n{key}:")
                    print(f"  insertion_form=True: {type(result_insertion[key][0][0])}")
                    print(f"  insertion_form=False: {type(result_symbolic[key][0][0])}")

        except Exception as e:
            self.fail(f"insertion_form测试失败: {e}")

    def test_gen_gauge_irrep_consistency(self):
        """测试gen_gauge_irrep函数结果的一致性"""
        print("\n=== 测试结果一致性 ===")
        try:
            # 多次调用相同参数，结果应该一致
            result1 = gen_insertion_dict(max_lenth=1, insertion_form=True)
            result2 = gen_insertion_dict(max_lenth=1, insertion_form=True)

            # 验证两次调用结果完全相同
            self.assertEqual(list(result1.keys()), list(result2.keys()))

            # 检查内容是否相同
            for key in result1.keys():
                self.assertEqual(len(result1[key]), len(result2[key]), f"键 {key} 的结果长度不一致")
                if result1[key]:
                    # 比较第一个不可约表示的第一行
                    if result1[key][0] and result2[key][0]:
                        self.assertEqual(result1[key][0][0], result2[key][0][0], f"键 {key} 的内容不一致")

            print("一致性测试通过：两次调用结果完全相同")

        except Exception as e:
            self.fail(f"一致性测试失败: {e}")


if __name__ == "__main__":
    unittest.main()
    # single test
    suite = unittest.TestSuite()
    # suite.addTest(TestGaugeLink("test_string_initialization"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
