import unittest
import numpy as np
import sympy as sp
from sympy import Matrix
from lattice.symmetry.gen_hardcoded_rep import genIrrepOhD
from lattice.symmetry.hardcoded_rep import (
    group_element,
    OhD_mul,
    OhD_inv,
    Fermion_rep,
    OD_irreps,
    little_group_irreps,
    refRotateDict,
    little_group_reduction_map,
    irrep_row_connection_dict,
)
from lattice.symmetry.group_generator import (
    Dic2_generator,
    Dic3_generator,
    Dic4_generator,
    C4_generator1,
    C4_generator2,
    OhD_generator,
    Fermion_generator,
)


class TestOhDGroup(unittest.TestCase):
    def test_identity(self):
        """测试单位元性质"""
        for ele in group_element:
            # 测试左单位元
            self.assertEqual(OhD_mul("iden", ele), ele)
            # 测试右单位元
            self.assertEqual(OhD_mul(ele, "iden"), ele)

    def test_inverse(self):
        """测试逆元性质"""
        for ele in group_element:
            inv_ele = OhD_inv(ele)
            # 测试左逆元
            self.assertEqual(OhD_mul(inv_ele, ele), "iden")
            # 测试右逆元
            self.assertEqual(OhD_mul(ele, inv_ele), "iden")

    def test_associativity(self):
        """测试结合律"""
        # 选择一些典型的群元进行测试
        test_elements = ["c4x", "c4y", "c4z", "c2x", "c2y", "c2z"]
        for a in test_elements:
            for b in test_elements:
                for c in test_elements:
                    # 测试 (a·b)·c = a·(b·c)
                    left = OhD_mul(OhD_mul(a, b), c)
                    right = OhD_mul(a, OhD_mul(b, c))
                    self.assertEqual(left, right)

    def test_specific_multiplications(self):
        """测试具体的乘法结果"""
        # 测试一些基本的乘法关系
        self.assertEqual(OhD_mul("c4x", "c4x"), "c2x")
        self.assertEqual(OhD_mul("c4y", "c4y"), "c2y")
        self.assertEqual(OhD_mul("c4z", "c4z"), "c2z")

        # 测试生成元之间的关系
        self.assertEqual(OhD_mul("c4x", "c4x^-1"), "iden")
        self.assertEqual(OhD_mul("c4y", "c4y^-1"), "iden")
        self.assertEqual(OhD_mul("c4z", "c4z^-1"), "iden")

    def test_generators_consistency(self):
        """测试表示与生成元是否一致"""
        generators = ["c4y", "c4z"]  # 主要的生成元
        for irrep_name, generator in OhD_generator.items():
            if irrep_name not in OD_irreps:
                continue
            irrep = OD_irreps[irrep_name]
            for g in generators:
                try:
                    gen_mat = np.array(generator[g]).astype(np.complex128)
                    rep_mat = np.array(irrep[g]).astype(np.complex128)
                    np.testing.assert_array_almost_equal(gen_mat, rep_mat)
                except AssertionError as e:
                    error_msg = (
                        f"\n不可约表示 {irrep_name} 的生成元 {g} 与定义不一致：\n"
                        f"生成元定义:\n{gen_mat}\n"
                        f"表示中的矩阵:\n{rep_mat}\n"
                        f"差异:\n{gen_mat - rep_mat}"
                    )
                    raise AssertionError(error_msg) from e


class TestFermionRepresentation(unittest.TestCase):
    def test_generators_consistency(self):
        """测试Fermion表示与生成元是否一致"""
        generators = ["c4y", "c4z", "inviden"]  # Fermion表示的生成元
        for g in generators:
            try:
                gen_mat = np.array(Fermion_generator[g]).astype(np.complex128)
                rep_mat = np.array(Fermion_rep[g]).astype(np.complex128)
                np.testing.assert_array_almost_equal(gen_mat, rep_mat)
            except AssertionError as e:
                error_msg = (
                    f"\nFermion表示的生成元 {g} 与定义不一致：\n"
                    f"生成元定义:\n{gen_mat}\n"
                    f"表示中的矩阵:\n{rep_mat}\n"
                    f"差异:\n{gen_mat - rep_mat}"
                )
                raise AssertionError(error_msg) from e

    def test_representation_multiplication(self):
        """测试表示是否保持群乘法关系"""
        for ele1 in group_element:
            for ele2 in group_element:
                prod = OhD_mul(ele1, ele2)
                mat_prod = np.dot(Fermion_rep[ele1], Fermion_rep[ele2])
                np.testing.assert_array_almost_equal(mat_prod, Fermion_rep[prod])


class TestODIrreps(unittest.TestCase):
    def test_generators_consistency(self):
        """测试表示与生成元是否一致"""
        generators = ["c4y", "c4z"]  # 主要的生成元
        for irrep_name, generator in OhD_generator.items():
            if irrep_name not in OD_irreps:
                continue
            irrep = OD_irreps[irrep_name]
            for g in generators:
                try:
                    gen_mat = np.array(generator[g]).astype(np.complex128)
                    rep_mat = np.array(irrep[g]).astype(np.complex128)
                    np.testing.assert_array_almost_equal(gen_mat, rep_mat)
                except AssertionError as e:
                    error_msg = (
                        f"\n不可约表示 {irrep_name} 的生成元 {g} 与定义不一致：\n"
                        f"生成元定义:\n{gen_mat}\n"
                        f"表示中的矩阵:\n{rep_mat}\n"
                        f"差异:\n{gen_mat - rep_mat}"
                    )
                    raise AssertionError(error_msg) from e

    def test_irreps_multiplication(self):
        """测试OD_irreps的每个不可约表示是否保持群乘法关系"""
        for irrep_name, irrep in OD_irreps.items():
            for ele1 in irrep.keys():
                for ele2 in irrep.keys():
                    prod = OhD_mul(ele1, ele2)
                    mat1 = np.array(irrep[ele1]).astype(np.complex128)
                    mat2 = np.array(irrep[ele2]).astype(np.complex128)
                    mat_prod = np.dot(mat1, mat2)
                    target = np.array(irrep[prod]).astype(np.complex128)
                    try:
                        np.testing.assert_array_almost_equal(mat_prod, target)
                    except AssertionError as e:
                        error_msg = (
                            f"\n不可约表示 {irrep_name} 的乘法关系不满足：\n"
                            f"元素: {ele1} * {ele2} = {prod}\n"
                            f"矩阵:\n{mat1}\n{mat2}\n"
                            f"矩阵乘积结果:\n{mat_prod}\n"
                            f"应该等于:\n{target}\n"
                            f"差异:\n{mat_prod - target}"
                        )
                        raise AssertionError(error_msg) from e


class TestLittleGroupIrreps(unittest.TestCase):
    def test_is_fixed_point(self):
        """测试是否为固定点"""
        for p_str in little_group_irreps.keys():
            momentun = [int(i) for i in p_str.split(",")]
            matrix_group = genIrrepOhD("T_1", -1)
            for ele in little_group_irreps[p_str]["A_1"].keys():
                np.testing.assert_array_almost_equal(matrix_group[ele] @ Matrix(momentun), Matrix(momentun))

    def test_ref_rotate(self):
        for pref_str in refRotateDict.keys():
            for p_str in refRotateDict[pref_str].keys():
                rotationn = refRotateDict[pref_str][p_str]
                matrix_group = genIrrepOhD("T_1", -1)
                pref = [int(i) for i in pref_str.split(",")]
                p = [int(i) for i in p_str.split(",")]
                np.testing.assert_array_almost_equal(matrix_group[rotationn] @ Matrix(pref), Matrix(p))

    def test_generators_consistency(self):
        for p_str, irreps in little_group_irreps.items():
            if p_str == "0,0,1":
                irrep_generators = Dic4_generator
            elif p_str == "0,1,1":
                irrep_generators = Dic2_generator
            elif p_str == "1,1,1":
                irrep_generators = Dic3_generator
            elif p_str == "0,0,0":
                irrep_generators = OhD_generator
            elif p_str == "0,1,2":
                irrep_generators = C4_generator2
            elif p_str == "2,1,1":
                irrep_generators = C4_generator1
            for irrep_name, generator in irrep_generators.items():
                for g in generator.keys():
                    try:
                        gen_mat = np.array(generator[g]).astype(np.complex128)
                        rep_mat = np.array(irreps[irrep_name][g]).astype(np.complex128)
                        np.testing.assert_array_almost_equal(gen_mat, rep_mat)
                    except AssertionError as e:
                        error_msg = (
                            f"不可约表示 {irrep_name} 的生成元 {g} 与定义不一致：\n",
                            f"p_str: {p_str}\n",
                            f"irrep_name: {irrep_name}\n",
                            f"g: {g}\n",
                            # f"gen_mat: {gen_mat}\n",
                            # f"rep_mat: {rep_mat}\n",
                            # f"gen_mat - rep_mat: {gen_mat - rep_mat}\n",
                        )
                        # error_msg = (
                        #     f"\n不可约表示 {irrep_name} 的生成元 {g} 与定义不一致：\n"
                        #     f"生成元定义:\n{gen_mat}\n"
                        #     f"表示中的矩阵:\n{rep_mat}\n"
                        #     f"差异:\n{gen_mat - rep_mat}"
                        # )
                        raise AssertionError(error_msg) from e

    def test_connection(self):
        for pstr in irrep_row_connection_dict.keys():
            for irrep_name in irrep_row_connection_dict[pstr].keys():
                connection = irrep_row_connection_dict[pstr][irrep_name]
                irrep = little_group_irreps[pstr][irrep_name]
                for i in range(len(connection)):
                    prj = Matrix.zeros(irrep["iden"].shape[0], 1)
                    origin = Matrix.zeros(irrep["iden"].shape[0], 1)
                    origin[0] = 1
                    for j in range(len(connection[i])):
                        prj += connection[i][j][0] * irrep[connection[i][j][1]] @ origin
                    prj = sp.simplify(prj)
                    target = Matrix.zeros(irrep["iden"].shape[0], 1)
                    target[i] = 1
                    np.testing.assert_array_almost_equal(prj, target)

    def test_irreps_multiplication(self):
        """测试小群不可约表示是否保持群乘法关系"""
        for p_str, irreps in little_group_irreps.items():
            for irrep_name, irrep in irreps.items():
                for ele1 in irrep.keys():
                    for ele2 in irrep.keys():
                        # 获取群元乘积
                        prod = OhD_mul(ele1, ele2)
                        # 获取表示矩阵
                        mat1 = np.array(irrep[ele1]).astype(np.complex128)
                        mat2 = np.array(irrep[ele2]).astype(np.complex128)
                        mat_prod = np.dot(mat1, mat2)
                        target = np.array(irrep[prod]).astype(np.complex128)
                        try:
                            np.testing.assert_array_almost_equal(mat_prod, target)
                        except AssertionError as e:
                            error_msg = (
                                f"\n动量点 {p_str} 的不可约表示 {irrep_name} 的乘法关系不满足：\n"
                                f"元素: {ele1} * {ele2} = {prod}\n"
                                f"矩阵:\n{mat1}\n{mat2}\n"
                                f"矩阵乘积结果:\n{mat_prod}\n"
                                f"应该等于:\n{target}\n"
                                f"差异:\n{mat_prod - target}"
                            )
                            raise AssertionError(error_msg) from e

    def test_reduction_map(self):
        for p_str in little_group_reduction_map.keys():
            for OhD_irrep_name in little_group_reduction_map[p_str].keys():
                if OhD_irrep_name.startswith("H") or OhD_irrep_name.startswith("G"):
                    continue
                if OhD_irrep_name.endswith("g"):
                    parity = 1
                else:
                    parity = -1
                OhD_matrix_group = genIrrepOhD(OhD_irrep_name[:-1], parity)
                for little_group_irrep_name, reduction_map in little_group_reduction_map[p_str][OhD_irrep_name].items():
                    reduction_matrix = Matrix(reduction_map[0])
                    lg_matrix_group = little_group_irreps[p_str][little_group_irrep_name]
                    for ele in lg_matrix_group.keys():
                        # print(ele)
                        # print(lg_matrix_group[ele])
                        # print(reduction_map)
                        for i in range(len(reduction_map)):
                            reduction_matrix = Matrix(reduction_map[i])
                            mat_proj = np.array(
                                reduction_matrix.conjugate() @ OhD_matrix_group[ele] @ reduction_matrix.T,
                                dtype=np.complex128,
                            )
                            target = np.array(lg_matrix_group[ele], dtype=np.complex128)
                            try:
                                np.testing.assert_array_almost_equal(mat_proj, target)
                            except AssertionError as e:
                                error_msg = (
                                    f"\n动量点 {p_str} 的OhD不可约表示 {OhD_irrep_name} 到小群不可约表示 {little_group_irrep_name} 的约化映射错误：\n"
                                    f"元素: {ele}\n"
                                    f"群元: {OhD_matrix_group[ele]}\n"
                                    f"小群群元: {lg_matrix_group[ele]}\n"
                                    f"约化矩阵:\n{reduction_matrix}\n"
                                    f"投影结果:\n{mat_proj}\n"
                                    f"目标矩阵:\n{target}\n"
                                    f"差异:\n{mat_proj - target}"
                                )
                                raise AssertionError(error_msg) from e
                        # exit()


class TestLittleGroup(unittest.TestCase):
    def test_little_group_origin(self):
        """测试原点 [0,0,0] 的小群"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        # 测试原点的小群，应该包含所有群元
        lg = littleGroup([0, 0, 0])
        self.assertIsInstance(lg, dict)
        self.assertGreater(len(lg), 0)

        # 验证每个群元都保持原点不变
        for key, matrix in lg.items():
            result = matrix @ Matrix([0, 0, 0])
            np.testing.assert_array_almost_equal(result, Matrix([0, 0, 0]))

    def test_little_group_momentum_001(self):
        """测试动量 [0,0,1] 的小群"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        lg = littleGroup([0, 0, 1])
        self.assertIsInstance(lg, dict)

        # 验证每个群元都保持 [0,0,1] 不变
        for key, matrix in lg.items():
            result = matrix @ Matrix([0, 0, 1])
            np.testing.assert_array_almost_equal(result, Matrix([0, 0, 1]))

    def test_little_group_momentum_011(self):
        """测试动量 [0,1,1] 的小群"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        lg = littleGroup([0, 1, 1])
        self.assertIsInstance(lg, dict)

        # 验证每个群元都保持 [0,1,1] 不变
        for key, matrix in lg.items():
            result = matrix @ Matrix([0, 1, 1])
            np.testing.assert_array_almost_equal(result, Matrix([0, 1, 1]))

    def test_little_group_momentum_111(self):
        """测试动量 [1,1,1] 的小群"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        lg = littleGroup([1, 1, 1])
        self.assertIsInstance(lg, dict)

        # 验证每个群元都保持 [1,1,1] 不变
        for key, matrix in lg.items():
            result = matrix @ Matrix([1, 1, 1])
            np.testing.assert_array_almost_equal(result, Matrix([1, 1, 1]))

    def test_little_group_momentum_112(self):
        """测试动量 [1,1,2] 的小群"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        lg = littleGroup([1, 1, 2])
        self.assertIsInstance(lg, dict)

        # 验证每个群元都保持 [1,1,2] 不变
        for key, matrix in lg.items():
            result = matrix @ Matrix([1, 1, 2])
            np.testing.assert_array_almost_equal(result, Matrix([1, 1, 2]))

    def test_little_group_size_comparison(self):
        """测试不同动量点的小群大小比较"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        # 原点的小群应该最大（包含所有群元）
        lg_origin = littleGroup([0, 0, 0])
        lg_001 = littleGroup([0, 0, 1])
        lg_011 = littleGroup([0, 1, 1])
        lg_111 = littleGroup([1, 1, 1])

        # 原点的小群应该包含所有群元
        self.assertGreater(len(lg_origin), len(lg_001))
        self.assertGreater(len(lg_origin), len(lg_011))
        self.assertGreater(len(lg_origin), len(lg_111))

    def test_little_group_identity_element(self):
        """测试小群包含单位元"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup

        test_momenta = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]]

        for momentum in test_momenta:
            lg = littleGroup(momentum)
            # 小群应该包含单位元
            self.assertIn("iden", lg)

            # 单位元应该保持动量不变
            identity_matrix = lg["iden"]
            result = identity_matrix @ Matrix(momentum)
            np.testing.assert_array_almost_equal(result, Matrix(momentum))

    def test_little_group_closure(self):
        """测试小群的封闭性"""
        from lattice.symmetry.gen_hardcoded_rep import littleGroup, OhD_mul

        lg = littleGroup([0, 0, 1])

        # 测试小群中任意两个元素的乘积仍然在小群中
        lg_keys = list(lg.keys())
        for i, key1 in enumerate(lg_keys):
            for key2 in lg_keys[i:]:  # 避免重复测试
                product = OhD_mul(key1, key2)
                self.assertIn(product, lg, f"小群不封闭: {key1} * {key2} = {product} 不在小群中")


if __name__ == "__main__":
    unittest.main()
    # single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestLittleGroupIrreps("test_generators_consistency"))
    # suite.addTest(TestLittleGroup("test_little_group_momentum_112"))
    # suite.addTest(TestLittleGroupIrreps("test_is_fixed_point"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
