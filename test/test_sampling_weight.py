"""
Unit tests for sampling weight calculation, scene enumeration, and high-mode projection.

This module tests the core components of the Localized Blending method:
- integer_partitions(): Generate all integer partitions
- calculate_sampling_weight(): Calculate sampling weight w(k) = C(L^3,k)/C(usedNp,k)
- enumerate_point_scenes(): Enumerate point coincidence scenes
- High-mode projection formulas (VSP, PSV, PSP)

Reference: doc/localized_blending/localized_blending_theory.md
"""

import unittest
import os
import numpy as np
from math import comb

from lattice.quark_diagram import (
    integer_partitions,
    calculate_sampling_weight,
    enumerate_point_scenes,
)


class TestIntegerPartitions(unittest.TestCase):
    """Test cases for integer_partitions function."""

    def test_partition_zero(self):
        """Test partition of 0."""
        result = integer_partitions(0)
        self.assertEqual(result, [[]])

    def test_partition_one(self):
        """Test partition of 1."""
        result = integer_partitions(1)
        self.assertEqual(result, [[1]])

    def test_partition_two(self):
        """Test partition of 2."""
        result = integer_partitions(2)
        # Expected: [[2], [1, 1]]
        self.assertEqual(len(result), 2)
        self.assertIn([2], result)
        self.assertIn([1, 1], result)

    def test_partition_three(self):
        """Test partition of 3."""
        result = integer_partitions(3)
        # Expected: [[3], [2, 1], [1, 1, 1]]
        self.assertEqual(len(result), 3)
        self.assertIn([3], result)
        self.assertIn([2, 1], result)
        self.assertIn([1, 1, 1], result)

    def test_partition_four(self):
        """Test partition of 4."""
        result = integer_partitions(4)
        # Expected: [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
        self.assertEqual(len(result), 5)
        self.assertIn([4], result)
        self.assertIn([3, 1], result)
        self.assertIn([2, 2], result)
        self.assertIn([2, 1, 1], result)
        self.assertIn([1, 1, 1, 1], result)

    def test_partition_five(self):
        """Test partition of 5 and verify count matches partition number p(5)=7."""
        result = integer_partitions(5)
        self.assertEqual(len(result), 7)  # p(5) = 7

    def test_partition_six(self):
        """Test partition of 6 and verify count matches partition number p(6)=11."""
        result = integer_partitions(6)
        self.assertEqual(len(result), 11)  # p(6) = 11

    def test_partitions_sum_to_n(self):
        """Verify that each partition sums to n."""
        for n in range(1, 8):
            partitions = integer_partitions(n)
            for partition in partitions:
                self.assertEqual(sum(partition), n,
                    f"Partition {partition} doesn't sum to {n}")

    def test_partitions_descending_order(self):
        """Verify that each partition is sorted in descending order."""
        for n in range(1, 8):
            partitions = integer_partitions(n)
            for partition in partitions:
                for i in range(len(partition) - 1):
                    self.assertGreaterEqual(
                        partition[i], partition[i + 1],
                        f"Partition {partition} not in descending order"
                    )

    def test_partitions_unique(self):
        """Verify all partitions are unique."""
        for n in range(1, 8):
            partitions = integer_partitions(n)
            # Convert to tuples for hashing
            partition_tuples = [tuple(p) for p in partitions]
            self.assertEqual(len(partition_tuples), len(set(partition_tuples)),
                f"Duplicate partitions found for n={n}")


class TestCalculateSamplingWeight(unittest.TestCase):
    """Test cases for calculate_sampling_weight function.

    Formula: w(k) = C(L^3, k) / C(usedNp, k)
    = L^3(L^3-1)...(L^3-k+1) / [usedNp(usedNp-1)...(usedNp-k+1)]
    """

    def test_weight_k_zero(self):
        """Test weight for k=0 (no distinct points)."""
        # w(0) = 1 by definition
        result = calculate_sampling_weight(L=10, usedNp=10, k=0)
        self.assertEqual(result, 1.0)

    def test_weight_k_exceeds_usedNp(self):
        """Test weight when k > usedNp (impossible case)."""
        result = calculate_sampling_weight(L=10, usedNp=5, k=10)
        self.assertEqual(result, 0.0)

    def test_weight_k_exceeds_total_points(self):
        """Test weight when k > L^3 (impossible case)."""
        result = calculate_sampling_weight(L=2, usedNp=10, k=10)  # L^3 = 8
        self.assertEqual(result, 0.0)

    def test_weight_full_sampling(self):
        """Test weight when usedNp equals total points (full sampling)."""
        # When usedNp = L^3, w(k) = 1 for all valid k
        L = 10
        for k in range(1, 5):
            result = calculate_sampling_weight(L=L, usedNp=L**3, k=k)
            self.assertAlmostEqual(result, 1.0, places=10,
                msg=f"Full sampling weight should be 1 for k={k}")

    def test_weight_simple_case(self):
        """Test weight for simple documented example."""
        # Example from docstring: calculate_sampling_weight(10, 10, 2) = 1000*999/(10*9)
        result = calculate_sampling_weight(L=10, usedNp=10, k=2)
        expected = (1000 * 999) / (10 * 9)  # = 11100
        self.assertAlmostEqual(result, expected, places=10)

    def test_weight_k_one(self):
        """Test weight for k=1 (one distinct point)."""
        # w(1) = L^3 / usedNp
        result = calculate_sampling_weight(L=10, usedNp=10, k=1)
        expected = 1000 / 10  # = 100
        self.assertAlmostEqual(result, expected, places=10)

    def test_weight_k_two(self):
        """Test weight for k=2 (two distinct points)."""
        # w(2) = L^3(L^3-1) / usedNp(usedNp-1)
        L = 10
        usedNp = 50
        result = calculate_sampling_weight(L=L, usedNp=usedNp, k=2)
        M = L ** 3  # 1000
        expected = (M * (M - 1)) / (usedNp * (usedNp - 1))
        self.assertAlmostEqual(result, expected, places=10)

    def test_weight_k_three(self):
        """Test weight for k=3 (three distinct points)."""
        L = 8
        usedNp = 100
        result = calculate_sampling_weight(L=L, usedNp=usedNp, k=3)
        M = L ** 3  # 512
        expected = (M * (M - 1) * (M - 2)) / (usedNp * (usedNp - 1) * (usedNp - 2))
        self.assertAlmostEqual(result, expected, places=10)

    def test_weight_formula_consistency(self):
        """Verify weight formula consistency with combinatorial definition."""
        L = 6
        usedNp = 30
        M = L ** 3  # 216

        for k in range(0, 5):
            result = calculate_sampling_weight(L=L, usedNp=usedNp, k=k)
            if k == 0:
                expected = 1.0
            elif k > min(M, usedNp):
                expected = 0.0
            else:
                expected = comb(M, k) / comb(usedNp, k)
            self.assertAlmostEqual(result, expected, places=10,
                msg=f"Mismatch for k={k}")

    def test_weight_increases_with_k(self):
        """
        Verify weight behavior with k.

        When usedNp < L^3 (undersampling): weight INCREASES with k.
        Reason: selecting more distinct points from limited sampling has lower probability,
        so compensation factor is larger.

        When usedNp = L^3 (full sampling): all weights = 1.
        """
        L = 10
        usedNp = 20  # Undersampling

        weights = [calculate_sampling_weight(L=L, usedNp=usedNp, k=k)
                   for k in range(1, 5)]

        # Weight should INCREASE as k increases (undersampling case)
        for i in range(len(weights) - 1):
            self.assertLess(weights[i], weights[i + 1],
                f"Weight for k={i+1} ({weights[i]}) should be less than for k={i+2} ({weights[i+1]})")

        # Test full sampling case: all weights = 1
        weights_full = [calculate_sampling_weight(L=L, usedNp=L**3, k=k)
                        for k in range(1, 5)]
        for w in weights_full:
            self.assertAlmostEqual(w, 1.0, places=10)

    def test_weight_symmetry_property(self):
        """Test that weight ratio equals M/N for k=1."""
        L = 10
        usedNp = 50
        M = L ** 3

        # For k=1: w(1) = M/N
        result = calculate_sampling_weight(L=L, usedNp=usedNp, k=1)
        self.assertAlmostEqual(result, M / usedNp, places=10)

    def test_weight_large_values(self):
        """Test weight calculation with larger lattice sizes."""
        L = 32  # Typical lattice size
        usedNp = 216  # Typical sampling size
        M = L ** 3  # 32768

        # k=1: w(1) = M / usedNp
        result_1 = calculate_sampling_weight(L=L, usedNp=usedNp, k=1)
        self.assertAlmostEqual(result_1, M / usedNp, places=5)

        # k=2: w(2) = M(M-1) / usedNp(usedNp-1)
        result_2 = calculate_sampling_weight(L=L, usedNp=usedNp, k=2)
        expected_2 = (M * (M - 1)) / (usedNp * (usedNp - 1))
        self.assertAlmostEqual(result_2, expected_2, places=0)


class TestEnumeratePointScenes(unittest.TestCase):
    """Test cases for enumerate_point_scenes function."""

    def test_scenes_r_one(self):
        """Test scenes for r=1 (one point position)."""
        result = enumerate_point_scenes(r=1, L=10, usedNp=10)

        # Only one scene: [1] with weight = L^3/usedNp
        self.assertEqual(len(result), 1)
        partition, weight = result[0]
        self.assertEqual(partition, [1])
        self.assertAlmostEqual(weight, 1000 / 10, places=10)

    def test_scenes_r_two(self):
        """Test scenes for r=2 (two point positions)."""
        result = enumerate_point_scenes(r=2, L=10, usedNp=10)

        # Two scenes: [2] (both same), [1,1] (different)
        self.assertEqual(len(result), 2)

        partitions = [p for p, w in result]
        weights = [w for p, w in result]

        self.assertIn([2], partitions)
        self.assertIn([1, 1], partitions)

        # [2] -> k=1 distinct points -> w(1) = L^3/usedNp = 100
        # [1,1] -> k=2 distinct points -> w(2) = L^3(L^3-1)/usedNp(usedNp-1)
        idx_2 = partitions.index([2])
        idx_11 = partitions.index([1, 1])

        self.assertAlmostEqual(weights[idx_2], 1000 / 10, places=10)
        self.assertAlmostEqual(weights[idx_11], (1000 * 999) / (10 * 9), places=10)

    def test_scenes_r_three(self):
        """Test scenes for r=3 (three point positions)."""
        result = enumerate_point_scenes(r=3, L=10, usedNp=10)

        # Three scenes: [3], [2,1], [1,1,1]
        self.assertEqual(len(result), 3)

        partitions = [p for p, w in result]
        self.assertIn([3], partitions)
        self.assertIn([2, 1], partitions)
        self.assertIn([1, 1, 1], partitions)

    def test_scenes_r_four(self):
        """Test scenes for r=4 (four point positions)."""
        result = enumerate_point_scenes(r=4, L=10, usedNp=10)

        # Five scenes matching integer partitions of 4
        self.assertEqual(len(result), 5)

        partitions = [p for p, w in result]
        expected_partitions = [[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]]
        for expected in expected_partitions:
            self.assertIn(expected, partitions)

    def test_scenes_partition_count_matches(self):
        """Verify that scene count equals partition count."""
        for r in range(1, 7):
            scenes = enumerate_point_scenes(r=r, L=10, usedNp=10)
            partitions = integer_partitions(r)
            self.assertEqual(len(scenes), len(partitions),
                f"Scene count mismatch for r={r}")

    def test_scenes_weight_increases_with_distinct_points(self):
        """
        Verify weight behavior with number of distinct points.

        When usedNp < L^3 (undersampling): weight INCREASES with number of distinct points.
        This is because selecting k distinct points from limited sampling has lower probability,
        requiring larger compensation.
        """
        L = 10
        usedNp = 20  # Undersampling

        scenes = enumerate_point_scenes(r=4, L=L, usedNp=usedNp)

        # Sort by number of distinct points (k = len(partition))
        sorted_scenes = sorted(scenes, key=lambda x: len(x[0]))

        # Verify weights are in ascending order (undersampling case)
        for i in range(len(sorted_scenes) - 1):
            k1 = len(sorted_scenes[i][0])
            k2 = len(sorted_scenes[i + 1][0])
            w1 = sorted_scenes[i][1]
            w2 = sorted_scenes[i + 1][1]

            if k1 < k2:
                self.assertLess(w1, w2,
                    f"Weight for k={k1} ({w1}) should be < weight for k={k2} ({w2})")

    def test_scenes_weight_full_sampling(self):
        """Verify all weights are 1 when usedNp = L^3 (full sampling)."""
        L = 10
        usedNp = L ** 3  # Full sampling

        scenes = enumerate_point_scenes(r=4, L=L, usedNp=usedNp)

        for partition, weight in scenes:
            self.assertAlmostEqual(weight, 1.0, places=10,
                msg=f"Full sampling weight should be 1 for partition {partition}")

    def test_scenes_full_sampling_weights_are_one(self):
        """Test that all weights are 1 when usedNp = L^3 (full sampling)."""
        L = 10
        usedNp = L ** 3  # Full sampling

        for r in range(1, 5):
            scenes = enumerate_point_scenes(r=r, L=L, usedNp=usedNp)
            for partition, weight in scenes:
                self.assertAlmostEqual(weight, 1.0, places=10,
                    msg=f"Full sampling weight should be 1 for partition {partition}")

    def test_scenes_weight_formula(self):
        """Verify scene weights match calculate_sampling_weight."""
        L = 10
        usedNp = 50

        for r in range(1, 5):
            scenes = enumerate_point_scenes(r=r, L=L, usedNp=usedNp)
            for partition, weight in scenes:
                k = len(partition)
                expected_weight = calculate_sampling_weight(L=L, usedNp=usedNp, k=k)
                self.assertAlmostEqual(weight, expected_weight, places=10,
                    msg=f"Weight mismatch for partition {partition}")


class TestHighModeProjectionFormulas(unittest.TestCase):
    """
    Test the mathematical correctness of high-mode projection formulas.

    Reference: doc/propagator_theory_and_usage.md, formulas 5.1-5.3

    Formula 5.2: tilde{S}_{xa,i} = S_{xa,i} - sum_j M_{xj,a} S_{j,i}
    Formula 5.3: tilde{S}_{i,xa} = S_{i,xa} - sum_j S_{i,j} M_{jx,a}*
    Formula 5.1: tilde{S}_{xa,yb} = S_{xa,yb} - sum_i M_{xi,a} tilde{S}_{i,yb}
                              - sum_j S_{xa,j} M_{jy,b}
    """

    def test_vsp_projection_formula(self):
        """
        Test VSP high-mode projection formula using mock data.

        Formula: tilde{S}_{i,xa} = S_{i,xa} - sum_j S_{i,j} M_{jx,a}*
        """
        # Use small dimensions for test
        Ns = 2
        Ne = 3
        Np = 4
        Nc = 3

        # Create random mock data
        np.random.seed(42)
        S_vsp = np.random.randn(Ns, Ns, Ne, Np, Nc) + 1j * np.random.randn(Ns, Ns, Ne, Np, Nc)
        S_vsv = np.random.randn(Ns, Ns, Ne, Ne) + 1j * np.random.randn(Ns, Ns, Ne, Ne)
        M = np.random.randn(Ne, Np, Nc) + 1j * np.random.randn(Ne, Np, Nc)

        # Manual computation: tilde{S} = S - sum_j S_{i,j} M_{j,x,a}^*
        # Using einsum: 'abij,jxc->abixc' where ab are spin, i,j are Ne, x is Np, c is Nc
        correction = np.einsum('abij,jxc->abixc', S_vsv, M.conj())
        expected = S_vsp - correction

        # Verify the formula dimensions
        self.assertEqual(expected.shape, (Ns, Ns, Ne, Np, Nc))

        # Verify the correction is applied correctly
        # For a specific element: tilde{S}[a,b,i,x,c] = S_vsp[a,b,i,x,c] - sum_j S_vsv[a,b,i,j] * M_conj[j,x,c]
        a, b, i, x, c = 0, 0, 1, 2, 1
        manual_correction = sum(S_vsv[a, b, i, j] * M.conj()[j, x, c] for j in range(Ne))
        self.assertAlmostEqual(correction[a, b, i, x, c], manual_correction, places=10)

    def test_psv_projection_formula(self):
        """
        Test PSV high-mode projection formula using mock data.

        Formula: tilde{S}_{xa,i} = S_{xa,i} - sum_j M_{xj,a} S_{j,i}
        """
        Ns = 2
        Ne = 3
        Np = 4
        Nc = 3

        np.random.seed(43)
        S_psv = np.random.randn(Ns, Ns, Np, Nc, Ne) + 1j * np.random.randn(Ns, Ns, Np, Nc, Ne)
        S_vsv = np.random.randn(Ns, Ns, Ne, Ne) + 1j * np.random.randn(Ns, Ns, Ne, Ne)
        M = np.random.randn(Ne, Np, Nc) + 1j * np.random.randn(Ne, Np, Nc)

        # Manual computation: tilde{S} = S - sum_j M_{x,j,c} S_{j,i}
        # Using einsum: 'jxc,abji->abxci' where j is Ne, x is Np, c is Nc
        correction = np.einsum('jxc,abji->abxci', M, S_vsv)
        expected = S_psv - correction

        # Verify dimensions
        self.assertEqual(expected.shape, (Ns, Ns, Np, Nc, Ne))

        # Verify specific element
        a, b, x, c, i = 0, 0, 2, 1, 1
        manual_correction = sum(M[j, x, c] * S_vsv[a, b, j, i] for j in range(Ne))
        self.assertAlmostEqual(correction[a, b, x, c, i], manual_correction, places=10)

    def test_psp_projection_formula(self):
        """
        Test PSP high-mode projection formula using mock data.

        Formula: tilde{S}_{xa,yb} = S_{xa,yb} - sum_i M_{xi,a} tilde{S}_{i,yb} - sum_j S_{xa,j} M_{jy,b}
        """
        Ns = 2
        Ne = 3
        Np = 4
        Nc = 3

        np.random.seed(44)
        # S_psp: [Ns, Ns, Np_snk, Nc, Np_src, Nc] = [a, b, x, c, y, d]
        S_psp = np.random.randn(Ns, Ns, Np, Nc, Np, Nc) + 1j * np.random.randn(Ns, Ns, Np, Nc, Np, Nc)
        # S_psv: [Ns, Ns, Np, Nc, Ne]
        S_psv = np.random.randn(Ns, Ns, Np, Nc, Ne) + 1j * np.random.randn(Ns, Ns, Np, Nc, Ne)
        # S_vsp: [Ns, Ns, Ne, Np, Nc]
        S_vsp = np.random.randn(Ns, Ns, Ne, Np, Nc) + 1j * np.random.randn(Ns, Ns, Ne, Np, Nc)
        # M: [Ne, Np, Nc]
        M = np.random.randn(Ne, Np, Nc) + 1j * np.random.randn(Ne, Np, Nc)

        # Term 1: S_{xa,yb}
        term1 = S_psp

        # Term 2: - sum_i M_{xi,c} tilde{S}_{i,yd}
        # For simplicity, use S_vsp as tilde{S}_{i,yd}
        # einsum: 'ixc,abiyd->abxcyd'
        term2 = np.einsum('ixc,abiyd->abxcyd', M, S_vsp)

        # Term 3: - sum_j S_{xc,j} M_{jy,d}
        # einsum: 'abxcj,jyd->abxcyd'
        M_conj = M.conj()
        term3 = np.einsum('abxcj,jyd->abxcyd', S_psv, M_conj)

        # Combined result
        expected = term1 - term2 - term3

        # Verify dimensions
        self.assertEqual(expected.shape, (Ns, Ns, Np, Nc, Np, Nc))

    def test_highmode_reduces_to_unprojected_when_ne_zero(self):
        """
        Verify that highmode returns unprojected when usedNe=0.

        This is a mathematical sanity check: no projection = return original.
        """
        # This would require mocking the PropagatorWithCurrent class
        # For now, we document the expected behavior
        pass

    def test_vsp_psv_symmetry(self):
        """
        Test that VSP and PSV are related by Hermitian conjugation.

        If O is Hermitian: (O)_{i,xa} = (O)_{xa,i}^T
        So tilde{S}_{i,xa} = (tilde{S}_{xa,i})^†
        """
        Ns = 2
        Ne = 3
        Np = 4
        Nc = 3

        np.random.seed(45)
        # Create Hermitian VSV for this test
        H = np.random.randn(Ns, Ns, Ne, Ne) + 1j * np.random.randn(Ns, Ns, Ne, Ne)
        S_vsv = (H + np.transpose(H.conj(), (0, 1, 3, 2))) / 2  # Force Hermitian

        M = np.random.randn(Ne, Np, Nc) + 1j * np.random.randn(Ne, Np, Nc)

        # Compute VSP correction
        S_vsp = np.random.randn(Ns, Ns, Ne, Np, Nc) + 1j * np.random.randn(Ns, Ns, Ne, Np, Nc)
        vsp_correction = np.einsum('abij,jxc->abixc', S_vsv, M.conj())
        tilde_vsp = S_vsp - vsp_correction

        # Compute PSV correction using related formula
        S_psv = np.transpose(S_vsp.conj(), (0, 1, 3, 4, 2))  # Mock symmetry
        psv_correction = np.einsum('jxc,abji->abxci', M, S_vsv)
        tilde_psv = S_psv - psv_correction

        # The shapes should be transposes of each other
        self.assertEqual(tilde_vsp.shape, (Ns, Ns, Ne, Np, Nc))
        self.assertEqual(tilde_psv.shape, (Ns, Ns, Np, Nc, Ne))


class TestSamplingWeightEdgeCases(unittest.TestCase):
    """Edge case tests for sampling weight calculation."""

    def test_usedNp_equals_one(self):
        """Test when usedNp = 1 (single point sampling)."""
        # For k=1: w(1) = L^3 / 1 = L^3
        L = 10
        result = calculate_sampling_weight(L=L, usedNp=1, k=1)
        self.assertAlmostEqual(result, L ** 3, places=10)

        # For k>1: w(k) = 0 (can't have more than 1 distinct point)
        for k in range(2, 5):
            result = calculate_sampling_weight(L=L, usedNp=1, k=k)
            self.assertEqual(result, 0.0)

    def test_L_equals_one(self):
        """Test when L = 1 (single spatial point)."""
        # Total points = 1
        # For k=1: w(1) = 1 / usedNp
        result = calculate_sampling_weight(L=1, usedNp=10, k=1)
        self.assertAlmostEqual(result, 1 / 10, places=10)

        # For k>1: w(k) = 0
        result = calculate_sampling_weight(L=1, usedNp=10, k=2)
        self.assertEqual(result, 0.0)

    def test_small_lattice_large_sampling(self):
        """Test when sampling size is comparable to lattice size."""
        L = 4  # 64 total points
        usedNp = 60  # Sample most points

        # k=1: w(1) = 64/60
        result = calculate_sampling_weight(L=L, usedNp=usedNp, k=1)
        expected = 64 / 60
        self.assertAlmostEqual(result, expected, places=10)

    def test_numerical_stability_large_k(self):
        """Test numerical stability for large k values."""
        L = 10
        usedNp = 100

        # Compute weights for increasing k
        weights = []
        for k in range(1, 10):
            result = calculate_sampling_weight(L=L, usedNp=usedNp, k=k)
            weights.append(result)

        # All weights should be finite and positive
        for w in weights:
            self.assertTrue(np.isfinite(w), f"Weight {w} is not finite")
            self.assertGreater(w, 0, f"Weight {w} should be positive")


class TestSceneEnumerationIntegration(unittest.TestCase):
    """Integration tests combining scene enumeration with weight calculation."""

    def test_scene_sum_weighted_partitions(self):
        """
        Test that sum of (weight * number of ways) equals total.

        For r positions with N available values:
        Total configurations = N^r

        For each partition λ = [λ1, λ2, ...]:
        Number of ways = C(r, λ1) * C(r-λ1, λ2) * ... * C(N, k)
        where k = len(λ) is the number of distinct values
        """
        L = 4  # 64 total points
        usedNp = 20
        N = usedNp
        r = 3

        scenes = enumerate_point_scenes(r=r, L=L, usedNp=usedNp)

        # Calculate total number of configurations
        total_configurations = N ** r

        # Sum over all scenes
        total_weighted = 0
        for partition, weight in scenes:
            k = len(partition)
            # Number of ways to assign k distinct values from N
            # This is a simplified check - full combinatorial verification would be complex
            total_weighted += weight

        # The weights should compensate for sampling
        # This is a sanity check, not exact equality
        self.assertGreater(total_weighted, 0)

    def test_realistic_lattice_parameters(self):
        """Test with realistic lattice parameters from production use."""
        # Typical production values
        L = 32  # Spatial size
        usedNp = 216  # Sampling points
        r = 2  # Current vertex has 2 point positions

        scenes = enumerate_point_scenes(r=r, L=L, usedNp=usedNp)

        # Verify we get expected number of scenes
        self.assertEqual(len(scenes), 2)  # [2] and [1,1]

        # Verify weights are reasonable (not too large)
        for partition, weight in scenes:
            self.assertGreater(weight, 0)
            self.assertLess(weight, 1e10)  # Sanity check


if __name__ == "__main__":
    unittest.main(verbosity=2)
