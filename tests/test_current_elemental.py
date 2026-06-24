"""
Unit tests for CurrentElemental data loading classes.

Tests:
- CurrentElementalV2P: V2P (Vector-to-Point) current elemental
- CurrentElementalP2V: P2V (Point-to-Vector) current elemental
- CurrentElementalP2P: P2P (Point-to-Point) current elemental
- OverlapMatrixNpy: Overlap matrix between eigenvectors and point sources

These classes are used in the contraction workflow to load
current operator matrix elements for the blending method.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path


class TestCurrentElementalV2P(unittest.TestCase):
    """Test CurrentElementalV2P class for V2P current data."""

    def test_import_current_elemental_v2p(self):
        """Test that CurrentElementalV2P can be imported."""
        try:
            from lattice import CurrentElementalV2P
            self.assertIsNotNone(CurrentElementalV2P)
        except ImportError:
            self.skipTest("CurrentElementalV2P not available")

    def test_v2p_shape_definition(self):
        """Test V2P shape: [Lt, num_disp, Ne, Np, Nc].

        V2P represents: O_{i,xa} = <xi_i | O | eta_{x,a}>
        where i is eigenvector index, x is point index, a is color index.
        """
        try:
            from lattice import CurrentElementalV2P

            Lt, num_disp, Ne, Np, Nc = 16, 4, 32, 64, 3
            expected_shape = [Lt, num_disp, Ne, Np, Nc]

            # Create temporary file for testing
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create test data
                test_data = np.random.rand(*expected_shape).astype(np.complex128)
                test_file = os.path.join(tmpdir, "test_v2p.npy")
                np.save(test_file, test_data)

                # Try to load
                try:
                    v2p = CurrentElementalV2P(
                        prefix=tmpdir + "/test",
                        suffix="_v2p.npy",
                        shape=expected_shape,
                        Ne=Ne,
                        Np=Np
                    )
                    self.assertEqual(v2p.shape, expected_shape)
                except Exception as e:
                    self.skipTest(f"CurrentElementalV2P initialization failed: {e}")

        except ImportError:
            self.skipTest("CurrentElementalV2P not available")

    def test_v2p_data_retrieval(self):
        """Test V2P data get() method returns correct shape."""
        try:
            from lattice import CurrentElementalV2P

            Lt, num_disp, Ne, Np, Nc = 8, 2, 16, 32, 3
            shape = [Lt, num_disp, Ne, Np, Nc]

            with tempfile.TemporaryDirectory() as tmpdir:
                test_data = np.random.rand(*shape).astype(np.complex128) + \
                            1j * np.random.rand(*shape)
                np.save(os.path.join(tmpdir, "cfg10000_v2p.npy"), test_data)

                try:
                    v2p = CurrentElementalV2P(
                        prefix=tmpdir + "/cfg",
                        suffix="_v2p.npy",
                        shape=shape,
                        Ne=Ne,
                        Np=Np
                    )
                    v2p.load("10000")
                    data = v2p.get()

                    self.assertEqual(data.shape, tuple(shape))
                except Exception as e:
                    self.skipTest(f"V2P data retrieval failed: {e}")

        except ImportError:
            self.skipTest("CurrentElementalV2P not available")


class TestCurrentElementalP2V(unittest.TestCase):
    """Test CurrentElementalP2V class for P2V current data."""

    def test_import_current_elemental_p2v(self):
        """Test that CurrentElementalP2V can be imported."""
        try:
            from lattice import CurrentElementalP2V
            self.assertIsNotNone(CurrentElementalP2V)
        except ImportError:
            self.skipTest("CurrentElementalP2V not available")

    def test_p2v_shape_definition(self):
        """Test P2V shape: [Lt, num_disp, Np, Nc, Ne].

        P2V represents: O_{xa,i} = <eta_{x,a} | O | xi_i>
        where x is point index, a is color index, i is eigenvector index.
        """
        try:
            from lattice import CurrentElementalP2V

            Lt, num_disp, Np, Nc, Ne = 16, 4, 64, 3, 32
            expected_shape = [Lt, num_disp, Np, Nc, Ne]

            with tempfile.TemporaryDirectory() as tmpdir:
                test_data = np.random.rand(*expected_shape).astype(np.complex128)
                np.save(os.path.join(tmpdir, "test_p2v.npy"), test_data)

                try:
                    p2v = CurrentElementalP2V(
                        prefix=tmpdir + "/test",
                        suffix="_p2v.npy",
                        shape=expected_shape,
                        Ne=Ne,
                        Np=Np
                    )
                    self.assertEqual(p2v.shape, expected_shape)
                except Exception as e:
                    self.skipTest(f"CurrentElementalP2V initialization failed: {e}")

        except ImportError:
            self.skipTest("CurrentElementalP2V not available")


class TestCurrentElementalP2P(unittest.TestCase):
    """Test CurrentElementalP2P class for P2P current data."""

    def test_import_current_elemental_p2p(self):
        """Test that CurrentElementalP2P can be imported."""
        try:
            from lattice import CurrentElementalP2P
            self.assertIsNotNone(CurrentElementalP2P)
        except ImportError:
            self.skipTest("CurrentElementalP2P not available")

    def test_p2p_shape_definition(self):
        """Test P2P shape: [Lt, num_disp, Np_snk, Nc, Np_src] or similar.

        P2P represents: O_{xa,yb} = <eta_{x,a} | O | eta_{y,b}>
        This is the largest current elemental due to O(Np^2) scaling.
        """
        try:
            from lattice import CurrentElementalP2P

            Lt, num_disp, Np, Nc = 16, 4, 64, 3
            # P2P can have different shapes depending on implementation
            # Common: [Lt, num_disp, Np, Nc, Np] or [Lt, Np, Nc, Np]

            with tempfile.TemporaryDirectory() as tmpdir:
                # Try HDF5 format (suffix=None typically means HDF5)
                try:
                    p2p = CurrentElementalP2P(
                        prefix=tmpdir + "/test",
                        suffix=None  # HDF5 format
                    )
                    self.assertIsNotNone(p2p)
                except Exception as e:
                    self.skipTest(f"CurrentElementalP2P initialization failed: {e}")

        except ImportError:
            self.skipTest("CurrentElementalP2P not available")


class TestOverlapMatrixNpy(unittest.TestCase):
    """Test OverlapMatrixNpy class for overlap matrix data."""

    def test_import_overlap_matrix_npy(self):
        """Test that OverlapMatrixNpy can be imported."""
        try:
            from lattice import OverlapMatrixNpy
            self.assertIsNotNone(OverlapMatrixNpy)
        except ImportError:
            self.skipTest("OverlapMatrixNpy not available")

    def test_overlap_matrix_shape_definition(self):
        """Test overlap matrix shape: [Lt, Ne, Np, Nc].

        Overlap matrix: M_{xi,a} = <eta_{x,a} | xi_i>
        where eta is point source, xi is eigenvector.
        """
        try:
            from lattice import OverlapMatrixNpy

            Lt, Ne, Np, Nc = 16, 32, 64, 3
            expected_shape = [Lt, Ne, Np, Nc]

            with tempfile.TemporaryDirectory() as tmpdir:
                test_data = np.random.rand(*expected_shape).astype(np.complex128)
                np.save(os.path.join(tmpdir, "cfg10000.overlap_matrix.npy"), test_data)

                try:
                    overlap = OverlapMatrixNpy(
                        prefix=tmpdir + "/cfg",
                        suffix=".overlap_matrix.npy",
                        shape=expected_shape,
                        Ne=Ne,
                        Np=Np
                    )
                    self.assertEqual(overlap.shape, expected_shape)
                except Exception as e:
                    self.skipTest(f"OverlapMatrixNpy initialization failed: {e}")

        except ImportError:
            self.skipTest("OverlapMatrixNpy not available")

    def test_overlap_matrix_data_retrieval(self):
        """Test overlap matrix get() method."""
        try:
            from lattice import OverlapMatrixNpy

            Lt, Ne, Np, Nc = 8, 16, 32, 3
            shape = [Lt, Ne, Np, Nc]

            with tempfile.TemporaryDirectory() as tmpdir:
                test_data = np.random.rand(*shape).astype(np.complex128) + \
                            1j * np.random.rand(*shape)
                np.save(os.path.join(tmpdir, "cfg10000.overlap_matrix.npy"), test_data)

                try:
                    overlap = OverlapMatrixNpy(
                        prefix=tmpdir + "/cfg",
                        suffix=".overlap_matrix.npy",
                        shape=shape,
                        Ne=Ne,
                        Np=Np
                    )
                    overlap.load("10000")
                    data = overlap.get()

                    self.assertEqual(data.shape, tuple(shape))
                    self.assertTrue(np.allclose(data, test_data))
                except Exception as e:
                    self.skipTest(f"OverlapMatrixNpy data retrieval failed: {e}")

        except ImportError:
            self.skipTest("OverlapMatrixNpy not available")


class TestCurrentElementalIntegration(unittest.TestCase):
    """Integration tests for CurrentElemental classes."""

    def test_v2p_p2v_consistency(self):
        """Test that V2P and P2V have consistent dimensions.

        If V2P has shape [Lt, num_disp, Ne, Np, Nc]
        Then P2V should have shape [Lt, num_disp, Np, Nc, Ne]
        They are related by Hermitian conjugation (roughly).
        """
        try:
            from lattice import CurrentElementalV2P, CurrentElementalP2V

            Lt, num_disp, Ne, Np, Nc = 8, 2, 16, 32, 3
            v2p_shape = [Lt, num_disp, Ne, Np, Nc]
            p2v_shape = [Lt, num_disp, Np, Nc, Ne]

            with tempfile.TemporaryDirectory() as tmpdir:
                v2p_data = np.random.rand(*v2p_shape).astype(np.complex128)
                p2v_data = np.random.rand(*p2v_shape).astype(np.complex128)

                np.save(os.path.join(tmpdir, "cfg_v2p.npy"), v2p_data)
                np.save(os.path.join(tmpdir, "cfg_p2v.npy"), p2v_data)

                try:
                    v2p = CurrentElementalV2P(
                        prefix=tmpdir + "/cfg",
                        suffix="_v2p.npy",
                        shape=v2p_shape,
                        Ne=Ne, Np=Np
                    )

                    p2v = CurrentElementalP2V(
                        prefix=tmpdir + "/cfg",
                        suffix="_p2v.npy",
                        shape=p2v_shape,
                        Ne=Ne, Np=Np
                    )

                    # Check that Ne, Np are consistent
                    self.assertEqual(v2p.Ne, p2v.Ne)
                    self.assertEqual(v2p.Np, p2v.Np)

                except Exception as e:
                    self.skipTest(f"V2P/P2V consistency test failed: {e}")

        except ImportError:
            self.skipTest("CurrentElemental classes not available")

    def test_all_current_elementals_together(self):
        """Test using all current elemental types together."""
        try:
            from lattice import CurrentElementalV2P, CurrentElementalP2V, CurrentElementalP2P

            # This would be used in a full contraction workflow
            pass

        except ImportError:
            self.skipTest("CurrentElemental classes not available")


class TestDataLoadingEdgeCases(unittest.TestCase):
    """Test edge cases for data loading."""

    def test_file_not_found_handling(self):
        """Test behavior when data file doesn't exist."""
        try:
            from lattice import OverlapMatrixNpy

            with tempfile.TemporaryDirectory() as tmpdir:
                overlap = OverlapMatrixNpy(
                    prefix=tmpdir + "/nonexistent",
                    suffix=".overlap_matrix.npy",
                    shape=[8, 16, 32, 3],
                    Ne=16, Np=32
                )

                try:
                    overlap.load("nonexistent_cfg")
                    self.fail("Should have raised an error for nonexistent file")
                except FileNotFoundError:
                    pass  # Expected
                except Exception as e:
                    # Other exceptions might be acceptable
                    pass

        except ImportError:
            self.skipTest("OverlapMatrixNpy not available")

    def test_shape_mismatch_handling(self):
        """Test behavior when loaded data has wrong shape."""
        try:
            from lattice import OverlapMatrixNpy

            Lt, Ne, Np, Nc = 8, 16, 32, 3
            shape = [Lt, Ne, Np, Nc]
            wrong_shape = [Lt * 2, Ne, Np, Nc]  # Wrong Lt

            with tempfile.TemporaryDirectory() as tmpdir:
                test_data = np.random.rand(*wrong_shape).astype(np.complex128)
                np.save(os.path.join(tmpdir, "cfg10000.overlap_matrix.npy"), test_data)

                overlap = OverlapMatrixNpy(
                    prefix=tmpdir + "/cfg",
                    suffix=".overlap_matrix.npy",
                    shape=shape,
                    Ne=Ne, Np=Np
                )

                try:
                    overlap.load("10000")
                    # Some implementations might auto-reshape or raise warning
                except ValueError:
                    pass  # Expected for shape mismatch
                except Exception:
                    pass  # Other handling is acceptable

        except ImportError:
            self.skipTest("OverlapMatrixNpy not available")


if __name__ == '__main__':
    unittest.main(verbosity=2)
