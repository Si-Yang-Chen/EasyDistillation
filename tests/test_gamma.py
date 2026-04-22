import unittest
import sys
import os

# Add project root directory to Python path
test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from lattice.insertion.gamma import *

        


if __name__ == "__main__":
    gamma_transform_dict=genGammaTransformDict()
    print(gamma_transform_dict)
    # unittest.main()
    # single test
    # suite = unittest.TestSuite()
    # # suite.addTest(TestGaugeLink("test_string_initialization"))
    # suite.addTest(TestGaugeLink("test_idx_initialization"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
