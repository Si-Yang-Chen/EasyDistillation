import pytest
import numpy as np

pytestmark = pytest.mark.gpu

# Pre-flight GPU check: verify CuPy can access a physical device
try:
    import cupy
    num_devices = cupy.cuda.runtime.getDeviceCount()
    if num_devices == 0:
        raise RuntimeError("No GPU devices found")
except Exception:
    pytest.skip("CuPy/GPU not available", allow_module_level=True)

from lattice import set_backend, get_backend
from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd
from lattice.generator.elemental import ElementalGenerator
from lattice.insertion.gauge_link import GaugeLink
from lattice.insertion.mom_dict import mom_dict_to_list

set_backend("cupy")
backend = get_backend()

# Test parameters
L = 24
T = 72
latt_size = [L, L, L, T]
Lx, Ly, Lz, Lt = latt_size
Ne = 128
Ndisp = 2
Nmom = 0

# Calculate num_disp and momentum_list
num_disp = list(GaugeLink.nmax_generator(Ndisp))[-1]
# mom_list = mom_dict_to_list(Nmom)
# mom_list = mom_list[:2]
mom_list = [(0, 0, 0), (1, 0, 0)]

# Load one configuration for testing
cfg = 10000
t_test = 0

# Test with minimal size for detailed debugging
test_usedNe = 1

print(f"Testing calc and calc_disp consistency")
print(f"Configuration: {cfg}")
print(f"Time slice: {t_test}")
print(f"Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"num_derivative: {(3 ** (Ndisp + 1) - 1) // 2}")
print(f"Nmom: {Nmom}, num_mom: {len(mom_list)}")
print(f"Test usedNe: {test_usedNe}")
print(f"=" * 80)

# Create data loaders
gauge_field = GaugeFieldIldg(
    f"/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}_cfg_",
    ".lime",
    [Lt, Lz, Ly, Lx, Nd, Nc, Nc],
)

eigenvector = EigenvectorNpy(
    f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/02.laplace_eigs.nev{Ne}/",
    ".npy",
    [Lt, Ne, Lz, Ly, Lx, Nc],
    Ne,
)

# Create generator
elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    debug=False,
)

# Load data
print("\nLoading data...")
elemental_gen.load(cfg)

# Compute results using calc method
print("\n" + "=" * 80)
print("Computing using calc method...")
print("=" * 80)
result_calc = elemental_gen.calc(t_test)
print(f"result_calc shape: {result_calc.shape}")

# Convert to numpy if using cupy
if backend.__name__ == "cupy":
    result_calc_np = result_calc.get()
else:
    result_calc_np = result_calc

print(f"result_calc shape: {result_calc_np.shape}")
print(f"result_calc dtype: {result_calc_np.dtype}")

# Print some sample values
print(f"\nSample values from calc:")
for deriv_idx in range(min(5, result_calc_np.shape[0])):
    for mom_idx in range(result_calc_np.shape[1]):
        print(f"  deriv_idx={deriv_idx}, mom_idx={mom_idx}:")
        print(
            f"    result_calc[{deriv_idx}, {mom_idx}, 0, 0] = {result_calc_np[deriv_idx, mom_idx, 0, 0]}"
        )

# Compute results using calc_disp method
print("\n" + "=" * 80)
print("Computing using calc_disp method...")
print("=" * 80)
result_calc_disp = elemental_gen.calc_disp(t_test)
print(f"result_calc_disp shape: {result_calc_disp.shape}")

# Convert to numpy if using cupy
if backend.__name__ == "cupy":
    result_calc_disp_np = result_calc_disp.get()
else:
    result_calc_disp_np = result_calc_disp

print(f"result_calc_disp shape: {result_calc_disp_np.shape}")
print(f"result_calc_disp dtype: {result_calc_disp_np.dtype}")

# Print some sample values
print(f"\nSample values from calc_disp:")
for disp_idx in range(min(5, result_calc_disp_np.shape[0])):
    for mom_idx in range(result_calc_disp_np.shape[1]):
        print(f"  disp_idx={disp_idx}, mom_idx={mom_idx}:")
        print(
            f"    result_calc_disp[{disp_idx}, {mom_idx}, 0, 0] = {result_calc_disp_np[disp_idx, mom_idx, 0, 0]}"
        )

# Print mapping information
print("\n" + "=" * 80)
print("Mapping information:")
print("=" * 80)
print(f"calc uses derivative_list with {elemental_gen.num_derivative} derivatives")
print(f"calc_disp uses GaugeLink with {num_disp} displacements")
print(f"\nDerivative to displacement mapping:")

# Build mapping from derivative to displacement
from lattice.insertion.derivative import derivative

for deriv_idx in range(elemental_gen.num_derivative):
    deriv = derivative(deriv_idx)
    print(f"  deriv_idx={deriv_idx}, derivative={deriv}")

print(f"\nDisplacement information:")
for disp_idx in range(num_disp):
    gauge_link = GaugeLink(disp_idx)
    print(
        f"  disp_idx={disp_idx}, displacement={gauge_link.displacement}, gauge_list={gauge_link.gauge_list}"
    )

# Output results for comparison
print("\n" + "=" * 80)
print("Output results for comparison:")
print("=" * 80)
print(f"\nresult_calc (full array):")
print(f"  Shape: {result_calc_np.shape}")

print(f"\nresult_calc_disp (full array):")
print(f"  Shape: {result_calc_disp_np.shape}")

# Comparison: result_calc_np[i] = result_calc_disp_np[i] - result_calc_disp_np[i+3], i=1,2,3
print("\n" + "=" * 80)
print(
    "Comparison: result_calc_np[i] = -2 * (result_calc_disp_np[i] - result_calc_disp_np[i+3]), i=1,2,3"
)
print("=" * 80)

# Check array bounds
max_deriv_idx = result_calc_np.shape[0] - 1
max_disp_idx = result_calc_disp_np.shape[0] - 1

print(f"\nArray bounds check:")
print(
    f"  result_calc_np shape[0] = {result_calc_np.shape[0]}, max index = {max_deriv_idx}"
)
print(
    f"  result_calc_disp_np shape[0] = {result_calc_disp_np.shape[0]}, max index = {max_disp_idx}"
)

# Test indices i=1,2,3
test_indices = [1, 2, 3]
num_tests = 0
num_pass = 0
num_tests_mom1 = 0
num_pass_mom1 = 0

for i in test_indices:
    if i > max_deriv_idx:
        print(f"\n  Skipping i={i}: result_calc_np index out of bounds")
        continue

    if i > max_disp_idx:
        print(f"\n  Skipping i={i}: result_calc_disp_np[{i}] index out of bounds")
        continue

    if i + 3 > max_disp_idx:
        print(f"\n  Skipping i={i}: result_calc_disp_np[{i+3}] index out of bounds")
        continue

    num_tests += 1

    # Extract values for mom_idx=0
    calc_value = result_calc_np[i, 0]
    disp_i = result_calc_disp_np[i, 0]
    disp_i_plus_3 = result_calc_disp_np[i + 3, 0]
    expected_value = -2 * (disp_i - disp_i_plus_3)

    # Calculate difference
    diff = calc_value - expected_value
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    # Calculate relative error
    max_val = np.max(np.abs(calc_value))
    rel_error = max_diff / max_val if max_val > 0 else 0

    print(f"\n  Test i={i}, mom_idx=0:")
    print(f"    result_calc_np[{i}, 0] shape: {calc_value.shape}")
    print(f"    result_calc_disp_np[{i}, 0] shape: {disp_i.shape}")
    print(f"    result_calc_disp_np[{i+3}, 0] shape: {disp_i_plus_3.shape}")
    print(
        f"    Expected: -2 * (result_calc_disp_np[{i}, 0] - result_calc_disp_np[{i+3}, 0])"
    )

    # Print sample values
    if len(calc_value.shape) >= 2:
        print(f"    result_calc_np[{i}, 0, 0, 0] = {calc_value[0, 0]}")
        print(f"    result_calc_disp_np[{i}, 0, 0, 0] = {disp_i[0, 0]}")
        print(f"    result_calc_disp_np[{i+3}, 0, 0, 0] = {disp_i_plus_3[0, 0]}")
        print(f"    Expected[{i}, 0, 0, 0] = {expected_value[0, 0]}")
        print(f"    Difference[{i}, 0, 0, 0] = {diff[0, 0]}")

    print(f"    Max |difference|: {max_diff:.6e}")
    print(f"    Mean |difference|: {mean_diff:.6e}")
    print(f"    Max |result_calc_np[{i}, 0]|: {max_val:.6e}")
    print(f"    Relative error: {rel_error:.6e}")

    # Check if they match (strict tolerance for mom_idx=0)
    tolerance = 1e-10
    if max_diff < tolerance:
        print(
            f"    ✓ PASS: result_calc_np[{i}, 0] = -2 * (result_calc_disp_np[{i}, 0] - result_calc_disp_np[{i+3}, 0])"
        )
        num_pass += 1
    elif rel_error < 1e-6:
        print(f"    ~ PASS (with numerical tolerance): Relative error acceptable")
        num_pass += 1
    else:
        print(f"    ✗ FAIL: Difference too large!")

    # Test for mom_idx=1 if available (with relaxed tolerance)
    if result_calc_np.shape[1] > 1:
        num_tests_mom1 += 1

        calc_value_mom1 = result_calc_np[i, 1]
        disp_i_mom1 = result_calc_disp_np[i, 1]
        disp_i_plus_3_mom1 = result_calc_disp_np[i + 3, 1]
        displacement_i = GaugeLink(i).displacement
        displacement_i_plus_3 = GaugeLink(i + 3).displacement
        expected_value_mom1 = -2 * (
            disp_i_mom1
            * np.cos(
                -1
                * np.pi
                * sum(
                    [
                        displacement_i[i] * mom_list[1][i] / latt_size[i]
                        for i in range(3)
                    ]
                )
            )
            - disp_i_plus_3_mom1
            * np.cos(
                1
                * np.pi
                * sum(
                    [
                        displacement_i_plus_3[i] * mom_list[1][i] / latt_size[i]
                        for i in range(3)
                    ]
                )
            )
        )

        # Calculate difference
        diff_mom1 = calc_value_mom1 - expected_value_mom1
        max_diff_mom1 = np.max(np.abs(diff_mom1))
        mean_diff_mom1 = np.mean(np.abs(diff_mom1))

        # Calculate relative error
        max_val_mom1 = np.max(np.abs(calc_value_mom1))
        rel_error_mom1 = max_diff_mom1 / max_val_mom1 if max_val_mom1 > 0 else 0

        print(f"\n  Test i={i}, mom_idx=1 (relaxed tolerance):")
        print(f"    result_calc_np[{i}, 1] shape: {calc_value_mom1.shape}")
        print(f"    result_calc_disp_np[{i}, 1] shape: {disp_i_mom1.shape}")
        print(f"    result_calc_disp_np[{i+3}, 1] shape: {disp_i_plus_3_mom1.shape}")
        print(f"    displacement[{i}]: {displacement_i}")
        print(f"    displacement[{i+3}]: {displacement_i_plus_3}")
        print(
            f"    Expected: -2 * (disp_i * cos(phase_i) - disp_i_plus_3 * cos(phase_i_plus_3))"
        )
        print(f"      where phase includes momentum {mom_list[1]} and displacement")

        # Print sample values
        if len(calc_value_mom1.shape) >= 2:
            print(f"    result_calc_np[{i}, 1, 0, 0] = {calc_value_mom1[0, 0]}")
            print(f"    result_calc_disp_np[{i}, 1, 0, 0] = {disp_i_mom1[0, 0]}")
            print(
                f"    result_calc_disp_np[{i+3}, 1, 0, 0] = {disp_i_plus_3_mom1[0, 0]}"
            )
            print(f"    Expected[{i}, 1, 0, 0] = {expected_value_mom1[0, 0]}")
            print(f"    Difference[{i}, 1, 0, 0] = {diff_mom1[0, 0]}")

        print(f"    Max |difference|: {max_diff_mom1:.6e}")
        print(f"    Mean |difference|: {mean_diff_mom1:.6e}")
        print(f"    Max |result_calc_np[{i}, 1]|: {max_val_mom1:.6e}")
        print(f"    Relative error: {rel_error_mom1:.6e}")

        # Check if they match (relaxed tolerance for mom_idx=1: 1/100 = 0.01)
        tolerance_mom1 = 1e-10
        relaxed_rel_tolerance = 0.01  # 1/10
        if max_diff_mom1 < tolerance_mom1:
            print(
                f"    ✓ PASS: result_calc_np[{i}, 1] matches expected value with phase factors"
            )
            num_pass_mom1 += 1
        elif rel_error_mom1 < relaxed_rel_tolerance:
            print(
                f"    ~ PASS (with relaxed tolerance 1/100): Relative error acceptable"
            )
            num_pass_mom1 += 1
        else:
            print(
                f"    ✗ FAIL: Difference too large (rel error = {rel_error_mom1:.6e} > {relaxed_rel_tolerance})"
            )

print(f"\n{'='*80}")
print(f"Comparison test completed!")
print(f"Results: {num_pass}/{num_tests} tests passed")
print(f"{'='*80}")

# Additional test: result_calc_np[4] = 4 * (result_calc_disp_np[7] + result_calc_disp_np[24] - 2 * result_calc_disp_np[0])
print("\n" + "=" * 80)
print(
    "Additional test: result_calc_np[4] = 4 * (result_calc_disp_np[7] + result_calc_disp_np[24] - 2 * result_calc_disp_np[0])"
)
print("=" * 80)

test_idx = 4
disp_idx_7 = 7
disp_idx_24 = 24
disp_idx_0 = 0

# Track second test result
second_test_passed = False
second_test_skipped = False
# Note: Test 2 only tests mom_idx=0, mom_idx=1 test is not implemented

# Check bounds
if test_idx > max_deriv_idx:
    print(
        f"\n  Skipping: result_calc_np[{test_idx}] index out of bounds (max={max_deriv_idx})"
    )
    second_test_skipped = True
elif disp_idx_7 > max_disp_idx:
    print(
        f"\n  Skipping: result_calc_disp_np[{disp_idx_7}] index out of bounds (max={max_disp_idx})"
    )
    second_test_skipped = True
elif disp_idx_24 > max_disp_idx:
    print(
        f"\n  Skipping: result_calc_disp_np[{disp_idx_24}] index out of bounds (max={max_disp_idx})"
    )
    second_test_skipped = True
elif disp_idx_0 > max_disp_idx:
    print(
        f"\n  Skipping: result_calc_disp_np[{disp_idx_0}] index out of bounds (max={max_disp_idx})"
    )
    second_test_skipped = True
else:
    # Extract values for mom_idx=0
    calc_value = result_calc_np[test_idx, 0]
    disp_7 = result_calc_disp_np[disp_idx_7, 0]
    disp_24 = result_calc_disp_np[disp_idx_24, 0]
    disp_0 = result_calc_disp_np[disp_idx_0, 0]
    expected_value = 4 * (disp_7 + disp_24 - 2 * disp_0)

    # Calculate difference
    diff = calc_value - expected_value
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))

    # Calculate relative error
    max_val = np.max(np.abs(calc_value))
    rel_error = max_diff / max_val if max_val > 0 else 0

    print(f"\n  Test result_calc_np[{test_idx}, 0]:")
    print(f"    result_calc_np[{test_idx}, 0] shape: {calc_value.shape}")
    print(f"    result_calc_disp_np[{disp_idx_7}, 0] shape: {disp_7.shape}")
    print(f"    result_calc_disp_np[{disp_idx_24}, 0] shape: {disp_24.shape}")
    print(f"    result_calc_disp_np[{disp_idx_0}, 0] shape: {disp_0.shape}")
    print(
        f"    Expected: 4* (result_calc_disp_np[{disp_idx_7}, 0] + result_calc_disp_np[{disp_idx_24}, 0] - 2 * result_calc_disp_np[{disp_idx_0}, 0])"
    )

    # Print sample values
    if len(calc_value.shape) >= 2:
        print(f"    result_calc_np[{test_idx}, 0, 0, 0] = {calc_value[0, 0]}")
        print(f"    result_calc_disp_np[{disp_idx_7}, 0, 0, 0] = {disp_7[0, 0]}")
        print(f"    result_calc_disp_np[{disp_idx_24}, 0, 0, 0] = {disp_24[0, 0]}")
        print(f"    result_calc_disp_np[{disp_idx_0}, 0, 0, 0] = {disp_0[0, 0]}")
        print(f"    Expected[{test_idx}, 0, 0, 0] = {expected_value[0, 0]}")
        print(f"    Difference[{test_idx}, 0, 0, 0] = {diff[0, 0]}")

    print(f"    Max |difference|: {max_diff:.6e}")
    print(f"    Mean |difference|: {mean_diff:.6e}")
    print(f"    Max |result_calc_np[{test_idx}, 0]|: {max_val:.6e}")
    print(f"    Relative error: {rel_error:.6e}")

    # Check if they match (for mom_idx=0)
    tolerance = 1e-10
    if max_diff < tolerance:
        print(
            f"    ✓ PASS: result_calc_np[{test_idx}, 0] = 4 * (result_calc_disp_np[{disp_idx_7}, 0] + result_calc_disp_np[{disp_idx_24}, 0] - 2 * result_calc_disp_np[{disp_idx_0}, 0])"
        )
        second_test_passed = True
    elif rel_error < 1e-6:
        print(f"    ~ PASS (with numerical tolerance): Relative error acceptable")
        second_test_passed = True
    else:
        print(f"    ✗ FAIL: Difference too large!")


# Final summary
print(f"\n{'='*80}")
print(f"FINAL TEST SUMMARY")
print(f"{'='*80}")
print(f"\nTest Configuration:")
print(f"  Configuration: {cfg}")
print(f"  Time slice: {t_test}")
print(f"  Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"  num_derivative: {(3 ** (Ndisp + 1) - 1) // 2}")
print(f"  usedNe: {test_usedNe}")
print(f"  Array shapes:")
print(f"    result_calc_np: {result_calc_np.shape}")
print(f"    result_calc_disp_np: {result_calc_disp_np.shape}")

print(
    f"\nTest 1: result_calc_np[i] = -2 * (result_calc_disp_np[i] - result_calc_disp_np[i+3]), i=1,2,3"
)
print(f"  Tests performed (mom_idx=0): {num_tests}")
print(f"  Tests passed (mom_idx=0): {num_pass}")
if num_tests > 0:
    print(
        f"  Pass rate (mom_idx=0): {num_pass}/{num_tests} ({100*num_pass/num_tests:.1f}%)"
    )
    if num_pass == num_tests:
        print(f"  Status (mom_idx=0): ✓ ALL PASSED")
    else:
        print(f"  Status (mom_idx=0): ✗ SOME FAILED")
else:
    print(f"  Status (mom_idx=0): - NO TESTS PERFORMED (index out of bounds)")

if num_tests_mom1 > 0:
    print(
        f"\n  Tests performed (mom_idx=1, relaxed tolerance 1/100, with phase factors): {num_tests_mom1}"
    )
    print(f"  Tests passed (mom_idx=1): {num_pass_mom1}")
    print(
        f"  Pass rate (mom_idx=1): {num_pass_mom1}/{num_tests_mom1} ({100*num_pass_mom1/num_tests_mom1:.1f}%)"
    )
    if num_pass_mom1 == num_tests_mom1:
        print(f"  Status (mom_idx=1): ✓ ALL PASSED")
    else:
        print(f"  Status (mom_idx=1): ✗ SOME FAILED")

print(
    f"\nTest 2: result_calc_np[4] = 4 * (result_calc_disp_np[7] + result_calc_disp_np[24] - 2 * result_calc_disp_np[0])"
)
if second_test_skipped:
    print(f"  Status (mom_idx=0): - SKIPPED (index out of bounds)")
elif second_test_passed:
    print(f"  Status (mom_idx=0): ✓ PASSED")
else:
    print(f"  Status (mom_idx=0): ✗ FAILED")

# Note: Test 2 only tests mom_idx=0, mom_idx=1 test is not implemented
# if not second_test_skipped and result_calc_np.shape[1] > 1:
#     if second_test_passed_mom1:
#         print(f"  Status (mom_idx=1, relaxed tolerance): ✓ PASSED")
#     else:
#         print(f"  Status (mom_idx=1, relaxed tolerance): ✗ FAILED")

print(f"\nOverall Result:")
# Test 1: num_tests (mom_idx=0) + num_tests_mom1 (mom_idx=1)
# Test 2: 1 test (mom_idx=0 only, if not skipped)
total_tests = (
    num_tests  # Test 1, mom_idx=0
    + num_tests_mom1  # Test 1, mom_idx=1
    + (0 if second_test_skipped else 1)  # Test 2, mom_idx=0
)
total_passed = (
    num_pass  # Test 1, mom_idx=0
    + num_pass_mom1  # Test 1, mom_idx=1
    + (1 if second_test_passed else 0)  # Test 2, mom_idx=0
)
if total_tests > 0:
    print(f"  Total tests: {total_tests}")
    print(f"    - Test 1, mom_idx=0: {num_tests} tests")
    if num_tests_mom1 > 0:
        print(f"    - Test 1, mom_idx=1 (relaxed tolerance): {num_tests_mom1} tests")
    if not second_test_skipped:
        print(f"    - Test 2, mom_idx=0: 1 test")
    print(f"  Total passed: {total_passed}")
    print(
        f"  Overall pass rate: {total_passed}/{total_tests} ({100*total_passed/total_tests:.1f}%)"
    )
    if total_passed == total_tests:
        print(f"  ✓ ALL TESTS PASSED - calc and calc_disp are consistent!")
    else:
        print(f"  ✗ SOME TESTS FAILED - check consistency between calc and calc_disp")
else:
    print(f"  - NO TESTS PERFORMED")

print(f"\n{'='*80}")
print(f"All tests completed!")
print(f"{'='*80}")
