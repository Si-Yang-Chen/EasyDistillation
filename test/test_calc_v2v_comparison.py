import pytest
from time import perf_counter
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
from lattice import GaugeFieldIldg, EigenvectorNpy, PointSourceNpy, Nc, Nd
from lattice.generator.elemental import ElementalGenerator, CurrentElementalGenerator
from lattice.insertion.gauge_link import GaugeLink
from lattice.insertion.mom_dict import mom_dict_to_list

set_backend("cupy")
backend = get_backend()

# Test parameters - match 3.gen_elemental.py
L = 24
T = 72
latt_size = [L, L, L, T]
Lx, Ly, Lz, Lt = latt_size
Np = 6**3
Ne = 128
Ndisp = 2
Nmom = 0

# Calculate num_disp and momentum_list
num_disp = list(GaugeLink.nmax_generator(Ndisp))[-1]
mom_list = mom_dict_to_list(Nmom)[:2]

num_mom = len(mom_list)

# Load one configuration for testing
cfg = 10000
t_test = 0

# Test with minimal size for detailed debugging
test_usedNe = 1
test_usedNp = 1

# Performance test parameters
num_perf_runs = 1  # Number of runs for performance averaging

# Numerical comparison tolerance settings
# Due to floating-point operation order differences between calc_disp and calc_v2v:
# - calc_disp: applies gauge links sequentially to eigenvectors
# - calc_v2v: pre-computes gauge link products then applies to eigenvectors
# These lead to small numerical differences due to non-associativity of floating-point arithmetic
ABSOLUTE_TOLERANCE = 1e-8  # Absolute error tolerance
RELATIVE_TOLERANCE = 1e-4  # Relative error tolerance (0.01%)
TINY_VALUE_THRESHOLD = 1e-12  # Values below this are considered essentially zero

print(f"Testing calc_v2v vs calc_disp")
print(f"Configuration: {cfg}")
print(f"Time slice: {t_test}")
print(f"Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"Nmom: {Nmom}, num_mom: {num_mom}")
print(f"Test usedNe: {test_usedNe}, usedNp: {test_usedNp}")
print(f"=" * 80)

# Create data loaders
sparsen_point_dir = f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/01.sparsened_field"
sparsen_point = PointSourceNpy(f"{sparsen_point_dir}/", ".npy", [Np, Lt, 3], Np)

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

# Test 1: Original ElementalGenerator.calc_disp
print("\n" + "=" * 80)
print("Test 1: ElementalGenerator.calc_disp (original method)")
print("=" * 80)

elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    debug=True,
)

print("Loading data...")
elemental_gen.load(cfg)
# print("Applying stout smearing...")
# elemental_gen.stout_smear(nstep=20, rho=0.12)

print("\nComputing v2v using calc_disp...")
# Warm up run to ensure GPU is ready (if using GPU)
if backend.__name__ == "cupy":
    _ = elemental_gen.calc_disp(t_test)
    backend.cuda.runtime.deviceSynchronize()

# Measure performance with multiple runs
time_calc_disp_list = []
for run_idx in range(num_perf_runs):
    time_start = perf_counter()
    v2v_original = elemental_gen.calc_disp(
        t_test
    )  # [num_disp, num_momentum, usedNe, usedNe]
    if backend.__name__ == "cupy":
        backend.cuda.runtime.deviceSynchronize()
    time_end = perf_counter()
    time_calc_disp_list.append(time_end - time_start)

time_calc_disp = np.mean(time_calc_disp_list)
time_calc_disp_std = np.std(time_calc_disp_list)
time_calc_disp_min = np.min(time_calc_disp_list)
time_calc_disp_max = np.max(time_calc_disp_list)

print(f"v2v_original shape: {v2v_original.shape}")
print(
    f"calc_disp execution time: {time_calc_disp:.4f} ± {time_calc_disp_std:.4f} seconds (min: {time_calc_disp_min:.4f}, max: {time_calc_disp_max:.4f})"
)
if backend.__name__ == "cupy":
    print(f"v2v_original[0, 0, 0, 0] = {v2v_original[0, 0, 0, 0].get()}")
else:
    print(f"v2v_original[0, 0, 0, 0] = {v2v_original[0, 0, 0, 0]}")

# Test 2: New CurrentElementalGenerator.calc_v2v
print("\n" + "=" * 80)
print("Test 2: CurrentElementalGenerator.calc_v2v (new method)")
print("=" * 80)

current_gen = CurrentElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    point=sparsen_point,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    usedNp=test_usedNp,
    debug=True,
)

print("Loading data...")
current_gen.load(cfg)
# print("Applying stout smearing...")
# current_gen.stout_smear(nstep=20, rho=0.12)

print("\nComputing v2v using calc_v2v...")
# Warm up run to ensure GPU is ready (if using GPU)
if backend.__name__ == "cupy":
    _ = current_gen.calc_v2v(t_test)
    backend.cuda.runtime.deviceSynchronize()

# Measure performance with multiple runs
time_calc_v2v_list = []
for run_idx in range(num_perf_runs):
    time_start = perf_counter()
    v2v_new = current_gen.calc_v2v(t_test)  # [num_disp, num_momentum, usedNe, usedNe]
    if backend.__name__ == "cupy":
        backend.cuda.runtime.deviceSynchronize()
    time_end = perf_counter()
    time_calc_v2v_list.append(time_end - time_start)

time_calc_v2v = np.mean(time_calc_v2v_list)
time_calc_v2v_std = np.std(time_calc_v2v_list)
time_calc_v2v_min = np.min(time_calc_v2v_list)
time_calc_v2v_max = np.max(time_calc_v2v_list)

print(f"v2v_new shape: {v2v_new.shape}")
print(
    f"calc_v2v execution time: {time_calc_v2v:.4f} ± {time_calc_v2v_std:.4f} seconds (min: {time_calc_v2v_min:.4f}, max: {time_calc_v2v_max:.4f})"
)
if backend.__name__ == "cupy":
    print(f"v2v_new[0, 0, 0, 0] = {v2v_new[0, 0, 0, 0].get()}")
else:
    print(f"v2v_new[0, 0, 0, 0] = {v2v_new[0, 0, 0, 0]}")

# Compare results
print("\n" + "=" * 80)
print("Comparison: calc_disp vs calc_v2v")
print("=" * 80)

if backend.__name__ == "cupy":
    v2v_original_cpu = v2v_original.get()
    v2v_new_cpu = v2v_new.get()
else:
    v2v_original_cpu = v2v_original
    v2v_new_cpu = v2v_new

num_tests = 0
num_pass = 0
failed_tests = []  # Store failed test information

for disp_idx in range(v2v_original_cpu.shape[0]):
    for momentum_idx in range(v2v_original_cpu.shape[1]):
        diff = np.abs(
            v2v_original_cpu[disp_idx, momentum_idx]
            - v2v_new_cpu[disp_idx, momentum_idx]
        )
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_val = np.max(np.abs(v2v_original_cpu[disp_idx, momentum_idx]))

        gauge_link = GaugeLink(disp_idx)
        disp = tuple(gauge_link.displacement)  # Convert to tuple
        momentum = mom_list[momentum_idx]

        num_tests += 1

        print(f"\nDisplacement {disp_idx}: {disp}, momentum {momentum_idx}: {momentum}")
        print(f"  gauge_list: {gauge_link.gauge_list}")
        print(f"  v2v_original[0,0]: {v2v_original_cpu[disp_idx, momentum_idx, 0, 0]}")
        print(f"  v2v_new[0,0]: {v2v_new_cpu[disp_idx, momentum_idx, 0, 0]}")
        print(f"  Max |v2v|: {max_val:.4e}")
        print(f"  Max difference: {max_diff:.4e}")
        print(f"  Mean difference: {mean_diff:.4e}")
        print(f"  Relative error: {max_diff / max_val if max_val > 0 else 0:.4e}")

        # Determine if test passes
        rel_error = max_diff / max_val if max_val > 0 else 0

        # For very small values, only check absolute error
        if max_val < TINY_VALUE_THRESHOLD:
            if max_diff < ABSOLUTE_TOLERANCE:
                print(
                    f"  ✓ PASS: Values near zero, absolute error < {ABSOLUTE_TOLERANCE:.0e}"
                )
                num_pass += 1
            else:
                print(f"  ✗ FAIL: Values near zero but absolute error too large")
                failed_tests.append(
                    {
                        "disp_idx": disp_idx,
                        "displacement": disp,
                        "momentum_idx": momentum_idx,
                        "momentum": momentum,
                        "gauge_list": gauge_link.gauge_list,
                        "max_diff": max_diff,
                        "mean_diff": mean_diff,
                        "max_val": max_val,
                        "rel_error": rel_error,
                    }
                )
        # For normal values, check both absolute and relative error
        elif max_diff < ABSOLUTE_TOLERANCE:
            print(f"  ✓ PASS: Absolute error < {ABSOLUTE_TOLERANCE:.0e}")
            num_pass += 1
        elif rel_error < RELATIVE_TOLERANCE:
            print(
                f"  ✓ PASS: Relative error < {RELATIVE_TOLERANCE:.0e} ({rel_error*100:.3f}%)"
            )
            num_pass += 1
        else:
            print(
                f"  ✗ FAIL: Errors exceed tolerance (abs: {ABSOLUTE_TOLERANCE:.0e}, rel: {RELATIVE_TOLERANCE:.0e})"
            )
            print(
                f"  Sample v2v_original[0,0]: {v2v_original_cpu[disp_idx, momentum_idx, 0, 0]}"
            )
            print(f"  Sample v2v_new[0,0]: {v2v_new_cpu[disp_idx, momentum_idx, 0, 0]}")
            failed_tests.append(
                {
                    "disp_idx": disp_idx,
                    "displacement": disp,
                    "momentum_idx": momentum_idx,
                    "momentum": momentum,
                    "gauge_list": gauge_link.gauge_list,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "max_val": max_val,
                    "rel_error": rel_error,
                }
            )

print(f"\n{'='*80}")
print(f"Comparison test completed!")
print(f"Results: {num_pass}/{num_tests} tests passed")
if num_pass == num_tests:
    print(f"✓ All tests passed! calc_v2v is equivalent to calc_disp")
else:
    print(f"✗ Some tests failed! Please check the implementation")
print(f"{'='*80}")

# Final summary
print(f"\n{'='*80}")
print(f"FINAL TEST SUMMARY")
print(f"{'='*80}")

print(f"\nTest Configuration:")
print(f"  Configuration: {cfg}")
print(f"  Time slice: {t_test}")
print(f"  Lattice size: {L}x{L}x{L}x{T}")
print(f"  Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"  Nmom: {Nmom}, num_mom: {num_mom}")
print(f"  usedNe: {test_usedNe}, usedNp: {test_usedNp}")
print(f"  Array shapes:")
print(f"    v2v_original: {v2v_original_cpu.shape}")
print(f"    v2v_new: {v2v_new_cpu.shape}")

print(f"\nNumerical Tolerance Settings:")
print(f"  Absolute tolerance: {ABSOLUTE_TOLERANCE:.0e}")
print(f"  Relative tolerance: {RELATIVE_TOLERANCE:.0e} ({RELATIVE_TOLERANCE*100:.3f}%)")
print(f"  Tiny value threshold: {TINY_VALUE_THRESHOLD:.0e}")

print(f"\nTest Results:")
print(f"  Total tests: {num_tests}")
print(f"  Tests passed: {num_pass}")
if num_tests > 0:
    pass_rate = 100 * num_pass / num_tests
    print(f"  Pass rate: {num_pass}/{num_tests} ({pass_rate:.1f}%)")
    if num_pass == num_tests:
        print(f"  Status: ✓ ALL PASSED")
    else:
        print(f"  Status: ✗ SOME FAILED")
else:
    print(f"  Status: - NO TESTS PERFORMED")

print(f"\nSample Element Values:")
print(f"  v2v_original[0, 0, 0, 0] = {v2v_original_cpu[0, 0, 0, 0]}")
print(f"  v2v_new[0, 0, 0, 0] = {v2v_new_cpu[0, 0, 0, 0]}")
if num_tests > 0:
    diff_sample = np.abs(v2v_original_cpu[0, 0, 0, 0] - v2v_new_cpu[0, 0, 0, 0])
    print(f"  Difference[0, 0, 0, 0] = {diff_sample:.6e}")
    max_val_sample = np.max(np.abs(v2v_original_cpu[0, 0, 0, 0]))
    if max_val_sample > 0:
        rel_error_sample = diff_sample / max_val_sample
        print(f"  Relative error[0, 0, 0, 0] = {rel_error_sample:.6e}")

print(f"\nOverall Statistics:")
if num_tests > 0:
    all_diff = np.abs(v2v_original_cpu - v2v_new_cpu)
    max_diff_overall = np.max(all_diff)
    mean_diff_overall = np.mean(all_diff)
    max_val_overall = np.max(np.abs(v2v_original_cpu))
    print(f"  Max |difference| (all elements): {max_diff_overall:.6e}")
    print(f"  Mean |difference| (all elements): {mean_diff_overall:.6e}")
    print(f"  Max |v2v_original| (all elements): {max_val_overall:.6e}")
    if max_val_overall > 0:
        rel_error_overall = max_diff_overall / max_val_overall
        print(f"  Max relative error: {rel_error_overall:.6e}")

print(f"\nPerformance Comparison:")
print(f"  Number of performance runs: {num_perf_runs}")
print(
    f"  calc_disp execution time: {time_calc_disp:.6f} ± {time_calc_disp_std:.6f} seconds"
)
print(f"    (min: {time_calc_disp_min:.6f}, max: {time_calc_disp_max:.6f})")
print(
    f"  calc_v2v execution time: {time_calc_v2v:.6f} ± {time_calc_v2v_std:.6f} seconds"
)
print(f"    (min: {time_calc_v2v_min:.6f}, max: {time_calc_v2v_max:.6f})")
if time_calc_disp > 0:
    speedup = time_calc_disp / time_calc_v2v
    slowdown = time_calc_v2v / time_calc_disp
    if speedup > 1:
        print(f"  calc_v2v is {speedup:.4f}x faster than calc_disp")
    elif slowdown > 1:
        print(f"  calc_v2v is {slowdown:.4f}x slower than calc_disp")
    else:
        print(f"  Both methods have similar performance")
    time_diff = abs(time_calc_disp - time_calc_v2v)
    time_diff_percent = 100 * time_diff / max(time_calc_disp, time_calc_v2v)
    print(f"  Time difference: {time_diff:.6f} seconds ({time_diff_percent:.2f}%)")
    # Calculate speedup using min times for best case comparison
    speedup_min = time_calc_disp_min / time_calc_v2v_min if time_calc_v2v_min > 0 else 0
    speedup_max = time_calc_disp_max / time_calc_v2v_max if time_calc_v2v_max > 0 else 0
    if speedup_min > 1 or speedup_max > 1:
        print(
            f"  Speedup range: {min(speedup_min, speedup_max):.4f}x - {max(speedup_min, speedup_max):.4f}x"
        )
    elif slowdown > 1:
        slowdown_min = (
            time_calc_v2v_min / time_calc_disp_min if time_calc_disp_min > 0 else 0
        )
        slowdown_max = (
            time_calc_v2v_max / time_calc_disp_max if time_calc_disp_max > 0 else 0
        )
        print(
            f"  Slowdown range: {min(slowdown_min, slowdown_max):.4f}x - {max(slowdown_min, slowdown_max):.4f}x"
        )

print(f"\nFailed Tests Details:")
if len(failed_tests) > 0:
    print(f"  Number of failed tests: {len(failed_tests)}")
    print(f"  {'='*76}")
    for idx, fail_info in enumerate(failed_tests, 1):
        print(f"\n  Failed Test #{idx}:")
        print(f"    Displacement index: {fail_info['disp_idx']}")
        print(f"    Displacement: {fail_info['displacement']}")
        print(f"    Gauge list: {fail_info['gauge_list']}")
        print(f"    Momentum index: {fail_info['momentum_idx']}")
        print(f"    Momentum: {fail_info['momentum']}")
        print(f"    Max difference: {fail_info['max_diff']:.6e}")
        print(f"    Mean difference: {fail_info['mean_diff']:.6e}")
        print(f"    Max |v2v|: {fail_info['max_val']:.6e}")
        print(f"    Relative error: {fail_info['rel_error']:.6e}")
else:
    print(f"  No failed tests - all tests passed!")

print(f"\nConclusion:")
if num_pass == num_tests and num_tests > 0:
    print(f"  ✓ calc_v2v method is equivalent to calc_disp method")
    print(f"  ✓ Both methods produce identical results within numerical tolerance")
else:
    print(f"  ✗ calc_v2v method differs from calc_disp method")
    print(f"  ✗ Please review the implementation for discrepancies")
    if len(failed_tests) > 0:
        print(f"  ✗ Failed test count: {len(failed_tests)}/{num_tests}")
        print(f"  ✗ See 'Failed Tests Details' section above for specific failures")

print(f"\n{'='*80}")
print(f"All tests completed!")
print(f"{'='*80}")
