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
from lattice.generator.elemental import CurrentElementalGenerator
from lattice.insertion.gauge_link import GaugeLink
from lattice.insertion.mom_dict import mom_dict_to_list

set_backend("cupy")
backend = get_backend()

# Test parameters
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
mom_list = mom_dict_to_list(Nmom)[:1]
num_mom = len(mom_list)

# Load one configuration for testing
cfg = 10000
t_test = 0

# Test with minimal size for detailed debugging
test_usedNe = 1
test_usedNp = 1

# Performance test parameters
num_perf_runs = 3  # Number of runs for performance averaging

print(f"Testing calc_all vs individual methods")
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

# Create generator
current_gen = CurrentElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    point=sparsen_point,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    usedNp=test_usedNp,
    debug=False,
)

print("Loading data...")
current_gen.load(cfg)

# Test 1: Individual methods
print("\n" + "=" * 80)
print("Test 1: Computing using individual methods")
print("=" * 80)

# Warm up run
if backend.__name__ == "cupy":
    _ = current_gen.calc_v2v(t_test)
    _ = current_gen.calc_v2p(t_test)
    _ = current_gen.calc_p2v(t_test)
    _ = current_gen.calc_p2p(t_test)
    backend.cuda.runtime.deviceSynchronize()

# Measure performance
time_individual_list = []
for run_idx in range(num_perf_runs):
    time_start = perf_counter()
    v2v_individual = current_gen.calc_v2v(t_test)
    v2p_individual = current_gen.calc_v2p(t_test)
    p2v_individual = current_gen.calc_p2v(t_test)
    p2p_individual = current_gen.calc_p2p(t_test)
    if backend.__name__ == "cupy":
        backend.cuda.runtime.deviceSynchronize()
    time_end = perf_counter()
    time_individual_list.append(time_end - time_start)

time_individual = np.mean(time_individual_list)
time_individual_std = np.std(time_individual_list)
time_individual_min = np.min(time_individual_list)
time_individual_max = np.max(time_individual_list)

print(f"Individual methods execution time: {time_individual:.4f} ± {time_individual_std:.4f} seconds")
print(f"  (min: {time_individual_min:.4f}, max: {time_individual_max:.4f})")
print(f"v2v shape: {v2v_individual.shape}")
print(f"v2p shape: {v2p_individual.shape}")
print(f"p2v shape: {p2v_individual.shape}")
print(f"p2p: {len(p2p_individual)} displacements")

# Test 2: calc_all method
print("\n" + "=" * 80)
print("Test 2: Computing using calc_all method")
print("=" * 80)

# Warm up run
if backend.__name__ == "cupy":
    _ = current_gen.calc_all(t_test)
    backend.cuda.runtime.deviceSynchronize()

# Measure performance
time_all_list = []
for run_idx in range(num_perf_runs):
    time_start = perf_counter()
    results_all = current_gen.calc_all(t_test)
    if backend.__name__ == "cupy":
        backend.cuda.runtime.deviceSynchronize()
    time_end = perf_counter()
    time_all_list.append(time_end - time_start)

time_all = np.mean(time_all_list)
time_all_std = np.std(time_all_list)
time_all_min = np.min(time_all_list)
time_all_max = np.max(time_all_list)

v2v_all = results_all["v2v"]
v2p_all = results_all["v2p"]
p2v_all = results_all["p2v"]
p2p_all = results_all["p2p"]

print(f"calc_all execution time: {time_all:.4f} ± {time_all_std:.4f} seconds")
print(f"  (min: {time_all_min:.4f}, max: {time_all_max:.4f})")
print(f"v2v shape: {v2v_all.shape}")
print(f"v2p shape: {v2p_all.shape}")
print(f"p2v shape: {p2v_all.shape}")
print(f"p2p: {len(p2p_all)} displacements")

# Convert to CPU for comparison
if backend.__name__ == "cupy":
    v2v_individual_cpu = v2v_individual.get()
    v2p_individual_cpu = v2p_individual.get()
    p2v_individual_cpu = p2v_individual.get()
    v2v_all_cpu = v2v_all.get()
    v2p_all_cpu = v2p_all.get()
    p2v_all_cpu = p2v_all.get()
else:
    v2v_individual_cpu = v2v_individual
    v2p_individual_cpu = v2p_individual
    p2v_individual_cpu = p2v_individual
    v2v_all_cpu = v2v_all
    v2p_all_cpu = v2p_all
    p2v_all_cpu = p2v_all

# Compare results
print("\n" + "=" * 80)
print("Comparison: Individual methods vs calc_all")
print("=" * 80)

num_tests = 0
num_pass = 0

# Compare v2v
print("\n=== Comparing v2v ===")
v2v_diff = np.abs(v2v_individual_cpu - v2v_all_cpu)
v2v_max_diff = np.max(v2v_diff)
v2v_mean_diff = np.mean(v2v_diff)
v2v_max_val = np.max(np.abs(v2v_individual_cpu))

print(f"Max |difference|: {v2v_max_diff:.6e}")
print(f"Mean |difference|: {v2v_mean_diff:.6e}")
print(f"Max |value|: {v2v_max_val:.6e}")
if v2v_max_val > 0:
    print(f"Relative error: {v2v_max_diff / v2v_max_val:.6e}")

num_tests += 1
if v2v_max_diff < 1e-10 or (v2v_max_val > 0 and v2v_max_diff / v2v_max_val < 1e-6):
    print("✓ v2v: PASS")
    num_pass += 1
else:
    print("✗ v2v: FAIL")

# Compare v2p
print("\n=== Comparing v2p ===")
v2p_diff = np.abs(v2p_individual_cpu - v2p_all_cpu)
v2p_max_diff = np.max(v2p_diff)
v2p_mean_diff = np.mean(v2p_diff)
v2p_max_val = np.max(np.abs(v2p_individual_cpu))

print(f"Max |difference|: {v2p_max_diff:.6e}")
print(f"Mean |difference|: {v2p_mean_diff:.6e}")
print(f"Max |value|: {v2p_max_val:.6e}")
if v2p_max_val > 0:
    print(f"Relative error: {v2p_max_diff / v2p_max_val:.6e}")

num_tests += 1
if v2p_max_diff < 1e-10 or (v2p_max_val > 0 and v2p_max_diff / v2p_max_val < 1e-6):
    print("✓ v2p: PASS")
    num_pass += 1
else:
    print("✗ v2p: FAIL")

# Compare p2v
print("\n=== Comparing p2v ===")
p2v_diff = np.abs(p2v_individual_cpu - p2v_all_cpu)
p2v_max_diff = np.max(p2v_diff)
p2v_mean_diff = np.mean(p2v_diff)
p2v_max_val = np.max(np.abs(p2v_individual_cpu))

print(f"Max |difference|: {p2v_max_diff:.6e}")
print(f"Mean |difference|: {p2v_mean_diff:.6e}")
print(f"Max |value|: {p2v_max_val:.6e}")
if p2v_max_val > 0:
    print(f"Relative error: {p2v_max_diff / p2v_max_val:.6e}")

num_tests += 1
if p2v_max_diff < 1e-10 or (p2v_max_val > 0 and p2v_max_diff / p2v_max_val < 1e-6):
    print("✓ p2v: PASS")
    num_pass += 1
else:
    print("✗ p2v: FAIL")

# Compare p2p
print("\n=== Comparing p2p ===")
p2p_pass = True
for disp_idx in range(len(p2p_individual)):
    p2p_ind = p2p_individual[disp_idx]
    p2p_a = p2p_all[disp_idx]
    
    if p2p_ind["type"] != p2p_a["type"]:
        print(f"✗ p2p[{disp_idx}]: type mismatch ({p2p_ind['type']} vs {p2p_a['type']})")
        p2p_pass = False
        continue
    
    if p2p_ind["type"] == "identity":
        continue
    
    # Compare sparse format
    if backend.__name__ == "cupy":
        indices_ind = p2p_ind["indices"].get()
        values_ind = p2p_ind["values"].get()
        indices_a = p2p_a["indices"].get()
        values_a = p2p_a["values"].get()
    else:
        indices_ind = p2p_ind["indices"]
        values_ind = p2p_ind["values"]
        indices_a = p2p_a["indices"]
        values_a = p2p_a["values"]
    
    if not np.array_equal(indices_ind, indices_a):
        print(f"✗ p2p[{disp_idx}]: indices mismatch")
        p2p_pass = False
        continue
    
    values_diff = np.abs(values_ind - values_a)
    max_diff = np.max(values_diff)
    max_val = np.max(np.abs(values_ind))
    
    if max_diff > 1e-10 and (max_val == 0 or max_diff / max_val > 1e-6):
        print(f"✗ p2p[{disp_idx}]: values mismatch (max diff: {max_diff:.6e}, max val: {max_val:.6e})")
        p2p_pass = False

num_tests += 1
if p2p_pass:
    print("✓ p2p: PASS")
    num_pass += 1
else:
    print("✗ p2p: FAIL")

# Performance comparison
print("\n" + "=" * 80)
print("Performance Comparison")
print("=" * 80)
print(f"Number of performance runs: {num_perf_runs}")
print(f"Individual methods time: {time_individual:.6f} ± {time_individual_std:.6f} seconds")
print(f"  (min: {time_individual_min:.6f}, max: {time_individual_max:.6f})")
print(f"calc_all time: {time_all:.6f} ± {time_all_std:.6f} seconds")
print(f"  (min: {time_all_min:.6f}, max: {time_all_max:.6f})")

if time_all > 0:
    speedup = time_individual / time_all
    print(f"\nSpeedup: {speedup:.4f}x")
    time_saved = time_individual - time_all
    percent_saved = 100 * time_saved / time_individual
    print(f"Time saved: {time_saved:.6f} seconds ({percent_saved:.2f}%)")

# Final summary
print("\n" + "=" * 80)
print("FINAL TEST SUMMARY")
print("=" * 80)

print(f"\nTest Configuration:")
print(f"  Configuration: {cfg}")
print(f"  Time slice: {t_test}")
print(f"  Lattice size: {L}x{L}x{L}x{T}")
print(f"  Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"  Nmom: {Nmom}, num_mom: {num_mom}")
print(f"  usedNe: {test_usedNe}, usedNp: {test_usedNp}")

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

print(f"\nConclusion:")
if num_pass == num_tests and num_tests > 0:
    print(f"  ✓ calc_all method produces identical results to individual methods")
    print(f"  ✓ calc_all is {speedup:.2f}x faster than calling methods individually")
    print(f"  ✓ Recommended to use calc_all for computing multiple elemental diagrams")
else:
    print(f"  ✗ calc_all method differs from individual methods")
    print(f"  ✗ Please review the implementation for discrepancies")

print(f"\n{'='*80}")
print(f"All tests completed!")
print(f"{'='*80}")

