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
Ndisp = 1
Nmom = 0

# Calculate num_disp and momentum_list
num_disp = list(GaugeLink.nmax_generator(Ndisp))[-1]
mom_list = mom_dict_to_list(Nmom)
mom_list = [mom_list[0]]

# Test with minimal size
test_usedNe = 2
test_usedNp = 3

# Load one configuration for testing
cfg = 10000
t_test = 0

print(f"Testing _gauge_links_product vs _apply_gauge_links_to_points")
print(f"Configuration: {cfg}")
print(f"Time slice: {t_test}")
print(f"Ndisp: {Ndisp}, num_disp: {num_disp}")
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
    debug=False,  # Disable debug for cleaner output
)

# Load data
print("\nLoading data...")
current_gen.load(cfg)
print("Applying stout smearing...")
current_gen.stout_smear(nstep=20, rho=0.12)

# Get point coordinates
point_data = current_gen._point_data[:test_usedNp]  # [usedNp, Lt, 3]

print(f"\nPoint coordinates at t={t_test}:")
for p in range(test_usedNp):
    print(f"  Point {p}: {point_data[p, t_test, :]}")

print(f"\n" + "="*80)
print(f"Testing gauge link methods consistency")
print(f"="*80)

num_tests = 0
num_pass = 0

# Test each displacement
for disp_idx in range(num_disp):
    gauge_link = GaugeLink(disp_idx)
    gauge_list = gauge_link.gauge_list
    disp = tuple(gauge_link.displacement)
    
    print(f"\n{'='*60}")
    print(f"Displacement {disp_idx}: {disp}")
    print(f"  gauge_list: {gauge_list}")
    print(f"{'='*60}")
    
    # Method 1: Compute full field gauge link product, then extract at points
    print("\nMethod 1: _gauge_links_product + extraction")
    gauge_product_field = current_gen._gauge_links_product(gauge_list, t=t_test)
    
    if gauge_product_field is not None:
        print(f"  gauge_product_field shape: {gauge_product_field.shape}")
        
        # Extract at point positions
        result_method1 = []
        for p in range(test_usedNp):
            x, y, z = point_data[p, t_test, :]
            x, y, z = int(x), int(y), int(z)
            gauge_at_point = gauge_product_field[z, y, x, :, :]  # [Nc, Nc]
            result_method1.append(gauge_at_point)
            print(f"  Point {p} ({x},{y},{z}): gauge_product[{z},{y},{x}] shape {gauge_at_point.shape}")
        
        result_method1 = backend.asarray(result_method1)  # [Np, Nc, Nc]
        print(f"  result_method1 shape: {result_method1.shape}")
    else:
        result_method1 = None
        print(f"  No gauge links (identity)")
    
    # Method 2: Direct computation at points
    print("\nMethod 2: _apply_gauge_links_to_points")
    result_method2 = current_gen._apply_gauge_links_to_points(point_data, gauge_list, t=t_test)
    
    if result_method2 is not None:
        print(f"  result_method2 shape: {result_method2.shape}")
    else:
        print(f"  No gauge links (identity)")
    
    # Compare results
    print("\nComparison:")
    num_tests += 1
    
    if result_method1 is None and result_method2 is None:
        print(f"  ✓ Both methods return None (identity case)")
        num_pass += 1
    elif result_method1 is not None and result_method2 is not None:
        if backend.__name__ == "cupy":
            result1_cpu = result_method1.get()
            result2_cpu = result_method2.get()
        else:
            result1_cpu = result_method1
            result2_cpu = result_method2
        
        diff = np.abs(result1_cpu - result2_cpu)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_val = np.max(np.abs(result1_cpu))
        
        print(f"  Max |result|: {max_val:.4e}")
        print(f"  Max difference: {max_diff:.4e}")
        print(f"  Mean difference: {mean_diff:.4e}")
        print(f"  Relative error: {max_diff / max_val if max_val > 0 else 0:.4e}")
        
        # Print sample values
        print(f"\n  Sample values at point 0:")
        print(f"    Method 1 [0, 0, 0]: {result1_cpu[0, 0, 0]}")
        print(f"    Method 2 [0, 0, 0]: {result2_cpu[0, 0, 0]}")
        
        if max_diff < 1e-10:
            print(f"  ✓ PASS: Methods match exactly")
            num_pass += 1
        elif max_diff / max_val < 1e-6 if max_val > 0 else False:
            print(f"  ~ PASS (with numerical tolerance): Relative error acceptable")
            num_pass += 1
        else:
            print(f"  ✗ FAIL: Methods differ significantly!")
    else:
        print(f"  ✗ FAIL: One method returned None, the other didn't")

print(f"\n{'='*80}")
print(f"Gauge links methods test completed!")
print(f"Results: {num_pass}/{num_tests} tests passed")
if num_pass == num_tests:
    print(f"✓ All tests passed! Both methods are equivalent")
else:
    print(f"✗ Some tests failed! Please check the implementation")
print(f"{'='*80}")

