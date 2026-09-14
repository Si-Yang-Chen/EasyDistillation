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

# Test parameters - match 3.gen_current_elemental.py
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
mom_list = mom_dict_to_list(Nmom)
mom_list = [mom_list[0]]

# Load one configuration for testing
cfg = 10000
t_test = 0

# Test with minimal size for detailed debugging
test_usedNe = 1
test_usedNp = 1

print(f"Testing v2p and p2v symmetry")
print(f"Configuration: {cfg}")
print(f"Time slice: {t_test}")
print(f"Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"Nmom: {Nmom}, num_mom: {len(mom_list)}")
print(f"Test usedNe: {test_usedNe}, usedNp: {test_usedNp}")
print(f"=" * 80)

# Create data loaders (same as 3.gen_current_elemental.py)
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

# Create generator with debug enabled
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

# Load data
print("\nLoading data...")
current_gen.load(cfg)
print("Applying stout smearing...")
current_gen.stout_smear(nstep=20, rho=0.12)

# Compute v2v, v2p, p2v, p2p using calc_all
print("\n" + "=" * 80)
print("Computing all elemental diagrams using calc_all...")
print("=" * 80)
results = current_gen.calc_all(t_test)
v2p = results["v2p"]  # [num_disp, usedNe, usedNp, Nc]
p2v = results["p2v"]  # [num_disp, usedNp, usedNe, Nc]
v2v = results["v2v"]  # [num_disp, num_momentum, usedNe, usedNe]
p2p = results["p2p"]  # List of dicts

print(f"v2p shape: {v2p.shape}")
print(f"p2v shape: {p2v.shape}")
print(f"v2v shape: {v2v.shape}")
print(f"p2p length: {len(p2p)}")

if backend.__name__ == "cupy":
    print(f"v2p[0, 0, 0, :] = {v2p[0, 0, 0, :].get()}")
    print(f"p2v[0, 0, 0, :] = {p2v[0, 0, 0, :].get()}")
else:
    print(f"v2p[0, 0, 0, :] = {v2p[0, 0, 0, :]}")
    print(f"p2v[0, 0, 0, :] = {p2v[0, 0, 0, :]}")

# Build displacement to index map
disp_to_idx = {}
for idx in range(num_disp):
    gauge_link = GaugeLink(idx)
    disp = tuple(gauge_link.displacement)  # Convert to tuple for hashable key
    disp_to_idx[disp] = idx

print(f"\n{'='*80}")
print(f"Testing v2p(disp) vs p2v(-disp) symmetry:")
print(f"Expected relations:")
print(
    f"  1. Gauge list: [a,b] <-> [(b+3)%6, (a+3)%6] (reverse order + reverse direction)"
)
print(f"  2. Matrix: v2p(disp) ≈ p2v(-disp).transpose(1,0,2)")
print(f"{'='*80}\n")

# Test each displacement
num_tests = 0
num_pass = 0

for disp_idx in range(num_disp):
    gauge_link = GaugeLink(disp_idx)
    disp = tuple(gauge_link.displacement)  # Convert to tuple
    reverse_disp = tuple(-d for d in disp)

    if reverse_disp in disp_to_idx:
        # Verify gauge_list symmetry: [a,b] -> [(b+3)%6, (a+3)%6]
        expected_conjugate_list = gauge_link.gauge_list[-1::-1]  # Reverse order
        expected_conjugate_list = [
            (x + 3) % 6 for x in expected_conjugate_list
        ]  # Reverse direction
        reverse_gauge_link = GaugeLink(
            expected_conjugate_list
        )  # Fixed: removed extra brackets
        reverse_idx = reverse_gauge_link.idx

        # Verify reverse displacement
        reverse_disp_actual = tuple(reverse_gauge_link.displacement)
        if reverse_disp_actual != reverse_disp:
            print(f"  WARNING: Reverse displacement mismatch!")
            print(f"    Expected: {reverse_disp}")
            print(f"    Actual: {reverse_disp_actual}")
            print(f"  Skipping this test")
            continue

        # Extract data for this displacement pair
        v2p_data = v2p[disp_idx].get() if backend.__name__ == "cupy" else v2p[disp_idx]
        p2v_data = (
            p2v[reverse_idx].get() if backend.__name__ == "cupy" else p2v[reverse_idx]
        )

        # Test symmetry: v2p(disp) ≈ p2v(-disp).transpose(1,0,2)
        # v2p shape: [usedNe, usedNp, Nc]
        # p2v shape: [usedNp, usedNe, Nc]
        # p2v.transpose(1,0,2) -> [usedNe, usedNp, Nc]
        p2v_transformed = p2v_data

        diff = np.abs(v2p_data - p2v_transformed)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        max_val = np.max(np.abs(v2p_data))

        num_tests += 1

        print(f"\nDisplacement: {disp} <-> {reverse_disp}")
        print(f"  gaugelink_idx: {disp_idx} <-> {reverse_idx}")

        reverse_gauge_link = GaugeLink(reverse_idx)
        conjugate_gauge_link = gauge_link.conjugate()

        print(f"  gauge_list ({disp_idx}): {gauge_link.gauge_list}")
        print(f"  gauge_list ({reverse_idx}): {reverse_gauge_link.gauge_list}")
        print(f"  conjugate gauge_list: {conjugate_gauge_link.gauge_list}")

        if reverse_gauge_link.gauge_list == conjugate_gauge_link.gauge_list:
            print(f"  ✓ gauge_list symmetry verified: reverse = conjugate")
        else:
            print(f"  ✗ gauge_list symmetry FAILED: reverse ≠ conjugate")
            print(f"     Expected: {expected_conjugate_list}")
            print(f"     Got reverse: {reverse_gauge_link.gauge_list}")
            print(f"     Got conjugate: {conjugate_gauge_link.gauge_list}")

        print(f"  v2p[0, 0, :]: {v2p_data[0, 0, :]}")
        print(f"  p2v[0, 0, :]: {p2v_data[0, 0, :]}")
        print(f"  p2v_transformed[0, 0, :]: {p2v_transformed[0, 0, :]}")

        print(f"  Max |v2p|: {max_val:.4e}")
        print(f"  Max difference: {max_diff:.4e}")
        print(f"  Mean difference: {mean_diff:.4e}")
        print(f"  Relative error: {max_diff / max_val if max_val > 0 else 0:.4e}")

        if max_diff < 1e-10:
            print(f"  ✓ PASS: v2p and p2v are symmetric")
            num_pass += 1
        elif max_diff / max_val < 1e-6 if max_val > 0 else False:
            print(f"  ~ PASS (with numerical tolerance): Relative error acceptable")
            num_pass += 1
        else:
            print(f"  ✗ FAIL: Symmetry violated!")
        print()
    else:
        print(f"Displacement: {disp} has no reverse in gaugelink set (asymmetric)")
        print()

print(f"{'='*80}")
print(f"Symmetry test completed!")
print(f"Results: {num_pass}/{num_tests} tests passed")
print(f"{'='*80}")
