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
from lattice.generator.elemental import ElementalGenerator, CurrentElementalGenerator
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

num_disp = list(GaugeLink.nmax_generator(Ndisp))[-1]
mom_list = mom_dict_to_list(Nmom)[:1]

cfg = 10000
t_test = 0
test_usedNe = 1

print(f"Analyzing _disp vs _gauge_links_product")
print(f"Configuration: {cfg}, Time slice: {t_test}")
print(f"Ndisp: {Ndisp}, num_disp: {num_disp}")
print(f"Test usedNe: {test_usedNe}")
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

# Create generators
elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    debug=False,
)

current_gen = CurrentElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    point=sparsen_point,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    usedNp=1,
    debug=False,
)

# Load data
print("\nLoading data...")
elemental_gen.load(cfg)
elemental_gen.stout_smear(nstep=20, rho=0.12)

current_gen.load(cfg)
current_gen.stout_smear(nstep=20, rho=0.12)

# Get eigenvector
V_elem = backend.zeros((test_usedNe, Lz, Ly, Lx, Nc), "<c8")
for e in range(test_usedNe):
    V_elem[e] = elemental_gen._eigenvector_data[t_test, e]

V_current = current_gen._eigenvector_data[t_test, :test_usedNe]

print(f"\nV_elem shape: {V_elem.shape}")
print(f"V_current shape: {V_current.shape}")

# Test each non-zero displacement
for disp_idx in range(1, min(3, num_disp)):  # Skip identity (disp_idx=0)
    gauge_link = GaugeLink(disp_idx)
    gauge_list = gauge_link.gauge_list
    disp = tuple(gauge_link.displacement)

    print(f"\n{'='*80}")
    print(f"Displacement {disp_idx}: {disp}")
    print(f"gauge_list: {gauge_list}")
    print(f"{'='*80}")

    # Method 1: Apply _disp step by step (ElementalGenerator method)
    print("\nMethod 1: Applying _disp step by step")
    V_right = V_elem.copy()
    for i, d in enumerate(gauge_list):
        print(f"  Step {i}: Applying direction {d}")
        print(f"    Before: V_right[0, 0, 0, 0, :] = {V_right[0, 0, 0, 0, :]}")
        V_right = elemental_gen._disp(V_right, elemental_gen._U[:, t_test], d)
        print(f"    After:  V_right[0, 0, 0, 0, :] = {V_right[0, 0, 0, 0, :]}")

    print(f"\n  Final V_right[0, 0, 0, 0, :] = {V_right[0, 0, 0, 0, :]}")

    # Method 2: Use _gauge_links_product (CurrentElementalGenerator method)
    print("\nMethod 2: Using _gauge_links_product")
    gauge_product = current_gen._gauge_links_product(gauge_list, t=t_test)

    if gauge_product is not None:
        print(f"  gauge_product shape: {gauge_product.shape}")
        print(f"  gauge_product[0, 0, 0, 0, 0] = {gauge_product[0, 0, 0, 0, 0]}")

        # Apply to V: V @ gauge_product
        # V: [Ne, Lz, Ly, Lx, Nc]
        # gauge_product: [Lz, Ly, Lx, Nc, Nc]
        # Result: [Ne, Lz, Ly, Lx, Nc]
        V_via_product = backend.einsum("ezyxb,zyxbc->ezyxc", V_current, gauge_product)
        print(f"  V_via_product[0, 0, 0, 0, :] = {V_via_product[0, 0, 0, 0, :]}")

        # Compare
        if backend.__name__ == "cupy":
            V_right_cpu = V_right.get()
            V_via_product_cpu = V_via_product.get()
        else:
            V_right_cpu = V_right
            V_via_product_cpu = V_via_product

        diff = np.abs(V_right_cpu - V_via_product_cpu)
        max_diff = np.max(diff)
        print(f"\n  Comparison:")
        print(f"    Max difference: {max_diff:.4e}")
        print(f"    Sample V_right[0,0,0,0,0]: {V_right_cpu[0,0,0,0,0]}")
        print(f"    Sample V_via_product[0,0,0,0,0]: {V_via_product_cpu[0,0,0,0,0]}")

        if max_diff < 1e-10:
            print(f"    ✓ Methods produce equivalent V_right")
        else:
            print(f"    ✗ Methods produce different V_right!")
            print(f"    This explains why calc_v2v differs from calc_disp")

print(f"\n{'='*80}")
print(f"Analysis completed!")
print(f"{'='*80}")
