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

print(f"Detailed comparison of ElementalGenerator vs CurrentElementalGenerator")
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

# Test 1: ElementalGenerator
print("\n1. ElementalGenerator")
print("-" * 80)
elemental_gen = ElementalGenerator(
    latt_size=latt_size,
    gauge_field=gauge_field,
    eigenvector=eigenvector,
    num_nabla=Ndisp,
    momentum_list=mom_list,
    usedNe=test_usedNe,
    debug=False,
)

elemental_gen.load(cfg)
elemental_gen.stout_smear(nstep=20, rho=0.12)

print(f"  _U shape: {elemental_gen._U.shape}")
print(f"  _eigenvector_data shape: {elemental_gen._eigenvector_data.shape}")
print(f"  _V shape: {elemental_gen._V.shape}")

# Check _V initialization
for e in range(test_usedNe):
    elemental_gen._V[e] = elemental_gen._eigenvector_data[t_test, e]

if backend.__name__ == "cupy":
    V_elem = elemental_gen._V[:test_usedNe].get()
    U_elem = elemental_gen._U[:, t_test].get()
    eig_elem = elemental_gen._eigenvector_data[t_test, :test_usedNe].get()
else:
    V_elem = elemental_gen._V[:test_usedNe]
    U_elem = elemental_gen._U[:, t_test]
    eig_elem = elemental_gen._eigenvector_data[t_test, :test_usedNe]

print(f"  V (from _V) [0, 0, 0, 0, :] = {V_elem[0, 0, 0, 0, :]}")
print(f"  eigenvector_data [0, 0, 0, 0, :] = {eig_elem[0, 0, 0, 0, :]}")
print(f"  U[0] [0, 0, 0, 0, 0] = {U_elem[0, 0, 0, 0, 0, 0]}")

# Test 2: CurrentElementalGenerator
print("\n2. CurrentElementalGenerator")
print("-" * 80)
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

current_gen.load(cfg)
current_gen.stout_smear(nstep=20, rho=0.12)

print(f"  _U shape: {current_gen._U.shape}")
print(f"  _eigenvector_data shape: {current_gen._eigenvector_data.shape}")

if backend.__name__ == "cupy":
    V_current = current_gen._eigenvector_data[t_test, :test_usedNe].get()
    U_current = current_gen._U[:, t_test].get()
else:
    V_current = current_gen._eigenvector_data[t_test, :test_usedNe]
    U_current = current_gen._U[:, t_test]

print(f"  V (from eigenvector_data) [0, 0, 0, 0, :] = {V_current[0, 0, 0, 0, :]}")
print(f"  U[0] [0, 0, 0, 0, 0] = {U_current[0, 0, 0, 0, 0, 0]}")

# Compare loaded data
print("\n3. Data Comparison")
print("-" * 80)
print(f"  V match: {np.allclose(V_elem, V_current)}")
print(f"  U match: {np.allclose(U_elem, U_current)}")

if not np.allclose(V_elem, V_current):
    print(f"  V difference: max = {np.max(np.abs(V_elem - V_current)):.4e}")
    print(f"  V_elem dtype: {V_elem.dtype}")
    print(f"  V_current dtype: {V_current.dtype}")

if not np.allclose(U_elem, U_current):
    print(f"  U difference: max = {np.max(np.abs(U_elem - U_current)):.4e}")
    print(f"  U_elem dtype: {U_elem.dtype}")
    print(f"  U_current dtype: {U_current.dtype}")

# Test one displacement manually
print("\n4. Manual Displacement Test (disp_idx=1, gauge_list=[0])")
print("-" * 80)

# ElementalGenerator approach
left_elem = V_elem.copy()
right_elem = V_elem.copy()

Vf_elem = np.roll(right_elem, -1, 3)  # roll in x direction
print(f"  Vf_elem[0, 0, 0, 0, :] = {Vf_elem[0, 0, 0, 0, :]}")

right_elem = np.einsum("zyxab,ezyxb->ezyxa", U_elem[0], Vf_elem)
print(f"  right_elem[0, 0, 0, 0, :] = {right_elem[0, 0, 0, 0, :]}")

# CurrentElementalGenerator approach
left_current = V_current.copy()
right_current = V_current.copy()

Vf_current = np.roll(right_current, -1, 3)
print(f"  Vf_current[0, 0, 0, 0, :] = {Vf_current[0, 0, 0, 0, :]}")

right_current = np.einsum("zyxab,ezyxb->ezyxa", U_current[0], Vf_current)
print(f"  right_current[0, 0, 0, 0, :] = {right_current[0, 0, 0, 0, :]}")

# Compare
print(
    f"\n  right_elem vs right_current match: {np.allclose(right_elem, right_current)}"
)
if not np.allclose(right_elem, right_current):
    print(f"  Max difference: {np.max(np.abs(right_elem - right_current)):.4e}")

print(f"\n{'='*80}")
print(f"Analysis completed!")
