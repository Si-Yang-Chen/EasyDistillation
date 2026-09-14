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
from opt_einsum import contract

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
disp_idx_test = 1  # Test displacement (1, 0, 0)

print(f"Step-by-step comparison for disp_idx={disp_idx_test}")
print(f"="*80)

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

# Load both generators
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

# Get data
if backend.__name__ == "cupy":
    V_elem = elemental_gen._eigenvector_data[t_test, :test_usedNe].get()
    V_current = current_gen._eigenvector_data[t_test, :test_usedNe].get()
    U_elem = elemental_gen._U[:, t_test].get()
    U_current = current_gen._U[:, t_test].get()
    phase_elem = elemental_gen._momentum_phase.get(mom_list[0]).get()
    phase_current = current_gen._momentum_phase.get(mom_list[0]).get()
else:
    V_elem = elemental_gen._eigenvector_data[t_test, :test_usedNe]
    V_current = current_gen._eigenvector_data[t_test, :test_usedNe]
    U_elem = elemental_gen._U[:, t_test]
    U_current = current_gen._U[:, t_test]
    phase_elem = elemental_gen._momentum_phase.get(mom_list[0])
    phase_current = current_gen._momentum_phase.get(mom_list[0])

print(f"\n1. Input Data")
print(f"  V match: {np.allclose(V_elem, V_current)}")
print(f"  U match: {np.allclose(U_elem, U_current)}")
print(f"  phase match: {np.allclose(phase_elem, phase_current)}")

# Manually compute for disp_idx=1
gauge_link = GaugeLink(disp_idx_test)
gauge_list = gauge_link.gauge_list
disp = tuple(gauge_link.displacement)
displacement = tuple([d / 2 for d in disp])

print(f"\n2. Displacement Info")
print(f"  disp: {disp}")
print(f"  displacement: {displacement}")
print(f"  gauge_list: {gauge_list}")

# ElementalGenerator.calc_disp logic
print(f"\n3. ElementalGenerator.calc_disp logic")
left_elem = V_elem.copy()
right_elem = V_elem.copy()

for d in gauge_list:
    print(f"  Applying d={d}")
    if d < 3:
        Vf = np.roll(right_elem, -1, 3 - d)
        right_elem = np.einsum("zyxab,ezyxb->ezyxa", U_elem[d], Vf)
    else:
        Vd = np.einsum("zyxba,ezyxb->ezyxa", np.conj(U_elem[d - 3]), right_elem)
        right_elem = np.roll(Vd, 1, 6 - d)
    print(f"    right[0, 0, 0, 0, :] = {right_elem[0, 0, 0, 0, :]}")

momentum = mom_list[0]
disp_phase_elem = np.exp(sum([-1j*momentum[i] * displacement[i] for i in range(3)]))
result_elem = disp_phase_elem * np.einsum(
    "zyx,ezyxc,fzyxc->ef",
    phase_elem,
    np.conj(left_elem),
    right_elem
)

print(f"  disp_phase: {disp_phase_elem}")
print(f"  result[0,0]: {result_elem[0,0]}")

# CurrentElementalGenerator.calc_v2v logic
print(f"\n4. CurrentElementalGenerator.calc_v2v logic")
left_current = V_current.copy()
right_current = V_current.copy()

for d in gauge_list:
    print(f"  Applying d={d}")
    if d < 3:
        Vf = np.roll(right_current, -1, 3 - d)
        right_current = np.einsum("zyxab,ezyxb->ezyxa", U_current[d], Vf)
    else:
        Vd = np.einsum("zyxba,ezyxb->ezyxa", np.conj(U_current[d - 3]), right_current)
        right_current = np.roll(Vd, 1, 6 - d)
    print(f"    right[0, 0, 0, 0, :] = {right_current[0, 0, 0, 0, :]}")

disp_phase_current = np.exp(sum([-1j*momentum[i] * displacement[i] for i in range(3)]))
result_current = disp_phase_current * np.einsum(
    "zyx,ezyxc,fzyxc->ef",
    phase_current,
    np.conj(left_current),
    right_current
)

print(f"  disp_phase: {disp_phase_current}")
print(f"  result[0,0]: {result_current[0,0]}")

# Final comparison
print(f"\n5. Final Comparison")
print(f"="*80)
print(f"  left match: {np.allclose(left_elem, left_current)}")
print(f"  right match: {np.allclose(right_elem, right_current)}")
print(f"  disp_phase match: {np.allclose(disp_phase_elem, disp_phase_current)}")
print(f"  result match: {np.allclose(result_elem[0,0], result_current[0,0])}")

if not np.allclose(result_elem[0,0], result_current[0,0]):
    print(f"\n  ✗ Results differ!")
    print(f"  result_elem[0,0]: {result_elem[0,0]}")
    print(f"  result_current[0,0]: {result_current[0,0]}")
    print(f"  Difference: {result_elem[0,0] - result_current[0,0]}")
    print(f"  Relative error: {np.abs(result_elem[0,0] - result_current[0,0]) / np.abs(result_elem[0,0]):.4e}")
else:
    print(f"\n  ✓ Results match perfectly!")

print(f"\n{'='*80}")

