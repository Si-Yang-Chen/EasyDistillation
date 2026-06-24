import os
import sys
import numpy as np
import gvar as gv


from lattice import set_backend, get_backend

from lattice import (
    PerambulatorGenerator,
    PerambulatorNpy,
    PointSourceNpy,
    PropagatorPSV,
    PropagatorPSVNpy,
)

# from pyquda import enum_quda,init
# from pyquda_utils import core

set_backend("cupy")
backend = get_backend()
print("Using backend:", backend.__name__)


def log_gpu_memory(tag: str) -> None:
    """Log current GPU memory usage."""
    try:
        free_bytes, total_bytes = backend.cuda.runtime.memGetInfo()
        used_bytes = total_bytes - free_bytes
        print(
            f"[GPU MEM][{tag}] used={used_bytes / 1024 ** 3:.3f}GB "
            f"free={free_bytes / 1024 ** 3:.3f}GB total={total_bytes / 1024 ** 3:.3f}GB"
        )
    except Exception as err:
        print(f"[GPU MEM][{tag}] query failed: {err}")


from lattice.insertion import (
    Insertion,
    InsertionGaugeLink,
    Operator,
    GammaName,
    DerivativeName,
    ProjectionName,
)
from lattice.insertion.mom_dict import momDict_test

from lattice import (
    GaugeFieldIldg,
    EigenvectorNpy,
    Nc,
    Nd,
    PerambulatorNpy,
    ElementalNpy,
    PerambulatorTimeslicesNpy,
    PropagatorPSVTimeslicesNpy,
    OverlapMatrixNpy,
    CurrentElementalV2P,
    CurrentElementalP2V,
    CurrentElementalP2P,
)
from lattice import Dispatch

from lattice.quark_diagram import (
    Meson,
    Current,
    QuarkDiagram,
    compute_diagrams_multitime,
    Propagator,
    PropagatorWithCurrent,
)
from lattice.symmetry.hardcoded_rep import gauge_link

log_gpu_memory("init")
L = 24
T = 72
latt_size = [L, L, L, T]
Lx, Ly, Lz, Lt = latt_size
Np = 6**3
Ne = 128
grid_size = [1, 1, 1, 1]
# init(grid_size, backend="cupy", resource_path="/public/home/siyangchen/.quda_cache")

# core.init(grid_size, latt_size, resource_path="/public/home/siyangchen/.quda_cache")
# latt_info = core.LatticeInfo([L, L, L, T], 1, 1.0)
multigrid = [[6, 6, 6, 3], [2, 2, 2, 3]]

# Setup cfg list and dispatcher (similar to 0.gen_laplace_spectrum.py)
cfg_list = np.arange(10000, 48001, 1000)
dispatcher = Dispatch("/public/home/siyangchen/qedinf/cfg_list.txt", "cont")
dispatcher = [10000, 13000, 14000, 15000, 16000, 17000, 18000, 19000]
# Create output directory if it doesn't exist
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

elemental = ElementalNpy(
    f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/03.elemental.ndisp1.np0.nev128/",
    ".npy",
    [Lt, Ne, Ne],
    Ne,
)

# Current elemental parameters (must match 3.gen_current_elemental_all.py)
num_nabla = 1  # Displacement degree (Ndisp)
num_momentum = 0  # Momentum level (Nmom) - must match production script
# Calculate num_disp from GaugeLink
from lattice.insertion.gauge_link import GaugeLink

num_disp = list(GaugeLink.nmax_generator(num_nabla))[-1]

# Create current elemental data loaders
base_dir = f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}"
elemental_dir = f"{base_dir}/03.current_elemental_all.ndisp{num_nabla}.nmom{num_momentum}.nev{Ne}.np{Np}/"

v2p_data = CurrentElementalV2P(
    elemental_dir,
    "_v2p.npy",  # New format: single file per configuration
    [Lt, num_disp, Ne, Np, Nc],  # Shape without momentum dimension
    Ne,
    Np,
)

p2v_data = CurrentElementalP2V(
    elemental_dir,
    "_p2v.npy",  # New format: single file per configuration
    [Lt, num_disp, Np, Nc, Ne],  # Shape without momentum dimension
    Ne,
    Np,
)

p2p_data = CurrentElementalP2P(
    elemental_dir,
    None,  # Use default HDF5 format (.h5)
)

dir_VSV = f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/04.perambulator.nev{Ne}_to_nev{Ne}/"
dir_PSV = f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/04.perambulator.nev{Ne}_to_np{Np}/"

# Create overlap matrix loader
overlap_matrix = OverlapMatrixNpy(
    f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/03.overlap_matrix/",
    ".overlap_matrix.npy",
    [Lt, Ne, Np, Nc],
    Ne,
    Np,
)

# Use PropagatorWithCurrent to support all four propagator types
# VSV and PSV are available, VSP and PSP can be added when data becomes available
perambulator_with_current = PropagatorWithCurrent(
    vsv=PerambulatorTimeslicesNpy(dir_VSV, ".t???.npy", [Lt, Ne, Ne], Ne),
    vsp=None,  # Not available yet
    psv=PropagatorPSVTimeslicesNpy(dir_PSV, ".t???.npy", [Lt, Ne, Np, Nc], Np, Ne),
    psp=None,  # Not available yet
    overlap_matrix=overlap_matrix,
    Lt=Lt,
    debug=True,
)
log_gpu_memory("after init perambulator_with_current")
# Keep old variable name for compatibility
prob = perambulator_with_current

ins = Insertion(GammaName.PI, DerivativeName.IDEN, ProjectionName.A1, momDict_test)
# ins_0 = InsertionGaugeLink(GammaName.PI, 'A_1g+',0, 'A_1u+', momDict_test, gauge_link)

ins_meson = InsertionGaugeLink(
    GammaName.RHO, "A_1g+", 0, "T_1", momDict_test, gauge_link
)
ins_current = InsertionGaugeLink(
    GammaName.A0, "T_1u-", 0, "T_1", momDict_test, gauge_link
)
print(ins_meson[0])
print(ins_current[0])
print(ins_current.rows[0])
op_meson = Operator("rho", [ins_meson[0](0, 0, 0)], [1])
op_current = Operator("v", [ins_current[0](0, 0, 0)], [1])
meson = Meson(elemental, op_meson, False)
meson2 = Meson(elemental, op_current, False)
current = Current(
    elemental,
    op_current,
    True,
    # v2p_data=v2p_data,
    p2v_data=p2v_data,
    p2p_data=p2p_data,
    debug=True,
)
log_gpu_memory("after init meson, current")
# Create QuarkDiagram with vertex_list to support current vertex
# vertex_list: [0, 1] means vertex[0] is normal (Meson), vertex[1] is current (Current)
diagram = QuarkDiagram([[0, 1], [1, 0]], vertex_list=[0, 1], debug=True,usedNp=Np,L=L)
exit()
output = np.zeros((Lt, Lt), dtype=np.complex128)

# compute_diagrams_multitime will auto-expand if diagram has expanded_diagrams
elemental_dir = f"/public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L{L}x{T}/05.correlator.current.nonlocal/"
os.makedirs(elemental_dir, exist_ok=True)
combine_result = []
log_gpu_memory("before loop")
for cfg in dispatcher:
    if os.path.exists(f"{elemental_dir}/{cfg}.npy"):
        print(f"Skipping configuration {cfg} as output file already exists.")
        combine_result.append(np.average(np.load(f"{elemental_dir}/{cfg}.npy"), axis=0))
        continue
    print(f"Processing configuration: {cfg}")
    log_gpu_memory(f"before load cfg={cfg}")
    with prob, meson, current:
        print("load perambulators, meson, current")
        prob.load(cfg, usedNe=20)
        log_gpu_memory(f"after load perambulators cfg={cfg}")
        # PSV_perambulator.load(cfg)
        meson.load(cfg, usedNe=20)
        log_gpu_memory(f"after load meson cfg={cfg}")
        # meson2.load(cfg)
        current.load(cfg, usedNe=20, usedNp=100)
        log_gpu_memory(f"after load current cfg={cfg}")
        log_gpu_memory(f"after load cfg={cfg}")
        for t in range(Lt):
            print(f"  Processing time slice: {t}")
            # Compute diagram (auto-expands and sums internally)
            result = compute_diagrams_multitime(
                [diagram],
                [t, np.arange(Lt)],
                [meson, current],
                [None, prob],
                multitime_shape=True,
                debug=True,
            )
            # result shape: [num_expanded_diagrams, Lt]
            # Sum all expanded diagrams
            output[t] = np.roll(backend.sum(result, axis=0).get(), -t, axis=0)
            if (t + 1) % 8 == 0:
                log_gpu_memory(f"after time slice {t} cfg={cfg}")
        np.save(f"{elemental_dir}/{cfg}.npy", output)
        combine_result.append(np.average(output, axis=0))
        log_gpu_memory(f"after save cfg={cfg}")
    log_gpu_memory(f"after release cfg={cfg}")
combine_result = gv.dataset.avg_data(np.array(combine_result))

print(combine_result)
print(
    print(
        gv.arccosh(
            (combine_result[2:] + combine_result[:-2]) / (2 * combine_result[1:-1])
        )
    )
)
