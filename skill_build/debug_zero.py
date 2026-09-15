"""Debug the zero-line vertex failure: find which correlator element / flavor
term produces an adjacency matrix row of all zeros."""
import sys, traceback
sys.path.insert(0, ".")
from lattice.backend import set_backend
set_backend("numpy")
import lattice.quark_diagram as qd
from lattice.base_types import Tag
from lattice.flavor_structure import HadronFlavorStructure
from lattice.spatial_structure import HadronIrrep
from lattice.group_projection import hadron_little_group_projection
from lattice.hadron import Hadron, gen_correlator
from lattice.symmetry.sympy_utils import convert_pow_to_mul
import sympy as sp

orig_validate = qd.validate_adjacency_matrix
def spy(adj):
    try:
        orig_validate(adj)
    except ValueError as e:
        print("FAIL adjacency:", adj)
        raise
qd.validate_adjacency_matrix = spy

orig_contract = qd.quark_contract
def spy_contract(expr, particles, degenerate=True):
    terms = sp.Add.make_args(convert_pow_to_mul(expr.expand()))
    print(f"quark_contract: n_particles={len(particles)} n_terms={len(terms)} expr={expr}")
    for t in terms:
        from sympy import Mul
        nfs = sum(1 for f in Mul.make_args(t) if isinstance(f, HadronFlavorStructure))
        npr = sum(1 for f in Mul.make_args(t) if isinstance(f, qd.Propagator))
        print(f"   term: nfs={nfs} expr={t}")
    return orig_contract(expr, particles, degenerate)
qd.quark_contract = spy_contract
# gen_correlator imported quark_contract by name; patch the reference it uses
import lattice.hadron as hd
hd.quark_contract = spy_contract

tag = Tag(0, 0)
rho = HadronIrrep("rho", [0, 0, 0], "T_1", -1, tag)
rows = hadron_little_group_projection([rho, rho], "E", 0, parity=None)
print("n rows:", len(rows))
for r in rows:
    print("  row:", r)
pi = HadronIrrep("pi", [0, 0, 0], "A_1", -1, tag)[0]
flavor1 = HadronFlavorStructure("ud")
flavor2 = HadronFlavorStructure("ud") * HadronFlavorStructure("ud")

hadrons1 = [Hadron(pi, flavor1)]
hadrons2 = [Hadron(r, flavor2) for r in rows]
try:
    c = gen_correlator([hadrons1, hadrons2])
    print("cross 1x2 OK", c.shape)
except ValueError as e:
    print("cross 1x2 FAILED:", e)
try:
    c2 = gen_correlator([hadrons2, hadrons2])
    print("2x2 OK", c2.shape)
except ValueError as e:
    print("2x2 FAILED:", e)
