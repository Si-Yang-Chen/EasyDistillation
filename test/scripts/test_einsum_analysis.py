import numpy as np
from lattice import set_backend, get_backend

set_backend("numpy")  # Use numpy for clarity
backend = get_backend()

# Simple test case
Ne = 2
Lz, Ly, Lx = 3, 3, 3
Nc = 3

# Create simple test data
V = backend.random.randn(Ne, Lz, Ly, Lx, Nc) + 1j * backend.random.randn(
    Ne, Lz, Ly, Lx, Nc
)
U = backend.random.randn(Lz, Ly, Lx, Nc, Nc) + 1j * backend.random.randn(
    Lz, Ly, Lx, Nc, Nc
)
phase = backend.ones((Lz, Ly, Lx))

print("Testing einsum formulas")
print(f"V shape: {V.shape}")
print(f"U shape: {U.shape}")
print(f"phase shape: {phase.shape}")
print("=" * 80)

# Method 1: calc_disp style
# right = U @ Vf
# Vf = roll(V, -1, 3) means shift in x direction
Vf = backend.roll(V, -1, 3)  # [Ne, Lz, Ly, Lx, Nc]
print(f"\nVf shape: {Vf.shape}")

# Contract U @ Vf: U[z,y,x,a,b] @ Vf[e,z,y,x,b] -> right[e,z,y,x,a]
right = backend.einsum("zyxab,ezyxb->ezyxa", U, Vf)
print(f"right shape (U @ Vf): {right.shape}")

# Final contraction: <V | right>
result1 = backend.einsum("zyx,ezyxc,fzyxc->ef", phase, V.conj(), right)
print(f"result1 shape: {result1.shape}")
print(f"result1[0,0] = {result1[0,0]}")

# Method 2: calc_v2v style (current implementation)
# shift_V = roll(V, -1, 3)
# result = <V | U | shift_V>
shift_V = backend.roll(V, -1, 3)
print(f"\nshift_V shape: {shift_V.shape}")

result2 = backend.einsum("zyx,ezyxa,zyxac,fzyxc->ef", phase, V.conj(), U, shift_V)
print(f"result2 shape: {result2.shape}")
print(f"result2[0,0] = {result2[0,0]}")

# Method 3: Alternative - V @ U @ shift_V
# First: V_via_U = V @ U
V_via_U = backend.einsum("ezyxb,zyxbc->ezyxc", V, U)
print(f"\nV_via_U shape: {V_via_U.shape}")

result3 = backend.einsum("zyx,ezyxc,fzyxc->ef", phase, V.conj(), V_via_U)
print(f"result3 shape (without shift_V): {result3.shape}")
print(f"result3[0,0] = {result3[0,0]}")

# Method 4: Correct - should use shift_V properly
# The issue is: right = U @ Vf means U acts on the shifted V
# So we should have: result = <V | U @ shift_V>
V_right = backend.einsum("zyxab,fzyxb->fzyxa", U, shift_V)
result4 = backend.einsum("zyx,ezyxc,fzyxc->ef", phase, V.conj(), V_right)
print(f"\nresult4 shape (U @ shift_V): {result4.shape}")
print(f"result4[0,0] = {result4[0,0]}")

# Compare all methods
print(f"\n" + "=" * 80)
print(f"Comparison:")
print(f"  result1 (calc_disp style): {result1[0,0]}")
print(f"  result2 (calc_v2v current): {result2[0,0]}")
print(f"  result3 (V @ U, no shift): {result3[0,0]}")
print(f"  result4 (U @ shift_V): {result4[0,0]}")
print(f"\n  |result1 - result2|: {np.abs(result1[0,0] - result2[0,0]):.4e}")
print(f"  |result1 - result3|: {np.abs(result1[0,0] - result3[0,0]):.4e}")
print(f"  |result1 - result4|: {np.abs(result1[0,0] - result4[0,0]):.4e}")

if np.abs(result1[0, 0] - result4[0, 0]) < 1e-10:
    print(f"\n✓ result4 matches result1! Use: V_right = U @ shift_V, then contract")
if np.abs(result1[0, 0] - result2[0, 0]) < 1e-10:
    print(f"\n✓ result2 matches result1! Current einsum formula is correct")
