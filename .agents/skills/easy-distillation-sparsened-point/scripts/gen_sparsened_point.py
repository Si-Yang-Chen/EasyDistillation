"""Generate and validate a sparse point-source coordinate table."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lattice.generator.sparsened_point import generate_sparsened_points


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latt-size", nargs=4, type=int, required=True, metavar=("LX", "LY", "LZ", "LT"))
    p.add_argument("--num-points", type=int, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--temporal-policy", choices=("independent", "same"), default="independent")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.out}; pass --overwrite after approval")
    latt = args.latt_size
    points = generate_sparsened_points(latt, args.num_points, args.seed, args.temporal_policy)
    lx, ly, lz, lt = latt
    assert points.shape == (args.num_points, lt, 3) and points.dtype == np.int32
    assert np.all(points[:, :, 0] < lx) and np.all(points[:, :, 1] < ly) and np.all(points[:, :, 2] < lz)
    for t in range(lt):
        assert len({tuple(x) for x in points[:, t, :]}) == args.num_points
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, points)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = {"latt_size": latt, "num_points": args.num_points, "seed": args.seed,
            "temporal_policy": args.temporal_policy,
            "shape": list(points.shape), "dtype": str(points.dtype), "sha256": digest}
    args.out.with_suffix(args.out.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.out} shape={points.shape} dtype={points.dtype}")


if __name__ == "__main__":
    main()
