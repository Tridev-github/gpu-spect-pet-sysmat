#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib.pyplot as plt

def read_f32(path):
    return np.fromfile(path, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sysmat", required=True)
    ap.add_argument("--param_detector", required=True)
    ap.add_argument("--param_image", required=True)
    ap.add_argument("--det_id", type=int, default=0)
    ap.add_argument("--out_png", default="ppdf_xz.png")
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    img = read_f32(args.param_image)
    nx, ny, nz = map(int, img[:3])
    if ny != 1:
        raise RuntimeError(f"This viewer expects NY=1 (2D XZ plane). Got (nx,ny,nz)=({nx},{ny},{nz})")

    det = read_f32(args.param_detector)
    num_det = int(det[0])

    vox = nx * ny * nz
    sm = read_f32(args.sysmat)
    expected = num_det * vox
    if sm.size < expected:
        raise RuntimeError(f"Sysmat too small: got {sm.size} floats, expected {expected} (num_det={num_det}, vox={vox})")

    if not (0 <= args.det_id < num_det):
        raise ValueError("det_id out of range")

    start = args.det_id * vox
    p = sm[start : start + vox]

    # reshape possibilities
    a1 = p.reshape(nz, nx)     # Z rows, X cols
    a2 = p.reshape(nx, nz).T   # also Z rows, X cols but swapped layout

    # pick the one with more structure (higher variance)
    A = a1 if a1.var() >= a2.var() else a2

    stats = dict(min=float(A.min()), max=float(A.max()), nnz=int((A != 0).sum()))
    print(f"PPDF stats det{args.det_id}: min={stats['min']:.3e} max={stats['max']:.3e} nnz={stats['nnz']} shape={A.shape}")

    plt.figure(figsize=(7,6))
    show = np.log10(A + 1e-30) if args.log else A
    plt.imshow(show, origin="lower", aspect="equal")
    plt.colorbar(label=("log10(PPDF+eps)" if args.log else "PPDF"))
    plt.title(f"PPDF det={args.det_id} (XZ plane)  nx={nx} nz={nz}")
    plt.xlabel("X index")
    plt.ylabel("Z index")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Wrote {args.out_png}")

if __name__ == "__main__":
    main()
